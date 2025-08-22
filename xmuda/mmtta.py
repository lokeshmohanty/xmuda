#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import copy
import numpy as np

import torch
import torch.nn.functional as F
import sparseconvnet as scn
from torch.utils.tensorboard import SummaryWriter
from aim.ext.tensorboard_tracker import Run


from xmuda.common.solver.build import build_optimizer, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.evaluate import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="MM-TTA training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def mmtta(cfg, output_dir="", run_name=""):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger("xmuda.train")

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info("Build 2D model:\n{}".format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print("#Parameters: {:.2e}".format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info("Build 3D model:\n{}".format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print("#Parameters: {:.2e}".format(num_params))

    slow_model_2d = copy.deepcopy(model_2d)
    slow_model_3d = copy.deepcopy(model_3d)

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    slow_modwl_2d = slow_model_2d.cuda()
    slow_modwl_3d = slow_model_3d.cuda()

    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(
        model_2d,
        save_dir=output_dir,
        logger=logger,
    )
    checkpointer_2d.load(cfg.MODEL_2D.CKPT_PATH, resume=False)
    checkpointer_3d = CheckpointerV2(
        model_3d,
        save_dir=output_dir,
        logger=logger,
    )
    checkpointer_3d.load(cfg.MODEL_3D.CKPT_PATH, resume=False)

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, "tb.{:s}".format(run_name))
        summary_writer = SummaryWriter(tb_dir)
        aim_run = Run(
            repo=cfg.TRACK.URI, 
            experiment=cfg.TRACK.EXPERIMENT,
            sync_tensorboard_log_dir=tb_dir
        )
        aim_run['config'] = cfg
        aim_run.add_tag(cfg.TRACK.RUN)
        aim_run.add_tag("test")
        for tag in cfg.TRACK.TAGS: 
            aim_run.add_tag(tag)
    else:
        summary_writer = None


    test_dataloader = build_dataloader(cfg, mode="test", domain="target")
    class_names = test_dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names)
    evaluator = Evaluator(class_names)
    set_random_seed(cfg.RNG_SEED)

    # add metrics
    # train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    train_metric_logger = MetricLogger(delimiter="  ")
    # set training mode
    model_2d.train()
    model_3d.train()
    for param in slow_model_2d.parameters():
        param.requires_grad = False
    for param in slow_model_3d.parameters():
        param.requires_grad = False
    for name, param in model_2d.named_parameters():
        if 'bn' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for layer in model_3d.modules():
        if isinstance(layer, scn.BatchNormalization):
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False

    # reset metric
    train_metric_logger.reset()


    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    end = time.time()
    for iteration, data_batch in enumerate(test_dataloader):
        data_time = time.time() - end
        # copy data from cpu to gpu
        if "SCN" in cfg.DATASET_SOURCE.TYPE and "SCN" in cfg.DATASET_TARGET.TYPE:
            data_batch["x"][1] = data_batch["x"][1].cuda()
            data_batch["img"] = data_batch["img"].cuda()
            data_batch["seg_label"] = data_batch["seg_label"].cuda()
        else:
            raise NotImplementedError("Only SCN is supported for now.")

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch)
        preds_3d = model_3d(data_batch)

        loss_2d, loss_3d = 0, 0

        if cfg.TRAIN.XMUDA.mmtta:
            # Pseudo-label generation
            tau = cfg.TRAIN.XMUDA.lambda_mmtta
            for param, slow_param in zip(model_2d.parameters(), slow_model_2d.parameters()):
                slow_param.data.copy_(tau * param.data + (1 - tau) * slow_param.data)
            for param, slow_param in zip(model_3d.parameters(), slow_model_3d.parameters()):
                slow_param.data.copy_(tau * param.data + (1 - tau) * slow_param.data)

            slow_preds_2d = slow_model_2d(data_batch)
            slow_preds_3d = slow_model_3d(data_batch)

            seg_logits_2d = 0.5 * (preds_2d["seg_logit"] + slow_preds_2d["seg_logit"])
            seg_logits_3d = 0.5 * (preds_3d["seg_logit"] + slow_preds_3d["seg_logit"])
            pl_2d = seg_logits_2d.argmax(1)
            pl_3d = seg_logits_3d.argmax(1)

            # Pseudo-label refinement
            eps = 1e-6
            klDiv = lambda x,y: F.kl_div(F.log_softmax(x, dim=1), F.softmax(y, dim=1), reduction="none").sum(1).mean() + eps
            sim = lambda x,y: 0.5 * (1 / klDiv(x,y) + 1 / klDiv(y,x))

            chi_2d = sim(preds_2d["seg_logit"], slow_preds_2d["seg_logit"])
            chi_3d = sim(preds_3d["seg_logit"], slow_preds_3d["seg_logit"])

            if cfg.TRAIN.XMUDA.hard_ensemble:
                pl = seg_logits_2d if chi_2d > chi_3d else seg_logits_3d
            else:
                pl = chi_2d * seg_logits_2d + chi_3d * seg_logits_3d
                pl /= chi_2d + chi_3d
            pl = pl.argmax(1)

            loss_2d += F.cross_entropy(seg_logits_2d, pl.type(torch.uint8), weight=class_weights)
            loss_3d += F.cross_entropy(seg_logits_3d, pl.type(torch.uint8), weight=class_weights)
            train_metric_logger.update(loss_2d=loss_2d, loss_3d=loss_3d)

        else:
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                # segmentation loss: cross entropy
                seg_loss_src_2d = F.cross_entropy(
                    preds_2d["seg_logit"], data_batch["seg_label"], weight=class_weights
                )
                seg_loss_src_3d = F.cross_entropy(
                    preds_3d["seg_logit"], data_batch["seg_label"], weight=class_weights
                )
                train_metric_logger.update(
                    seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d
                )
                loss_2d += cfg.TRAIN.XMUDA.lambda_pl * seg_loss_src_2d
                loss_3d += cfg.TRAIN.XMUDA.lambda_pl * seg_loss_src_3d

            if cfg.TRAIN.XMUDA.lambda_xm > 0:
                # cross-modal loss: KL divergence
                seg_logit_2d = (
                    preds_2d["seg_logit2"]
                    if cfg.MODEL_2D.DUAL_HEAD
                    else preds_2d["seg_logit"]
                )
                seg_logit_3d = (
                    preds_3d["seg_logit2"]
                    if cfg.MODEL_3D.DUAL_HEAD
                    else preds_3d["seg_logit"]
                )
                xm_loss_2d = (
                    F.kl_div(
                        F.log_softmax(seg_logit_2d, dim=1),
                        F.softmax(preds_3d["seg_logit"].detach(), dim=1),
                        reduction="none",
                    )
                    .sum(1)
                    .mean()
                )
                xm_loss_3d = (
                    F.kl_div(
                        F.log_softmax(seg_logit_3d, dim=1),
                        F.softmax(preds_2d["seg_logit"].detach(), dim=1),
                        reduction="none",
                    )
                    .sum(1)
                    .mean()
                )
                train_metric_logger.update(
                    xm_loss_2d=xm_loss_2d, xm_loss_3d=xm_loss_3d
                )
                loss_2d += cfg.TRAIN.XMUDA.lambda_xm * xm_loss_2d
                loss_3d += cfg.TRAIN.XMUDA.lambda_xm * xm_loss_3d

        # if cfg.TRAIN.XMUDA.lambda_minent > 0:
        #     # MinEnt
        #     minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d["seg_logit"], dim=1))
        #     minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d["seg_logit"], dim=1))
        #     train_metric_logger.update(
        #         minent_loss_trg_2d=minent_loss_trg_2d,
        #         minent_loss_trg_3d=minent_loss_trg_3d,
        #     )
        #     loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d)
        #     loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d)


        # update metric (e.g. IoU)
        # with torch.no_grad():
        #     train_metric_2d.update_dict(preds_2d, data_batch)
        #     train_metric_3d.update_dict(preds_3d, data_batch)

        # backward
        loss_2d.backward()
        loss_3d.backward()

        optimizer_2d.step()
        optimizer_3d.step()

        # get original point cloud from before voxelization
        seg_label = data_batch["orig_seg_label"]
        points_idx = data_batch["orig_points_idx"]
        # loop over batch
        left_idx = 0
        for batch_ind in range(len(seg_label)):
            curr_points_idx = points_idx[batch_ind]
            # check if all points have predictions (= all voxels inside receptive field)
            assert np.all(curr_points_idx)

            curr_seg_label = seg_label[batch_ind]
            right_idx = left_idx + curr_points_idx.sum()
            pred_label_2d = pl_2d[left_idx:right_idx]
            pred_label_3d = pl_3d[left_idx:right_idx]
            pred_label = pl[left_idx:right_idx]

            # evaluate
            evaluator_2d.update(pred_label_2d.cpu(), curr_seg_label)
            evaluator_3d.update(pred_label_3d.cpu(), curr_seg_label)
            evaluator.update(pred_label.cpu(), curr_seg_label)

            left_idx = right_idx

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)
        end = time.time()

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (
            cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0
        ):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        "iter: {iter}/{total_iter}",
                        "{meters}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=cur_iter,
                    total_iter=len(test_dataloader),
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0**2),
                )
            )

        # summary
        if (
            summary_writer is not None
            and cfg.TRAIN.SUMMARY_PERIOD > 0
            and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0
        ):
            keywords = ("loss", "acc", "iou")
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar(
                    "train/" + name, meter.avg, global_step=cur_iter
                )


    eval_list = []
    train_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
    eval_list.append(("2D", evaluator_2d))

    train_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
    eval_list.append(("3D", evaluator_3d))

    eval_list.append(("2D+3D", evaluator))
    for modality, evaluator in eval_list:
        logger.info(
            "{} overall accuracy: {:.2f}%".format(
                modality, 100.0 * evaluator.overall_acc
            )
        )
        logger.info(
            "{} overall IOU: {:.2f}".format(modality, 100.0 * evaluator.overall_iou)
        )
        logger.info(
            "{} class-wise segmentation accuracy and IoU.\n{}".format(
                modality, evaluator.print_table()
            )
        )



def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace("@", config_path.replace("configs/", ""))
        if osp.isdir(output_dir):
            warnings.warn("Output directory exists.")
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime("%m-%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_name = "{:s}.{:s}".format(timestamp, hostname)

    logger = setup_logger("xmuda", output_dir, comment="test.{:s}".format(run_name))
    logger.info("{:d} GPUs available".format(torch.cuda.device_count()))
    logger.info(args)

    logger.info("Loaded configuration file {:s}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    # # check that 2D and 3D model use either both single head or both dual head
    # assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # # check if there is at least one loss on target set
    # assert (
    #     cfg.TRAIN.XMUDA.lambda_xm_src > 0
    #     or cfg.TRAIN.XMUDA.lambda_xm_trg > 0
    #     or cfg.TRAIN.XMUDA.lambda_pl > 0
    #     or cfg.TRAIN.XMUDA.lambda_minent > 0
    # )
    mmtta(cfg, output_dir, run_name)


if __name__ == "__main__":
    main()
