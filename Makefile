# ===============================
# baseline learns only on source data
ckpt2d_baseline = /home/lokesh/workspace/outputs/xmuda_journal/nuscenes_lidarseg/usa_singapore/uda/baseline/model_2d_100000.pth
ckpt3d_baseline = /home/lokesh/workspace/outputs/xmuda_journal/nuscenes_lidarseg/usa_singapore/uda/baseline/model_3d_100000.pth

# xmuda_single / xmuda_dual / xmuda

ckpt2d_xmuda = /home/lokesh/workspace/outputs/xmuda_journal/nuscenes_lidarseg/usa_singapore/uda/xmuda/model_2d_100000.pth
ckpt3d_xmuda = /home/lokesh/workspace/outputs/xmuda_journal/nuscenes_lidarseg/usa_singapore/uda/xmuda/model_3d_100000.pth
# ===============================

# ===============================
# official weights
ckpt3d_baseline_off = checkpoints/nuscenes_lidarseg/day_night/uda/baseline/model_3d_080000.pth

ckpt2d_xmuda_off = checkpoints/nuscenes_lidarseg/day_night/uda/xmuda/model_2d_040000.pth
ckpt3d_xmuda_off = checkpoints/nuscenes_lidarseg/day_night/uda/xmuda/model_3d_100000.pth

ckpt2d_xmuda_pl_off = checkpoints/nuscenes_lidarseg/day_night/uda/xmuda_pl/model_2d_045000.pth
ckpt3d_xmuda_pl_off = checkpoints/nuscenes_lidarseg/day_night/uda/xmuda_pl/model_3d_100000.pth
# ===============================

# ===============================
# cfg_baseline = configs/nuscenes_lidarseg/day_night/uda/baseline.yaml
cfg_baseline = configs/nuscenes_lidarseg/usa_singapore/uda/baseline.yaml

# cfg_xmuda = configs/nuscenes_lidarseg/day_night/uda/xmuda.yaml
cfg_xmuda = configs/nuscenes_lidarseg/usa_singapore/uda/xmuda.yaml

# cfg_xmuda_pl = configs/nuscenes_lidarseg/day_night/uda/xmuda_pl.yaml
cfg_xmuda_pl = configs/nuscenes_lidarseg/usa_singapore/uda/xmuda_pl.yaml

cfg_mmtta = configs/nuscenes_lidarseg/usa_singapore/uda/mmtta.yaml
# ===============================

.PHONY: all train_baseline train_xmuda train_xmuda_pl test

all: 
	uv sync

train_baseline: 
	uv run xmuda/train_baseline.py --cfg=$(cfg_baseline)
train_xmuda: 
	uv run xmuda/train_xmuda.py --cfg=$(cfg_xmuda)
train_xmuda_pl: 
	uv run xmuda/train_xmuda.py --cfg=$(cfg_xmuda_pl)
train_baseline_with_trg: 
	uv run xmuda/train_baseline_src_trg.py --cfg=$(cfg_baseline)

mmtta:
	uv run xmuda/mmtta.py --cfg=$(cfg_mmtta)

test-pdb:
	uv run python -m pdb xmuda/test.py \
		--cfg=$(cfg_xmuda) \
		--ckpt2d=$(ckpt2d_xmuda) \
		--ckpt3d=$(ckpt3d_xmuda)

test_xmuda:
	uv run xmuda/test.py \
		--cfg=$(cfg_xmuda) \
		--ckpt2d=$(ckpt2d_xmuda) \
		--ckpt3d=$(ckpt3d_xmuda)

test_baseline:
	uv run xmuda/test.py \
		--cfg=$(cfg_baseline) \
		--ckpt2d=$(ckpt2d_baseline) \
		--ckpt3d=$(ckpt3d_baseline)
