#!/usr/bin/env bash
# GPU 冒烟测试：验证环境 + 训练 pipeline 能进 4090（非完整训练）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyrite
source "$REPO_ROOT/scripts/setup_env.sh"

cd "$REPO_ROOT/adaptive_compliance_policy/PyriteML"

MODE="${1:-spec}"
CONFIG=$([[ "$MODE" == "conv" ]] && echo "train_conv_workspace" || echo "train_spec_workspace")

echo "=== ACP GPU smoke test (短跑, batch=4) ==="
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch \
  --config_file "$REPO_ROOT/config/accelerate_config.yaml" \
  train.py --config-name="$CONFIG" \
  training.device=cuda \
  dataloader.batch_size=4 \
  dataloader.num_workers=2 \
  val_dataloader.batch_size=4 \
  val_dataloader.num_workers=2 \
  training.num_epochs=5 \
  training.max_train_steps=10 \
  training.max_val_steps=5 \
  logging.mode=disabled \
  policy.obs_encoder.vision_encoder_cfg.pretrained=false
