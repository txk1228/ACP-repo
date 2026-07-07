#!/usr/bin/env bash
# CPU 冒烟测试：验证环境 + 数据集 + 训练 pipeline 能跑通（非完整训练）
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyrite
source /home/xiaoke/ACP_fx/setup_env.sh

cd /home/xiaoke/ACP_fx/adaptive_compliance_policy/PyriteML

MODE="${1:-spec}"
CONFIG=$([[ "$MODE" == "conv" ]] && echo "train_conv_workspace" || echo "train_spec_workspace")

echo "=== ACP CPU smoke test (5 epochs, batch=4) ==="
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch train.py --config-name="$CONFIG" \
  training.device=cpu \
  dataloader.batch_size=4 \
  dataloader.num_workers=2 \
  val_dataloader.batch_size=4 \
  val_dataloader.num_workers=2 \
  training.num_epochs=5 \
  training.max_train_steps=10 \
  training.max_val_steps=5 \
  logging.mode=disabled \
  policy.obs_encoder.vision_encoder_cfg.pretrained=false
