#!/usr/bin/env bash
# 断电后续训 vase（从已有 run 的 latest.ckpt 接着跑）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyrite
source "$REPO_ROOT/scripts/setup_env.sh"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT/adaptive_compliance_policy/PyriteML:$REPO_ROOT/adaptive_compliance_policy${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# 默认续当前这次正式 run；也可手动传入目录
RUN_DIR="${1:-$HOME/training_outputs/2026.07.18_10.23.05_vase_wiping_resnet_230}"
CKPT="$RUN_DIR/checkpoints/latest.ckpt"

if [[ ! -f "$CKPT" ]]; then
  echo "找不到 checkpoint: $CKPT"
  echo "用法: bash train_vase_resume.sh [/path/to/run_dir]"
  exit 1
fi

echo "Resuming from: $CKPT"
echo "Run dir:       $RUN_DIR"

cd "$REPO_ROOT/adaptive_compliance_policy/PyriteML"

HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch \
  --config_file "$REPO_ROOT/config/accelerate_config.yaml" \
  train.py --config-name=train_spec_workspace \
  task=vase_wiping_spec \
  logging.mode=disabled \
  training.resume=true \
  hydra.run.dir="$RUN_DIR" \
  dataloader.batch_size=32 \
  dataloader.num_workers=2 \
  dataloader.persistent_workers=false \
  val_dataloader.batch_size=32 \
  val_dataloader.num_workers=2 \
  val_dataloader.persistent_workers=false \
  training.eval_metrics_max_batches=20
