#!/usr/bin/env bash
# 断点续训：从已有 run 目录的 checkpoints/latest.ckpt 接着训（不重开新目录）
#
# 用法:
#   bash scripts/train_resume.sh <run_dir> <config-name> [hydra overrides...]
#
# 示例（vase Spec，batch=32）:
#   bash scripts/train_resume.sh \
#     ~/training_outputs/2026.07.18_10.23.05_vase_wiping_resnet_230 \
#     train_spec_workspace \
#     task=vase_wiping_spec \
#     dataloader.batch_size=32 val_dataloader.batch_size=32 \
#     dataloader.num_workers=2 val_dataloader.num_workers=2 \
#     dataloader.persistent_workers=false val_dataloader.persistent_workers=false \
#     training.eval_metrics_max_batches=20
#
# 示例（vase Conv compare）:
#   bash scripts/train_resume.sh \
#     ~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230 \
#     train_conv_compare_vase_workspace \
#     training.eval_metrics_max_batches=20
#
# 示例（flip Spec）:
#   bash scripts/train_resume.sh \
#     ~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230 \
#     train_spec_workspace \
#     task=flip_up_spec
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "用法: bash scripts/train_resume.sh <run_dir> <config-name> [hydra overrides...]"
  echo "说明: run_dir 须含 checkpoints/latest.ckpt；会设置 training.resume=true 与 hydra.run.dir=该目录"
  exit 1
fi

RUN_DIR="$(readlink -f "$1")"
CONFIG="$2"
shift 2

CKPT="$RUN_DIR/checkpoints/latest.ckpt"
if [[ ! -f "$CKPT" ]]; then
  echo "找不到 checkpoint: $CKPT"
  echo "请确认训练曾至少保存过一次（默认每 10 epoch 存盘）。"
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyrite
source "$REPO_ROOT/scripts/setup_env.sh"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT/adaptive_compliance_policy/PyriteML:$REPO_ROOT/adaptive_compliance_policy${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

echo "Resuming from: $CKPT"
echo "Run dir:       $RUN_DIR"
echo "Config:        $CONFIG"
echo "Overrides:     $*"

cd "$REPO_ROOT/adaptive_compliance_policy/PyriteML"

HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch \
  --config_file "$REPO_ROOT/config/accelerate_config.yaml" \
  train.py --config-name="$CONFIG" \
  logging.mode=disabled \
  training.resume=true \
  hydra.run.dir="$RUN_DIR" \
  "$@"
