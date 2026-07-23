#!/usr/bin/env bash
# vase Spec 断点续训快捷入口（默认正式 Spec 基线目录，可改传 run_dir）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$HOME/training_outputs/2026.07.18_10.23.05_vase_wiping_resnet_230}"

exec bash "$REPO_ROOT/scripts/train_resume.sh" "$RUN_DIR" train_spec_workspace \
  task=vase_wiping_spec \
  dataloader.batch_size=32 \
  dataloader.num_workers=2 \
  dataloader.persistent_workers=false \
  val_dataloader.batch_size=32 \
  val_dataloader.num_workers=2 \
  val_dataloader.persistent_workers=false \
  training.eval_metrics_max_batches=20
