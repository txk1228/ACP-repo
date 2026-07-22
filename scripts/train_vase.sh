#!/usr/bin/env bash
# Train vase_wiping (dual-arm) with physical val RMSE → eval_latest_val_metrics.json
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

cd "$REPO_ROOT/adaptive_compliance_policy/PyriteML"

# batch=32 / workers=2: 93GB RAM cannot hold vase dataset + 32 DataLoader workers
echo "Training vase_wiping_spec (dual-arm) with val RMSE metrics"
HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch \
  --config_file "$REPO_ROOT/config/accelerate_config.yaml" \
  train.py --config-name=train_spec_workspace \
  task=vase_wiping_spec \
  logging.mode=disabled \
  dataloader.batch_size=32 \
  dataloader.num_workers=2 \
  dataloader.persistent_workers=false \
  val_dataloader.batch_size=32 \
  val_dataloader.num_workers=2 \
  val_dataloader.persistent_workers=false \
  training.eval_metrics_max_batches=20
