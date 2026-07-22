#!/usr/bin/env bash
# 启动 ACP 训练（需先 activate pyrite 环境并 source setup_env.sh）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyrite
source "$REPO_ROOT/scripts/setup_env.sh"

cd "$REPO_ROOT/adaptive_compliance_policy/PyriteML"

MODE="${1:-spec}"  # spec=FFT, conv=TCN

if [[ "$MODE" == "spec" ]]; then
    CONFIG=train_spec_workspace
elif [[ "$MODE" == "conv" ]]; then
    CONFIG=train_conv_workspace
else
    echo "Usage: $0 [spec|conv]"
    exit 1
fi

echo "Training ACP with config: $CONFIG"
export PYTHONNOUSERSITE=1
HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch \
  --config_file "$REPO_ROOT/config/accelerate_config.yaml" \
  train.py --config-name="$CONFIG"
