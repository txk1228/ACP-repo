#!/usr/bin/env bash
# 启动 ACP 训练（需先 activate pyrite 环境并 source setup_env.sh）
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyrite
source /home/xiaoke/ACP_fx/setup_env.sh

cd /home/xiaoke/ACP_fx/adaptive_compliance_policy/PyriteML

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
HYDRA_FULL_ERROR=1 python -m accelerate.commands.launch train.py --config-name="$CONFIG"
