#!/usr/bin/env bash
# 轻量 Demo 环境（无需 GPU / PyTorch / 数据集）
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME="acp-demo"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[$ENV_NAME] 环境已存在，跳过创建"
else
    echo "=== 创建 $ENV_NAME 环境 ==="
    conda env create -f /home/xiaoke/ACP_fx/conda_environment_demo.yaml -y
fi

conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

# cvxpy 可能已通过 yaml pip 安装；兜底再装一次
pip install -q cvxpy -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

echo "=== 验证依赖 ==="
python -c "
import numpy, scipy, matplotlib, spatialmath, cvxpy
print('numpy', numpy.__version__)
print('spatialmath OK | cvxpy OK')
print('环境就绪: conda activate $ENV_NAME')
"
