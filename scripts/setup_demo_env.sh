#!/usr/bin/env bash
# 轻量 Demo 环境（无需 GPU / PyTorch / 数据集）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME="acp-demo"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[$ENV_NAME] 环境已存在，跳过创建"
else
    echo "=== 创建 $ENV_NAME 环境 ==="
    conda env create -f "$REPO_ROOT/config/conda_environment_demo.yaml" -y
fi

conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

# yaml 里的 pip 段偶发失败；用清华源兜底
pip install -q cvxpy plotly -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 验证依赖 ==="
python -c "
import numpy, scipy, matplotlib, spatialmath, cvxpy, plotly
print('numpy', numpy.__version__)
print('spatialmath OK | cvxpy OK | plotly OK')
print('环境就绪: conda activate $ENV_NAME')
"
