#!/usr/bin/env bash
# 轻量安装 ACP 训练环境（CPU 版，避免 conda 求解器卡死）
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "^pyrite "; then
    echo "pyrite 环境已存在，跳过创建"
else
    echo "=== 创建 pyrite 环境 (python=3.10) ==="
    conda create -n pyrite python=3.10 pip -y
fi

conda activate pyrite

echo "=== 安装 PyTorch CPU ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "=== 安装 ACP 核心依赖 ==="
pip install \
    huggingface_hub wandb timm diffusers accelerate \
    threadpoolctl plotly dill einops hydra-core \
    ipykernel ipython matplotlib omegaconf opencv-python \
    pandas pyyaml scipy tqdm zarr spatialmath-python \
    numcodecs scikit-video scikit-fda \
    v4l2py toppra atomics vit-pytorch imagecodecs cvxpy

echo "=== 验证关键包 ==="
python -c "
import torch, diffusers, timm, zarr, spatialmath, cvxpy, imagecodecs
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('All imports OK')
"

echo "=== 完成！激活方式: conda activate pyrite ==="
