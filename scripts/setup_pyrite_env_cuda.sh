#!/usr/bin/env bash
# RTX 4090 / CUDA 版 ACP 训练环境
# 在新工作站上执行；不要在无 GPU 的旧笔记本上当正式训练环境用
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：未检测到 nvidia-smi。请先安装 NVIDIA 驱动并 reboot。"
    echo "参考：docs/WORKSTATION_4090_SETUP.zh.md"
    exit 1
fi

echo "=== GPU 信息 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

ENV_NAME="pyrite"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[$ENV_NAME] 环境已存在，将更新 CUDA 版 PyTorch 与依赖"
else
    echo "=== 创建 $ENV_NAME (python=3.10) ==="
    conda create -n "$ENV_NAME" python=3.10 pip -y
fi

conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

echo "=== 安装 PyTorch CUDA 12.1（适配 4090）==="
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# 清华主源拉通用依赖，PyTorch 附加源拉 CUDA wheel（国内更稳）
pip install torch torchvision torchaudio \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --retries 10 --timeout 180

echo "=== 安装 ACP 依赖 ==="
pip install \
    huggingface_hub wandb timm diffusers accelerate \
    threadpoolctl plotly dill einops hydra-core \
    ipykernel ipython matplotlib omegaconf opencv-python \
    pandas pyyaml scipy tqdm zarr spatialmath-python \
    numcodecs scikit-video scikit-fda \
    v4l2py toppra atomics vit-pytorch imagecodecs cvxpy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --retries 10 --timeout 180

echo "=== 验证 ==="
python - <<'EOF'
import torch
import diffusers, timm, zarr, spatialmath, cvxpy, imagecodecs
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA 不可用：请检查驱动与 PyTorch CUDA 版本")
print("All imports OK")
EOF

echo ""
echo "=== 完成 ==="
echo "激活: conda activate pyrite"
echo "务必: export PYTHONNOUSERSITE=1"
echo "下一步: 见 docs/WORKSTATION_4090_SETUP.zh.md 第 6–8 节"
