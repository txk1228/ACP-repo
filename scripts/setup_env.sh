#!/usr/bin/env bash
# ACP 训练环境变量 — source 此文件: source /home/zj/ACP_fx/scripts/setup_env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYRITE_RAW_DATASET_FOLDERS="${PYRITE_RAW_DATASET_FOLDERS:-$HOME/data/real}"
export PYRITE_DATASET_FOLDERS="${PYRITE_DATASET_FOLDERS:-$HOME/data/real_processed}"
export PYRITE_CHECKPOINT_FOLDERS="${PYRITE_CHECKPOINT_FOLDERS:-$HOME/training_outputs}"
export PYRITE_HARDWARE_CONFIG_FOLDERS="${PYRITE_HARDWARE_CONFIG_FOLDERS:-$REPO_ROOT/config}"
export PYRITE_CONTROL_LOG_FOLDERS="${PYRITE_CONTROL_LOG_FOLDERS:-$HOME/data/control_log}"

# 避免 ~/.local 包与 conda 环境冲突
export PYTHONNOUSERSITE=1
# HuggingFace 镜像（国内网络）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$PYRITE_RAW_DATASET_FOLDERS" \
         "$PYRITE_DATASET_FOLDERS" \
         "$PYRITE_CHECKPOINT_FOLDERS" \
         "$PYRITE_HARDWARE_CONFIG_FOLDERS" \
         "$PYRITE_CONTROL_LOG_FOLDERS"

echo "ACP env vars set:"
echo "  PYRITE_DATASET_FOLDERS=$PYRITE_DATASET_FOLDERS"
echo "  PYRITE_CHECKPOINT_FOLDERS=$PYRITE_CHECKPOINT_FOLDERS"
