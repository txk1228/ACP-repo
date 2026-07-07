#!/usr/bin/env bash
# ACP 训练环境变量 — source 此文件: source /home/xiaoke/ACP_fx/setup_env.sh

export PYRITE_RAW_DATASET_FOLDERS=/home/xiaoke/data/real
export PYRITE_DATASET_FOLDERS=/home/xiaoke/data/real_processed
export PYRITE_CHECKPOINT_FOLDERS=/home/xiaoke/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=/home/xiaoke/ACP_fx/config
export PYRITE_CONTROL_LOG_FOLDERS=/home/xiaoke/data/control_log

# 避免 ~/.local 包与 conda 环境冲突
export PYTHONNOUSERSITE=1
# HuggingFace 镜像（国内网络）
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "$PYRITE_RAW_DATASET_FOLDERS" \
         "$PYRITE_DATASET_FOLDERS" \
         "$PYRITE_CHECKPOINT_FOLDERS" \
         "$PYRITE_HARDWARE_CONFIG_FOLDERS" \
         "$PYRITE_CONTROL_LOG_FOLDERS"

echo "ACP env vars set:"
echo "  PYRITE_DATASET_FOLDERS=$PYRITE_DATASET_FOLDERS"
echo "  PYRITE_CHECKPOINT_FOLDERS=$PYRITE_CHECKPOINT_FOLDERS"
