#!/usr/bin/env bash
# 一键运行 ACP 核心算法 Demo
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate acp-demo
export PYTHONNOUSERSITE=1

cd /home/xiaoke/ACP_fx
python demo/virtual_target_stiffness_demo.py --save demo/output "$@"
