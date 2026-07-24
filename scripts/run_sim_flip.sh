#!/usr/bin/env bash
# =============================================================================
# ACP 仿真翻方块 — 一键启动
# =============================================================================
# 用法：
#   bash scripts/run_sim_flip.sh              # 默认：v2-ft + 窗口 + 循环
#   bash scripts/run_sim_flip.sh v1           # 脚本专家（无策略）
#   bash scripts/run_sim_flip.sh v2           # 真机 ckpt 零样本（验链路）
#   bash scripts/run_sim_flip.sh v2-ft        # 微调策略翻方块（推荐）
#   bash scripts/run_sim_flip.sh v2-ft-once   # 微调策略跑一轮（不循环）
#   bash scripts/run_sim_flip.sh record       # 录制专家数据
#   bash scripts/run_sim_flip.sh label        # VT 标注
#   bash scripts/run_sim_flip.sh finetune     # 短训微调
#   bash scripts/run_sim_flip.sh help
#
# 环境（可覆盖）：
#   ACP_I7_MODEL_ROOT          i7 MJCF 根目录
#   DISPLAY                    默认 :1
#   ACP_SIM_FT_CKPT            v2-ft 权重路径
#   ACP_SIM_REAL_CKPT          真机零样本权重
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-v2-ft}"
shift || true

# --- conda ---
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate pyrite 2>/dev/null || true

# --- ACP 环境变量 ---
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/setup_env.sh"

export ACP_I7_MODEL_ROOT="${ACP_I7_MODEL_ROOT:-/home/zj/robot-control-v1.5/model_new}"
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTHONUNBUFFERED=1

REAL_CKPT="${ACP_SIM_REAL_CKPT:-$HOME/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230/checkpoints/latest.ckpt}"
FT_CKPT_DEFAULT="$HOME/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt"
FT_CKPT="${ACP_SIM_FT_CKPT:-$FT_CKPT_DEFAULT}"
SIM_DATASET="${ACP_SIM_DATASET:-$PYRITE_DATASET_FOLDERS/flip_up_sim_v1}"

_check_i7() {
  if [[ ! -f "$ACP_I7_MODEL_ROOT/mjcf/URDF_V1.5_0416.xml" ]]; then
    echo "[error] 找不到 i7 MJCF: $ACP_I7_MODEL_ROOT/mjcf/URDF_V1.5_0416.xml"
    echo "  请设置: export ACP_I7_MODEL_ROOT=/path/to/robot-control-v1.5/model_new"
    exit 1
  fi
}

_check_ckpt() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[error] 找不到 checkpoint: $p"
    exit 1
  fi
}

_banner() {
  echo "============================================================"
  echo " ACP 仿真翻方块  mode=$MODE"
  echo "  REPO=$REPO_ROOT"
  echo "  DISPLAY=$DISPLAY  MUJOCO_GL=$MUJOCO_GL"
  echo "  ACP_I7_MODEL_ROOT=$ACP_I7_MODEL_ROOT"
  echo "============================================================"
}

case "$MODE" in
  help|-h|--help)
    cat <<'EOF'
ACP 仿真翻方块 — 一键启动

用法：
  bash scripts/run_sim_flip.sh                 # 默认：v2-ft + 窗口 + 循环
  bash scripts/run_sim_flip.sh v1              # 脚本专家
  bash scripts/run_sim_flip.sh v1-loop         # 脚本循环
  bash scripts/run_sim_flip.sh v2              # 真机 ckpt 零样本（验链路）
  bash scripts/run_sim_flip.sh v2-ft           # 微调策略循环（推荐）
  bash scripts/run_sim_flip.sh v2-ft-once      # 微调策略单轮 + 窗口
  bash scripts/run_sim_flip.sh v2-ft-headless  # 无头评测
  bash scripts/run_sim_flip.sh record          # 录制专家数据
  bash scripts/run_sim_flip.sh label           # VT 标注
  bash scripts/run_sim_flip.sh finetune        # 短训
  bash scripts/run_sim_flip.sh pipeline        # record→label→finetune

环境变量（可选）：
  ACP_I7_MODEL_ROOT   i7 MJCF 根目录
  DISPLAY             默认 :1
  ACP_SIM_FT_CKPT     v2-ft 权重
  ACP_SIM_REAL_CKPT   真机零样本权重
  ACP_SIM_DATASET     仿真 zarr 路径
  ACP_SIM_RECORD_N    录制条数（默认 50）

详见：sim_acp/README.md
EOF
    exit 0
    ;;

  v1)
    _check_i7
    _banner
    echo "[run] 脚本专家翻方块（无 RGB）"
    exec python -m sim_acp.run_flip_cube_demo --render --no-plot "$@"
    ;;

  v1-loop)
    _check_i7
    _banner
    echo "[run] 脚本专家循环（Ctrl+C / 关窗退出）"
    exec python -m sim_acp.run_flip_cube_demo --render --loop --no-plot "$@"
    ;;

  v2)
    _check_i7
    _check_ckpt "$REAL_CKPT"
    _banner
    echo "[run] 真机 Flip Spec 零样本（只验链路，不宣称翻成功）"
    echo "  ckpt=$REAL_CKPT"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --ckpt "$REAL_CKPT" \
      --render --steps 2500 --exec-horizon 12 --action-ds 50 \
      --viewer-sync-every 5 --no-plot "$@"
    ;;

  v2-ft|ft|demo)
    _check_i7
    _check_ckpt "$FT_CKPT"
    _banner
    echo "[run] v2-ft 微调策略 + 腕部 RGB + 循环演示"
    echo "  ckpt=$FT_CKPT"
    echo "  验收: tilt≥55°；Ctrl+C 或关闭 MuJoCo 窗口退出"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --require-flip --ckpt "$FT_CKPT" \
      --render --loop \
      --steps 3600 --exec-horizon 12 --action-ds 50 \
      --viewer-sync-every 5 --no-plot \
      --rgb-dump-dir "$REPO_ROOT/sim_acp/outputs/wrist_rgb_ft" \
      "$@"
    ;;

  v2-ft-once|once)
    _check_i7
    _check_ckpt "$FT_CKPT"
    _banner
    echo "[run] v2-ft 单轮评测（tilt≥55°）"
    echo "  ckpt=$FT_CKPT"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --require-flip --ckpt "$FT_CKPT" \
      --render \
      --steps 3600 --exec-horizon 12 --action-ds 50 \
      --viewer-sync-every 5 --no-plot \
      --rgb-dump-dir "$REPO_ROOT/sim_acp/outputs/wrist_rgb_ft" \
      "$@"
    ;;

  v2-ft-headless|eval)
    _check_i7
    _check_ckpt "$FT_CKPT"
    _banner
    echo "[run] v2-ft 无头评测"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --require-flip --ckpt "$FT_CKPT" \
      --steps 3600 --exec-horizon 12 --action-ds 50 --no-plot "$@"
    ;;

  record)
    _check_i7
    _banner
    N="${ACP_SIM_RECORD_N:-50}"
    echo "[run] 录制 $N 条成功专家 episode → $SIM_DATASET"
    exec python -m sim_acp.scripts.record_flip_episodes \
      --n "$N" --out "$SIM_DATASET" "$@"
    ;;

  label)
    _banner
    echo "[run] virtual target 标注 → $SIM_DATASET"
    exec python -m sim_acp.data.label_virtual_target --dataset "$SIM_DATASET" "$@"
    ;;

  finetune)
    _check_ckpt "$REAL_CKPT"
    _banner
    echo "[run] 从真机 ckpt 微调（+30 epoch）"
    echo "  pretrained=$REAL_CKPT"
    echo "  dataset=$SIM_DATASET"
    exec python -m sim_acp.scripts.finetune_flip_spec \
      --pretrained "$REAL_CKPT" --epochs 30 --batch-size 32 "$@"
    ;;

  pipeline)
    _check_i7
    _banner
    echo "[pipeline] record → label → finetune"
    N="${ACP_SIM_RECORD_N:-50}"
    python -m sim_acp.scripts.record_flip_episodes --n "$N" --out "$SIM_DATASET"
    python -m sim_acp.data.label_virtual_target --dataset "$SIM_DATASET"
    python -m sim_acp.scripts.finetune_flip_spec --pretrained "$REAL_CKPT" --epochs 30
    echo "[pipeline] 完成。请把最新 flip_sim_ft 目录设为 ACP_SIM_FT_CKPT 后跑 demo。"
    ;;

  *)
    echo "[error] 未知 mode: $MODE"
    echo "  可用: v1 | v1-loop | v2 | v2-ft | v2-ft-once | v2-ft-headless | record | label | finetune | pipeline | help"
    exit 1
    ;;
esac
