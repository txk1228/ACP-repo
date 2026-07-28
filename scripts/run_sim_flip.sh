#!/usr/bin/env bash
# =============================================================================
# ACP 仿真翻方块 — 一键启动
# =============================================================================
# 用法：
#   bash scripts/run_sim_flip.sh              # 默认：v2-ft + MuJoCo 窗口 + 循环
#   bash scripts/run_sim_flip.sh v1           # 脚本专家（无策略）
#   bash scripts/run_sim_flip.sh v2           # 真机 ckpt 零样本（验链路）
#   bash scripts/run_sim_flip.sh v2-ft        # 微调策略 + MuJoCo 窗口（GitHub 原版）
#   bash scripts/run_sim_flip.sh v2-ft-live   # 微调策略 + ACP Live 分屏（增强演示）
#   bash scripts/run_sim_flip.sh record       # 录制专家数据
#   bash scripts/run_sim_flip.sh help
#
# 环境（可覆盖）：
#   ACP_I7_MODEL_ROOT          i7 MJCF 根目录
#   DISPLAY                    自动检测
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
export PYTHONUNBUFFERED=1

# 某些桌面代理会把 socks 代理注入当前 shell，huggingface_hub/httpx 默认不认。
for _k in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
  unset "$_k" 2>/dev/null || true
done

_detect_display() {
  local d sock
  if [[ -n "${DISPLAY:-}" ]]; then
    sock="/tmp/.X11-unix/X${DISPLAY#:}"
    if [[ -S "$sock" ]]; then
      echo "$DISPLAY"
      return
    fi
    echo "[warn] DISPLAY=$DISPLAY 不可用（无 $sock），尝试自动检测…" >&2
  fi
  d=$(who 2>/dev/null | awk '/\(:[0-9]+\)/ {print $2; exit}')
  if [[ -n "$d" ]] && [[ -S "/tmp/.X11-unix/X${d#:}" ]]; then
    echo "$d"
    return
  fi
  d=$(loginctl show-user "$(whoami)" -p Display --value 2>/dev/null || true)
  if [[ -n "$d" ]] && [[ -S "/tmp/.X11-unix/X${d}" ]]; then
    echo ":${d}"
    return
  fi
  for sock in /tmp/.X11-unix/X*; do
    [[ -e "$sock" ]] || continue
    echo ":${sock##*X}"
    return
  done
  echo ":1"
}

_setup_gl_env() {
  case "$MODE" in
    v2-ft-headless|eval|record|label|finetune|pipeline)
      export MUJOCO_GL="${MUJOCO_GL:-egl}"
      ;;
    v2-ft-live|v2-ft-live-once|live|demo-live)
      export DISPLAY="${DISPLAY:-$(_detect_display)}"
      export MUJOCO_GL="${MUJOCO_GL:-egl}"
      if [[ ! -S "/tmp/.X11-unix/X${DISPLAY#:}" ]]; then
        echo "[error] 无法打开图形显示 DISPLAY=$DISPLAY"
        echo "  请在 GNOME 桌面里打开终端再跑（不要 SSH 无 X11 转发）。"
        echo "  或改用：bash scripts/run_sim_flip.sh v2-ft-headless"
        exit 1
      fi
      ;;
    *)
      export DISPLAY="${DISPLAY:-$(_detect_display)}"
      export MUJOCO_GL="${MUJOCO_GL:-glfw}"
      if [[ ! -S "/tmp/.X11-unix/X${DISPLAY#:}" ]]; then
        echo "[error] 无法打开图形显示 DISPLAY=$DISPLAY"
        echo "  请在 GNOME 桌面里打开终端再跑（不要 SSH 无 X11 转发）。"
        echo "  或改用：bash scripts/run_sim_flip.sh v2-ft-headless"
        exit 1
      fi
      ;;
  esac
}
_setup_gl_env

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
  echo "  DISPLAY=${DISPLAY:-}  MUJOCO_GL=${MUJOCO_GL:-}"
  echo "  ACP_I7_MODEL_ROOT=$ACP_I7_MODEL_ROOT"
  echo "============================================================"
}

case "$MODE" in
  help|-h|--help)
    cat <<'EOF'
ACP 仿真翻方块 — 一键启动

用法：
  bash scripts/run_sim_flip.sh                 # 默认：v2-ft + MuJoCo 窗口 + 循环
  bash scripts/run_sim_flip.sh v1              # 脚本专家
  bash scripts/run_sim_flip.sh v1-loop         # 脚本循环
  bash scripts/run_sim_flip.sh v2              # 真机 ckpt 零样本（验链路）
  bash scripts/run_sim_flip.sh v2-ft           # 微调策略 + MuJoCo 窗口（GitHub 原版）
  bash scripts/run_sim_flip.sh v2-ft-once      # 微调策略单轮 + MuJoCo 窗口
  bash scripts/run_sim_flip.sh v2-ft-live      # 微调策略 + ACP Live 分屏循环
  bash scripts/run_sim_flip.sh v2-ft-live-once # ACP Live 分屏单轮
  bash scripts/run_sim_flip.sh v2-ft-headless  # 无头评测
  bash scripts/run_sim_flip.sh record          # 录制专家数据
  bash scripts/run_sim_flip.sh label           # VT 标注
  bash scripts/run_sim_flip.sh finetune        # 短训
  bash scripts/run_sim_flip.sh pipeline        # record→label→finetune

环境变量（可选）：
  ACP_I7_MODEL_ROOT   i7 MJCF 根目录
  DISPLAY             图形显示（live 模式自动检测）
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
    echo "[run] v2-ft 微调策略 + MuJoCo 窗口 + 循环（GitHub 原版）"
    echo "  ckpt=$FT_CKPT"
    echo "  验收: tilt≥55°；Ctrl+C 或关闭 MuJoCo 窗口退出"
    echo "  录屏媒体: bash scripts/make_github_media.sh"
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
    echo "[run] v2-ft 单轮评测 + MuJoCo 窗口（GitHub 原版）"
    echo "  ckpt=$FT_CKPT"
    echo "  验收: tilt≥55°"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --require-flip --ckpt "$FT_CKPT" \
      --render \
      --steps 3600 --exec-horizon 12 --action-ds 50 \
      --viewer-sync-every 5 --no-plot \
      --rgb-dump-dir "$REPO_ROOT/sim_acp/outputs/wrist_rgb_ft" \
      "$@"
    ;;

  v2-ft-live|live|demo-live)
    _check_i7
    _check_ckpt "$FT_CKPT"
    _banner
    echo "[run] v2-ft-live 微调策略 + ACP Live 分屏循环"
    echo "  ckpt=$FT_CKPT"
    echo "  验收: tilt≈完全翻转（85°+ hold）；Ctrl+C 退出"
    echo "  布局: 左 demo : 右曲线 ≈ 6:4，600×480，2.5x 超采样"
    echo "  原版 MuJoCo 窗口请用: bash scripts/run_sim_flip.sh v2-ft"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --require-flip --ckpt "$FT_CKPT" \
      --loop \
      --steps 2400 --exec-horizon 12 --action-ds 50 \
      --viewer-sync-every 5 --no-plot \
      --show-live-panel \
      --live-width 600 --live-height 480 --live-panel-width 400 \
      --live-render-scale 2.5 --live-panel-render-scale 2.0 \
      --flip-done-deg 85 \
      --hold-after-flip 500 \
      --rgb-dump-dir "$REPO_ROOT/sim_acp/outputs/live_rgb_ft" \
      "$@"
    ;;

  v2-ft-live-once|live-once)
    _check_i7
    _check_ckpt "$FT_CKPT"
    _banner
    echo "[run] v2-ft-live 单轮 ACP Live 分屏评测"
    echo "  ckpt=$FT_CKPT"
    exec python -m sim_acp.run_flip_cube_demo \
      --policy --require-flip --ckpt "$FT_CKPT" \
      --steps 2400 --exec-horizon 12 --action-ds 50 \
      --viewer-sync-every 5 --no-plot \
      --show-live-panel \
      --live-width 600 --live-height 480 --live-panel-width 400 \
      --live-render-scale 2.5 --live-panel-render-scale 2.0 \
      --flip-done-deg 85 \
      --hold-after-flip 500 \
      --rgb-dump-dir "$REPO_ROOT/sim_acp/outputs/live_rgb_ft" \
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
    echo "[pipeline] 完成。请 export ACP_SIM_FT_CKPT=<新 ckpt> 后跑 v2-ft 或 v2-ft-live。"
    ;;

  *)
    echo "[error] 未知 mode: $MODE"
    echo "  可用: v1 | v1-loop | v2 | v2-ft | v2-ft-once | v2-ft-live | v2-ft-live-once | v2-ft-headless | record | label | finetune | pipeline | help"
    exit 1
    ;;
esac
