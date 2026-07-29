#!/usr/bin/env bash
# 生成 GitHub 展示媒体（默认只录主线 v2-ft）：
#   docs/media/sim_flip_v2ft.{mp4,gif}
# 可选 Live UI 媒体：
#   bash scripts/make_github_media.sh --also-live
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate pyrite
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/setup_env.sh"

export ACP_I7_MODEL_ROOT="${ACP_I7_MODEL_ROOT:-/home/zj/robot-control-v1.5/model_new}"
# 优先 Mesa EGL（避开 NVIDIA 驱动不匹配时的 GL 挂掉）
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

# 代理会干扰部分本机推理/下载路径；录屏时清掉
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

# 默认跳过 live；显式 --also-live / --only-live 时再录
ARGS=("$@")
HAS_LIVE_FLAG=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$a" in
    --also-live|--only-live|--skip-live) HAS_LIVE_FLAG=1 ;;
  esac
done
if [[ $HAS_LIVE_FLAG -eq 0 ]]; then
  ARGS+=(--skip-live)
fi

python -m sim_acp.scripts.record_github_media "${ARGS[@]}"
echo "[done] docs/media/ → commit these files for GitHub README embeds"
