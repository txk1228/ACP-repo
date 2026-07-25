#!/usr/bin/env bash
# 生成 GitHub 展示媒体：docs/media/sim_flip_v2ft.{mp4,gif}
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

python -m sim_acp.scripts.record_github_media "$@"
echo "[done] docs/media/ → commit these files for GitHub README embeds"
