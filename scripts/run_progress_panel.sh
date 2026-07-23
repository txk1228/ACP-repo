#!/usr/bin/env bash
# 一键启动训练进度可视化面板 → http://127.0.0.1:8765
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8765
PID_FILE="${TMPDIR:-/tmp}/acp_train_progress_panel.pid"

# 端口已被占用：若是本面板则直接提示；否则报错
if ss -ltn 2>/dev/null | grep -q ":${PORT} " || netstat -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已在监听 — 面板可能已在运行。"
  echo "浏览器打开: http://127.0.0.1:${PORT}"
  echo "若要重启: 先结束占用进程，例如  fuser -k ${PORT}/tcp"
  exit 0
fi

cd "$REPO_ROOT"
# 标准库即可，无需 pyrite / GPU
nohup python3 "$REPO_ROOT/scripts/train_progress_server.py" \
  >"$REPO_ROOT/logs/train_progress_panel.log" 2>&1 &
echo $! >"$PID_FILE"

sleep 0.4
echo "训练进度面板已启动 (pid=$(cat "$PID_FILE"))"
echo "→ http://127.0.0.1:${PORT}"
echo "日志: $REPO_ROOT/logs/train_progress_panel.log"
echo "停止: kill \$(cat $PID_FILE)  或  fuser -k ${PORT}/tcp"
