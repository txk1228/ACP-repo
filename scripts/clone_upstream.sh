#!/usr/bin/env bash
# 克隆 ACP 官方代码到 adaptive_compliance_policy/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/adaptive_compliance_policy"

if [[ -d "$TARGET/.git" ]]; then
    echo "[skip] adaptive_compliance_policy 已存在，执行 git pull 更新"
    git -C "$TARGET" pull --recurse-submodules
else
    echo "[clone] yifan-hou/adaptive_compliance_policy"
    git clone --recursive https://github.com/yifan-hou/adaptive_compliance_policy.git "$TARGET"
fi

echo "[done] $TARGET"
