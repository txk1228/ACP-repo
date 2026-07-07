#!/usr/bin/env bash
# 将本地改动同步到 GitHub: txk1228/ACP-repo
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MSG="${1:-update: sync local changes}"

if [[ ! -d .git ]]; then
    echo "错误: 当前目录不是 git 仓库"
    exit 1
fi

git add -A
git status --short

if git diff --cached --quiet; then
    echo "没有需要提交的改动"
    exit 0
fi

git commit -m "$MSG"
git push origin main

echo "[done] https://github.com/txk1228/ACP-repo"
