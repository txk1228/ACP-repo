#!/usr/bin/env bash
# 可选：从官方仓库刷新 adaptive_compliance_policy/
# 注意：本仓库已直接纳入该目录（含本地复现改动）。直接 pull 官方可能覆盖
# 你的修改；仅在你明确要与上游对齐时使用。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/adaptive_compliance_policy"
UPSTREAM_URL="https://github.com/yifan-hou/adaptive_compliance_policy.git"

echo "警告: adaptive_compliance_policy/ 已纳入 ACP-repo 版本管理。"
echo "本脚本仅用于对照/临时拉取官方树，不会自动覆盖已跟踪文件。"
echo ""

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[clone] $UPSTREAM_URL -> $TMP/upstream"
git clone --depth 1 --recursive "$UPSTREAM_URL" "$TMP/upstream"

echo "[done] 官方快照在: $TMP/upstream"
echo "如需对比: diff -ruN \"$TMP/upstream\" \"$TARGET\" | less"
echo "请勿直接 rm -rf \"$TARGET\" 后整目录替换，以免丢失本地 Conv/Spec 复现改动。"
