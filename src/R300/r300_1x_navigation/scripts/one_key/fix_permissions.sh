#!/usr/bin/env bash
set -euo pipefail

# 一键修复功能包 Python 与 one_key Shell 脚本执行权限。
WS="${R300_WS:-$HOME/r300_ws}"
PKG_DIR="${PKG_DIR:-$WS/src/R300/r300_1x_navigation}"

chmod +x "$PKG_DIR"/scripts/*.py 2>/dev/null || true
chmod +x "$PKG_DIR"/scripts/*.sh 2>/dev/null || true
chmod +x "$PKG_DIR"/scripts/one_key/*.sh 2>/dev/null || true

echo "[INFO] 已修复 $PKG_DIR/scripts 与 scripts/one_key 下脚本执行权限。"
find "$PKG_DIR/scripts" -maxdepth 2 -type f -name '*.sh' -printf '%m %p\n' | sort
