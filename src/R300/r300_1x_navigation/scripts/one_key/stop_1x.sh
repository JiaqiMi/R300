#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi
rosnode kill /one_x_serial_driver >/dev/null 2>&1 || true
echo "[INFO] 已请求停止 /one_x_serial_driver。"
