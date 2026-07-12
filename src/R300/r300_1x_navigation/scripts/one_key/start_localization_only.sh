#!/usr/bin/env bash
set -euo pipefail

# 一键启动：只启动 1X 惯导定位，不启动 move_base / scout_base。
# 用法：
#   ./start_localization_only.sh
#   INS_PORT=/dev/ttyUSB0 ./start_localization_only.sh

WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
INS_PORT="${INS_PORT:-/dev/ttyACM0}"
INS_BAUD="${INS_BAUD:-460800}"
FULL_ATT="${FULL_ATT:-false}"
ODOM_PATH="${ODOM_PATH:-true}"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

cleanup() {
  rosnode kill /odom_to_path >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$ODOM_PATH" == "true" ]]; then
  rosrun "$PKG" odom_to_path.py _odom_topic:=/one_x/odom _path_topic:=/one_x/path _max_points:=5000 &
  sleep 0.5
fi

roslaunch "$PKG" one_x_localization_only.launch \
  serial_port:="$INS_PORT" \
  baudrate:="$INS_BAUD" \
  publish_full_attitude:="$FULL_ATT"
