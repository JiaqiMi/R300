#!/usr/bin/env bash
set -euo pipefail

# 一键启动：实车导航（纯 DWA 链路）
# 默认不自动开始航点，避免上电后车辆直接运动。
# 用法：
#   ./start_real_nav.sh
#   INS_PORT=/dev/ttyUSB0 CAN_PORT=can0 AUTO_START=true ./start_real_nav.sh
#   LAUNCH_LIDAR=false LAUNCH_RVIZ=true ./start_real_nav.sh

WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
INS_PORT="${INS_PORT:-/dev/ttyACM0}"
INS_BAUD="${INS_BAUD:-460800}"
CAN_PORT="${CAN_PORT:-can0}"
AUTO_START="${AUTO_START:-false}"
LAUNCH_BASE="${LAUNCH_BASE:-true}"
LAUNCH_LIDAR="${LAUNCH_LIDAR:-false}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-false}"
ODOM_PATH="${ODOM_PATH:-true}"
MAX_GOAL_DIST="${MAX_GOAL_DIST:-180.0}"
SETUP_CAN="${SETUP_CAN:-true}"
CAN_BITRATE="${CAN_BITRATE:-500000}"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

if ! rospack find "$PKG" >/dev/null 2>&1; then
  echo "[ERROR] 找不到 ROS 包 $PKG，请检查 R300_WS=$WS 是否正确。"
  exit 1
fi

if [[ ! -e "$INS_PORT" ]]; then
  echo "[WARN] 惯导串口 $INS_PORT 不存在。请检查 INS_PORT，例如 INS_PORT=/dev/ttyUSB0。"
fi

if [[ "$SETUP_CAN" == "true" ]]; then
  echo "[INFO] 配置 CAN: $CAN_PORT bitrate=$CAN_BITRATE"
  sudo ip link set "$CAN_PORT" down >/dev/null 2>&1 || true
  sudo ip link set "$CAN_PORT" type can bitrate "$CAN_BITRATE"
  sudo ip link set "$CAN_PORT" up
fi

cleanup() {
  echo "[INFO] 停止辅助节点..."
  rosnode kill /odom_to_path >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$ODOM_PATH" == "true" ]]; then
  rosrun "$PKG" odom_to_path.py _odom_topic:=/one_x/odom _path_topic:=/one_x/path _max_points:=5000 &
  sleep 0.5
fi

echo "[INFO] 启动实车导航。启动后如 AUTO_START=false，请手动执行："
echo "       rosservice call /subject1/start_waypoints"

roslaunch "$PKG" subject1_waypoint_nav.launch \
  ins_serial_port:="$INS_PORT" \
  ins_baudrate:="$INS_BAUD" \
  can_port:="$CAN_PORT" \
  launch_base:="$LAUNCH_BASE" \
  launch_lidar:="$LAUNCH_LIDAR" \
  launch_rviz:="$LAUNCH_RVIZ" \
  auto_start:="$AUTO_START" \
  max_goal_distance_from_origin_m:="$MAX_GOAL_DIST"
