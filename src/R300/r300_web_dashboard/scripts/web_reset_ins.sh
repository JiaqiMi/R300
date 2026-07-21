#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
if [ -f "$HOME/r300_ws/devel/setup.bash" ]; then
  source "$HOME/r300_ws/devel/setup.bash"
fi

TOPIC="/one_x/command_hex"
RESET_CMD="55 AA 55 AA 5A A5 5A A5 BB 78 56 34 12 78 56 34 12"

echo "[INS] 检查命令话题 ${TOPIC}"
if ! rostopic list 2>/dev/null | grep -qx "${TOPIC}"; then
  echo "[INS] 未发现 ${TOPIC}"
  echo "[INS] 请先编译并重启 one_x_serial_driver：catkin_make --pkg r300_1x_navigation"
  exit 2
fi

echo "[INS] 发送系统复位命令：${RESET_CMD}"
rostopic pub -1 "${TOPIC}" std_msgs/String "data: '${RESET_CMD}'"
echo "[INS] 已发送。惯导复位后会重新启动并重新对准，请等待状态重新进入对准/导航。"
