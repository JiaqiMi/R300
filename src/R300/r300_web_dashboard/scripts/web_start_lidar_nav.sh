#!/usr/bin/env bash
# Web 按钮包装层：只负责日志、sudo 缓存和调用导航包的一键脚本。
# 雷达导航检查、原点设置、CAN、roslaunch 与全链路自检均位于：
#   r300_1x_navigation/scripts/one_key/start_r300_lidar_nav.sh

set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_lidar_nav.log"

WS="${R300_WS:-$HOME/r300_ws}"
SUDO_PASS="${R300_SUDO_PASS:-1234}"
KEEPALIVE_PID=""

cleanup() {
  if [[ -n "$KEEPALIVE_PID" ]] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
    kill "$KEEPALIVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_lidar_nav.sh"
  echo "说明：Web 仅调用 r300_1x_navigation 的正式雷达导航脚本。"

  source /opt/ros/noetic/setup.bash
  source "$WS/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  NAV_PKG_PATH="$(rospack find r300_1x_navigation)"
  NAV_SCRIPT="$NAV_PKG_PATH/scripts/one_key/start_r300_lidar_nav.sh"
  [[ -f "$NAV_SCRIPT" ]] || {
    echo "未找到雷达导航脚本：$NAV_SCRIPT"
    exit 1
  }
  [[ -x "$NAV_SCRIPT" ]] || chmod +x "$NAV_SCRIPT"

  echo "NAV_SCRIPT=$NAV_SCRIPT"
  echo "验证 sudo 密码缓存（仅用于配置 CAN）..."
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v

  # start_r300_lidar_nav.sh 在 Web 模式下使用 sudo -n；保持缓存避免长时间自检后失效。
  while true; do
    sudo -n -v 2>/dev/null || true
    sleep 45
  done &
  KEEPALIVE_PID=$!

  export R300_NONINTERACTIVE_SUDO=true
  echo "启动：start_r300_lidar_nav.sh --no-rviz"
  echo "说明：不会自动执行航点；就绪后在网页点击‘开始航点’。"
  exec "$NAV_SCRIPT" --no-rviz
} 2>&1 | tee -a "$LOG_FILE"
