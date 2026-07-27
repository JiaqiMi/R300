#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_stop_lidar_nav.log"
NAV_DIR="$HOME/r300_ws/src/R300/r300_1x_navigation/scripts/one_key"
STOP_SCRIPT="$NAV_DIR/stop_r300_nav.sh"
SUDO_PASS="${R300_SUDO_PASS:-1234}"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_stop_lidar_nav.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  if [[ -f "$HOME/r300_ws/devel/setup.bash" ]]; then
    source "$HOME/r300_ws/devel/setup.bash"
  fi
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  [[ -f "$STOP_SCRIPT" ]] || {
    echo "未找到导航停止脚本：$STOP_SCRIPT"
    exit 1
  }
  [[ -x "$STOP_SCRIPT" ]] || chmod +x "$STOP_SCRIPT"

  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v

  cd "$NAV_DIR"
  echo "停止雷达避障导航：./stop_r300_nav.sh"
  echo "说明：只停止导航，保留独立 1X 和雷达高程感知。"
  ./stop_r300_nav.sh
  echo "雷达避障导航停止完成"
} 2>&1 | tee -a "$LOG_FILE"
