#!/usr/bin/env bash
# Web 按钮包装层：调用 r300_1x_navigation 的公共导航停止脚本。
# 仅停止 move_base/DWA/底盘/雷达障碍适配器，保留独立 1X 和雷达感知栈。

set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_stop_lidar_nav.log"
WS="${R300_WS:-$HOME/r300_ws}"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_stop_lidar_nav.sh"

  source /opt/ros/noetic/setup.bash
  [[ -f "$WS/devel/setup.bash" ]] && source "$WS/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

  NAV_PKG_PATH="$(rospack find r300_1x_navigation)"
  STOP_SCRIPT="$NAV_PKG_PATH/scripts/one_key/stop_r300_nav.sh"
  [[ -f "$STOP_SCRIPT" ]] || {
    echo "未找到导航停止脚本：$STOP_SCRIPT"
    exit 1
  }
  [[ -x "$STOP_SCRIPT" ]] || chmod +x "$STOP_SCRIPT"

  echo "停止雷达避障导航：$STOP_SCRIPT"
  "$STOP_SCRIPT"
  echo "雷达避障导航停止完成；1X 和雷达高程感知保持运行。"
} 2>&1 | tee -a "$LOG_FILE"
