#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_lidar_elevation.log"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_lidar_elevation.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
  export PYTHONUNBUFFERED=1

  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "检查 single_lidar_elevation 包..."
  rospack find single_lidar_elevation
  LIDAR_SCRIPT="$HOME/r300_ws/scripts/start_single_lidar_elevation.sh"
  if [[ ! -f "$LIDAR_SCRIPT" ]]; then
    echo "未找到正式雷达启动脚本：$LIDAR_SCRIPT"
    exit 1
  fi
  [[ -x "$LIDAR_SCRIPT" ]] || chmod +x "$LIDAR_SCRIPT"

  echo "启动：MID-360 + FAST-LIO + GPU高程计算（自动查找持有192.168.1.50的网口）"
  echo "不启动RViz、不向Web传输点云/高程；需要显示时请单独点击‘显示点云 / 高程’。"
  exec env RVIZ=0 RESTART_EXISTING=1 bash "$LIDAR_SCRIPT"
} 2>&1 | tee -a "$LOG_FILE"
