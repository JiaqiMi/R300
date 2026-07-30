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
  echo "启动：MID-360 + FAST-LIO + GPU高程计算（不启动RViz、不向Web传输点云/高程）"
  echo "需要浏览器显示时，请单独点击‘显示点云 / 高程’。"
  exec roslaunch r300_web_dashboard r300_lidar_web.launch \
    start_sensor_stack:=true \
    start_web_adapter:=false
} 2>&1 | tee -a "$LOG_FILE"
