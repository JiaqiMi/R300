#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_lidar_display.log"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_lidar_display.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
  export PYTHONUNBUFFERED=1

  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "启动：点云/高程 Web 显示适配器"
  echo "传输方式：ROS std_msgs/String -> rosbridge WebSocket 9090（不是8080）"
  echo "源话题不存在时适配器会等待；不会启动或修改雷达导航。"
  exec roslaunch r300_web_dashboard r300_lidar_web.launch \
    start_sensor_stack:=false \
    start_web_adapter:=true
} 2>&1 | tee -a "$LOG_FILE"
