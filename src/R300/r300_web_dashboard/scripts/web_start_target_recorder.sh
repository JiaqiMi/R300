#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_target_recorder.log"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_target_recorder.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  if rosnode list 2>/dev/null | grep -qx '/r300_target_snapshot_recorder'; then
    echo "比赛目标图片记录器已经运行。"
    exit 0
  fi

  echo "启动比赛目标图片记录器："
  echo "  输入图像：/r300_vision/annotated_image"
  echo "  检测结果：/r300_vision/detections"
  echo "  定位：/one_x/fix"
  echo "  航向：/one_x/heading_deg"
  echo "  输出目录：/home/explorer/r300_target_records"

  exec roslaunch r300_yolo_detector target_snapshot_recorder.launch
} 2>&1 | tee -a "$LOG_FILE"
