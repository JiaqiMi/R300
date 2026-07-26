#!/usr/bin/env bash
set -u
LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_camera.log"
{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_camera.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"
  source "$HOME/venvs/yolo26/bin/activate"
  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
  export PYTHONUNBUFFERED=1
  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "which python3=$(which python3)"
  echo "start: $HOME/r300_ws/scripts/start_r300.sh web"
  exec "$HOME/r300_ws/scripts/start_r300.sh" web
} 2>&1 | tee -a "$LOG_FILE"
