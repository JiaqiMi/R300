#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_stop_ins.log"
NAV_DIR="$HOME/r300_ws/src/R300/r300_1x_navigation/scripts/one_key"
STOP_SCRIPT="$NAV_DIR/stop_1x.sh"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_stop_ins.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  if [[ -f "$HOME/r300_ws/devel/setup.bash" ]]; then
    source "$HOME/r300_ws/devel/setup.bash"
  fi
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "STOP_SCRIPT=$STOP_SCRIPT"

  if [[ ! -f "$STOP_SCRIPT" ]]; then
    echo "未找到 1X 停止脚本：$STOP_SCRIPT"
    exit 1
  fi
  if [[ ! -x "$STOP_SCRIPT" ]]; then
    chmod +x "$STOP_SCRIPT"
  fi

  cd "$NAV_DIR"
  echo "stop: ./stop_1x.sh"
  ./stop_1x.sh
  echo "1X 停止完成"
} 2>&1 | tee -a "$LOG_FILE"
