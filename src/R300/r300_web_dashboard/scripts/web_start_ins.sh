#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_ins.log"
NAV_DIR="$HOME/r300_ws/src/R300/r300_1x_navigation/scripts/one_key"
INS_SCRIPT="$NAV_DIR/start_1x.sh"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_ins.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
  export PYTHONUNBUFFERED=1

  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "INS_SCRIPT=$INS_SCRIPT"

  if [[ ! -f "$INS_SCRIPT" ]]; then
    echo "未找到导航（1X）启动脚本：$INS_SCRIPT"
    exit 1
  fi
  if [[ ! -x "$INS_SCRIPT" ]]; then
    echo "脚本没有执行权限，正在 chmod +x"
    chmod +x "$INS_SCRIPT"
  fi

  cd "$NAV_DIR"
  echo "start: ./start_1x.sh"
  exec ./start_1x.sh
} 2>&1 | tee -a "$LOG_FILE"
