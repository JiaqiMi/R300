#!/usr/bin/env bash
set -u
LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_nav.log"
SUDO_PASS="1234"
NAV_DIR="$HOME/r300_ws/src/R300/r300_1x_navigation/scripts/one_key"
NAV_SCRIPT="$NAV_DIR/start_r300_vision_nav.sh"
{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_nav.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"
  source "$HOME/venvs/yolo26/bin/activate"
  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
  export PYTHONUNBUFFERED=1
  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "which python3=$(which python3)"
  echo "NAV_SCRIPT=$NAV_SCRIPT"
  if [ ! -x "$NAV_SCRIPT" ]; then
    echo "脚本没有执行权限，正在 chmod +x"
    chmod +x "$NAV_SCRIPT"
  fi
  echo "验证 sudo 密码缓存..."
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v
  if [ $? -ne 0 ]; then
    echo "sudo 密码验证失败，请确认密码是否为 1234"
    exit 1
  fi
  while true; do
    sudo -n -v 2>/dev/null || true
    sleep 45
  done &
  KEEPALIVE_PID=$!
  trap 'kill $KEEPALIVE_PID 2>/dev/null || true' EXIT INT TERM
  cd "$NAV_DIR"
  echo "start: ./start_r300_vision_nav.sh --no-rviz"
  exec ./start_r300_vision_nav.sh --no-rviz
} 2>&1 | tee -a "$LOG_FILE"
