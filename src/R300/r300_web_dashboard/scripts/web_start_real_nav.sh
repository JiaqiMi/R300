#!/usr/bin/env bash
set -u

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_real_nav.log"
SUDO_PASS="${R300_SUDO_PASS:-1234}"
NAV_DIR="$HOME/r300_ws/src/R300/r300_1x_navigation/scripts/one_key"
NAV_SCRIPT="$NAV_DIR/start_real_nav.sh"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_real_nav.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "NAV_SCRIPT=$NAV_SCRIPT"

  if [[ ! -f "$NAV_SCRIPT" ]]; then
    echo "未找到纯实车导航脚本：$NAV_SCRIPT"
    exit 1
  fi
  if [[ ! -x "$NAV_SCRIPT" ]]; then
    echo "脚本没有执行权限，正在 chmod +x"
    chmod +x "$NAV_SCRIPT"
  fi

  # 与已正常工作的视觉避障/代价地图按钮保持相同 sudo 处理。
  echo "验证 sudo 密码缓存（用于配置 CAN）..."
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v
  if [[ $? -ne 0 ]]; then
    echo "sudo 密码验证失败，请确认密码是否为 1234"
    exit 1
  fi

  while true; do
    sudo -n -v 2>/dev/null || true
    sleep 45
  done &
  KEEPALIVE_PID=$!
  trap 'kill "$KEEPALIVE_PID" 2>/dev/null || true' EXIT INT TERM

  cd "$NAV_DIR"
  echo "start pure real navigation: ./start_real_nav.sh --no-rviz"
  echo "说明：本按钮只启动纯实车导航，不自动执行航点；就绪后请在网页点击‘开始航点’。"
  exec ./start_real_nav.sh --no-rviz
} 2>&1 | tee -a "$LOG_FILE"
