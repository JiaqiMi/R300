#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_stop_real_nav.log"
NAV_DIR="$HOME/r300_ws/src/R300/r300_1x_navigation/scripts/one_key"
STOP_SCRIPT="$NAV_DIR/stop_r300_nav.sh"
SUDO_PASS="${R300_SUDO_PASS:-1234}"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_stop_real_nav.sh"
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
    echo "未找到导航停止脚本：$STOP_SCRIPT"
    exit 1
  fi
  if [[ ! -x "$STOP_SCRIPT" ]]; then
    chmod +x "$STOP_SCRIPT"
  fi

  # 与视觉避障启动方式一致，后台预先验证 sudo，避免网页按钮等待密码。
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v

  cd "$NAV_DIR"
  echo "stop pure real navigation: ./stop_r300_nav.sh"
  echo "说明：只停止导航，保留独立运行的 1X。"
  ./stop_r300_nav.sh
  echo "纯实车导航停止完成，1X 保持运行"
} 2>&1 | tee -a "$LOG_FILE"
