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

  # 2026-07-30 与命令行流程(od1)对齐: 复用同一份感知启动脚本——
  # 自动识别雷达网口(192.168.1.50 所在口, eth1/eth2 漂移免疫)、
  # RESTART_EXISTING=1 幂等重启、RVIZ=0。此前直接 roslaunch 跳过网口检查,
  # 网口未就绪时按钮"成功"但雷达永远无数据(历史上"web 启不起来"的根源之一)。
  PERCEP_SCRIPT="$HOME/r300_ws/scripts/start_single_lidar_elevation.sh"
  if [[ ! -f "$PERCEP_SCRIPT" ]]; then
    echo "未找到感知启动脚本：$PERCEP_SCRIPT"
    exit 1
  fi
  LIF="$(ip -4 -br addr | awk '/192\.168\.1\.50/{print $1; exit}')"
  if [[ -z "$LIF" ]]; then
    echo "未找到持有 192.168.1.50 的网口——请检查雷达网线/网口配置后重试。"
    exit 1
  fi
  echo "雷达网口: $LIF"
  echo "启动：MID-360 + FAST-LIO + GPU高程计算（不启动RViz、不向Web传输点云/高程）"
  echo "需要浏览器显示时，请单独点击‘显示点云 / 高程’。"
  exec env LIDAR_INTERFACE="$LIF" RVIZ=0 RESTART_EXISTING=1 bash "$PERCEP_SCRIPT"
} 2>&1 | tee -a "$LOG_FILE"
