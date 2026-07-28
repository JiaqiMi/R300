#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_stop_target_recorder.log"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_stop_target_recorder.sh"

  source /opt/ros/noetic/setup.bash
  source "$HOME/r300_ws/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

  echo "停止比赛目标图片记录器，保留已经保存的图片、JSON 和 CSV。"

  # 先停止 roslaunch，避免 launch 中 respawn=true 把节点重新拉起。
  pkill -INT -f '[r]oslaunch.*r300_yolo_detector.*target_snapshot_recorder.launch' 2>/dev/null || true
  sleep 1

  if rosnode list 2>/dev/null | grep -qx '/r300_target_snapshot_recorder'; then
    rosnode kill /r300_target_snapshot_recorder 2>/dev/null || true
  fi

  for _ in {1..20}; do
    if ! rosnode list 2>/dev/null | grep -qx '/r300_target_snapshot_recorder'; then
      echo "比赛目标图片记录器已停止。"
      exit 0
    fi
    sleep 0.25
  done

  pkill -TERM -f '[t]arget_snapshot_recorder.py.*__name:=r300_target_snapshot_recorder' 2>/dev/null || true
  sleep 1

  if rosnode list 2>/dev/null | grep -qx '/r300_target_snapshot_recorder'; then
    echo "记录器仍未退出。"
    exit 1
  fi

  echo "比赛目标图片记录器已停止。"
} 2>&1 | tee -a "$LOG_FILE"
