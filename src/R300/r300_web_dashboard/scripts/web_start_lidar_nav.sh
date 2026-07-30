#!/usr/bin/env bash
# Web 按钮包装层：只负责日志、sudo 缓存和调用导航包的一键脚本。
# 雷达导航检查、原点设置、CAN、roslaunch 与全链路自检均位于：
#   r300_1x_navigation/scripts/one_key/start_r300_lidar_nav.sh
#
# SIGN_GUIDANCE_ENABLED=true：
#   需要相机/YOLO，启动视觉指示牌临时转向。
# SIGN_GUIDANCE_ENABLED=false：
#   不依赖相机/YOLO，保持纯雷达避障导航。

set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_lidar_nav.log"

# R300_LIDAR_LOG_ROTATE_V1
# 当前日志超过5 MiB时滚动一次；仅保留当前文件和 .1，避免无限增长。
if [[ -f "$LOG_FILE" ]]; then
  LOG_SIZE="$(stat -c '%s' "$LOG_FILE" 2>/dev/null || echo 0)"
  if [[ "$LOG_SIZE" =~ ^[0-9]+$ ]] && (( LOG_SIZE > 5 * 1024 * 1024 )); then
    mv -f "$LOG_FILE" "${LOG_FILE}.1"
  fi
fi
WS="${R300_WS:-$HOME/r300_ws}"
SUDO_PASS="${R300_SUDO_PASS:-1234}"
SIGN_GUIDANCE_ENABLED="${SIGN_GUIDANCE_ENABLED:-true}"
KEEPALIVE_PID=""

case "$SIGN_GUIDANCE_ENABLED" in
  true|false) ;;
  *)
    echo "SIGN_GUIDANCE_ENABLED 只能是 true 或 false：$SIGN_GUIDANCE_ENABLED" >&2
    exit 2
    ;;
esac

cleanup() {
  if [[ -n "$KEEPALIVE_PID" ]] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
    kill "$KEEPALIVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_lidar_nav.sh"
  echo "说明：Web 仅调用 r300_1x_navigation 的正式雷达导航脚本。"
  echo "视觉指示牌临时转向：$SIGN_GUIDANCE_ENABLED"
  # 2026-07-30 操作员定：Web 链与命令行(od2/nr_od)一致，默认不开启限速器执行。
  # 由 start_r300_lidar_nav.sh 的 SPEED_LIMIT_APPLY 默认 false 实现；
  # 需要恢复限速时以 SPEED_LIMIT_APPLY=true 环境变量启动本脚本。
  echo "限速器执行：${SPEED_LIMIT_APPLY:-false}（默认关闭）"

  source /opt/ros/noetic/setup.bash
  source "$WS/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  NAV_PKG_PATH="$(rospack find r300_1x_navigation)"
  NAV_SCRIPT="$NAV_PKG_PATH/scripts/one_key/start_r300_lidar_nav.sh"
  [[ -f "$NAV_SCRIPT" ]] || {
    echo "未找到雷达导航脚本：$NAV_SCRIPT"
    exit 1
  }
  [[ -x "$NAV_SCRIPT" ]] || chmod +x "$NAV_SCRIPT"

  echo "NAV_SCRIPT=$NAV_SCRIPT"
  echo "验证 sudo 密码缓存（仅用于配置 CAN）..."
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v

  # start_r300_lidar_nav.sh 在 Web 模式下使用 sudo -n；保持缓存避免长时间自检后失效。
  while true; do
    sudo -n -v 2>/dev/null || true
    sleep 45
  done &
  KEEPALIVE_PID=$!

  export R300_NONINTERACTIVE_SUDO=true
  if [[ "$SIGN_GUIDANCE_ENABLED" == "true" ]]; then
    echo "启动：start_r300_lidar_nav.sh --no-rviz --sign-guidance"
    echo "要求：相机、YOLO 与 /r300_vision/detections 已运行。"
    NAV_ARGS=(--no-rviz --sign-guidance)
  else
    echo "启动：start_r300_lidar_nav.sh --no-rviz --no-sign-guidance"
    echo "说明：本次不依赖相机/YOLO，保持纯雷达避障。"
    NAV_ARGS=(--no-rviz --no-sign-guidance)
  fi

  echo "说明：不会自动执行航点；就绪后在网页点击‘开始航点’。"
  exec "$NAV_SCRIPT" "${NAV_ARGS[@]}"
} 2>&1 | tee -p -a "$LOG_FILE"
