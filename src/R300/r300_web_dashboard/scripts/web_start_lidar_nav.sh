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
  # 2026-07-30 晚：限速器升级为选择性记账后默认重新开启——
  # 正障碍(墙/桩)不减速(DWA+快反层全权), 只对 坑/陡坡/水坑空洞 减速。
  # 退回纯观测影子模式: SPEED_LIMIT_APPLY=false 环境变量启动本脚本。
  echo "限速器执行：${SPEED_LIMIT_APPLY:-true}（选择性: 墙不限速, 坑/水减速）"

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

  # 2026-07-30 操作员需求：Web 启动导航后在板载屏幕同步拉起 rviz(雷达导航配置)。
  # 后台等 /move_base 注册(至多150s)再起，rviz 用 setsid 脱离进程组——
  # Web 的进程组清理只影响导航本体，rviz 由 web_stop_lidar_nav.sh 显式关闭。
  RVIZ_CONFIG="$WS/src/R300/r300_1x_navigation/config/subject1_lidar_nav.rviz"
  (
    for _ in $(seq 1 150); do
      rosnode list 2>/dev/null | grep -qx /move_base && break
      sleep 1
    done
    if rosnode list 2>/dev/null | grep -qx /move_base; then
      pkill -x rviz 2>/dev/null || true
      sleep 1
      echo "[$(date '+%F %T')] /move_base 已就绪，板载屏幕拉起 rviz"
      setsid env DISPLAY="${RVIZ_DISPLAY:-:0}" DISABLE_ROS1_EOL_WARNINGS=1 \
        rviz -d "$RVIZ_CONFIG" >>"$LOG_DIR/web_rviz.log" 2>&1 </dev/null &
    else
      echo "[$(date '+%F %T')] 150s 内未见 /move_base，本次不拉起 rviz"
    fi
  ) &

  exec "$NAV_SCRIPT" "${NAV_ARGS[@]}"
} 2>&1 | tee -p -a "$LOG_FILE"
