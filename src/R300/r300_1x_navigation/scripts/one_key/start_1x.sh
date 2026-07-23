#!/usr/bin/env bash
# Start only the 1X serial parser.  Navigation is started separately.
#
# Default behaviour is deferred origin initialisation:
#   1. raw 1X topics start immediately;
#   2. /one_x/odom and odom->base_link are held back;
#   3. a navigation start script calls /one_x/set_current_origin;
#   4. the latest valid 1X position becomes the ENU origin.

set -Eeuo pipefail

WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
INS_PORT="${INS_PORT:-/dev/ttyACM0}"
INS_BAUD="${INS_BAUD:-460800}"
FULL_ATT="${FULL_ATT:-false}"
ORIGIN_MODE="${ORIGIN_MODE:-deferred}"
ORIGIN_MAX_AGE="${ORIGIN_MAX_AGE:-0.50}"
READY_TIMEOUT="${READY_TIMEOUT:-30}"
LOG_DIR="${LOG_DIR:-$WS/log/one_x}"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }

[[ -f /opt/ros/noetic/setup.bash ]] || { error "未找到 ROS Noetic"; exit 1; }
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
[[ -f "$WS/devel/setup.bash" ]] || { error "未找到 $WS/devel/setup.bash，请先编译"; exit 1; }
# shellcheck disable=SC1090
source "$WS/devel/setup.bash"

rospack find "$PKG" >/dev/null 2>&1 || { error "ROS 找不到功能包 $PKG"; exit 1; }

[[ -e "$INS_PORT" ]] || { error "惯导串口不存在：$INS_PORT"; exit 1; }
[[ -r "$INS_PORT" && -w "$INS_PORT" ]] || {
  error "当前用户没有 $INS_PORT 的读写权限"; exit 1;
}

if rosnode list 2>/dev/null | grep -qx '/one_x_serial_driver'; then
  error "/one_x_serial_driver 已经运行，不能重复打开同一个串口。"
  exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/one_x_$(date +%Y%m%d_%H%M%S).log"
ROSLAUNCH_PID=""

cleanup() {
  local rc=$?
  trap - INT TERM EXIT
  if [[ -n "$ROSLAUNCH_PID" ]] && kill -0 "$ROSLAUNCH_PID" 2>/dev/null; then
    info "正在停止 1X 解析……"
    kill -INT "$ROSLAUNCH_PID" 2>/dev/null || true
    wait "$ROSLAUNCH_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup INT TERM EXIT

info "启动 1X 串口解析：port=$INS_PORT baud=$INS_BAUD"
info "原点模式：$ORIGIN_MODE"
info "日志文件：$LOG_FILE"

roslaunch "$PKG" one_x_localization_only.launch \
  serial_port:="$INS_PORT" \
  baudrate:="$INS_BAUD" \
  publish_full_attitude:="$FULL_ATT" \
  origin_mode:="$ORIGIN_MODE" \
  origin_set_max_age_s:="$ORIGIN_MAX_AGE" \
  > >(tee "$LOG_FILE") 2>&1 &
ROSLAUNCH_PID=$!

for _ in $(seq 1 $((READY_TIMEOUT * 5))); do
  if rosnode list 2>/dev/null | grep -qx '/one_x_serial_driver'; then
    break
  fi
  kill -0 "$ROSLAUNCH_PID" 2>/dev/null || {
    error "1X launch 已退出，请查看：$LOG_FILE"; exit 1;
  }
  sleep 0.2
done

rosnode list 2>/dev/null | grep -qx '/one_x_serial_driver' || {
  error "等待 ${READY_TIMEOUT}s 后未发现 /one_x_serial_driver"; exit 1;
}

if ! timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/ins_fix >/dev/null 2>&1; then
  error "没有收到 /one_x/ins_fix，请检查串口、波特率和1X输出。"
  exit 1
fi

for _ in $(seq 1 $((READY_TIMEOUT * 5))); do
  rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' && break
  sleep 0.2
done
rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' || {
  error "未发现 /one_x/set_current_origin 服务"; exit 1;
}

ok "1X 原始解析已就绪"
if [[ "$ORIGIN_MODE" == "deferred" ]]; then
  warn "当前尚未建立导航原点；这是正常状态。"
  echo "启动视觉导航或实车导航时，会自动调用："
  echo "  rosservice call /one_x/set_current_origin \"{}\""
else
  echo "当前 origin_mode=$ORIGIN_MODE，/one_x/odom 会按该模式发布。"
fi

echo
echo "可查看："
echo "  rostopic hz /one_x/ins_fix"
echo "  rostopic echo -n 1 /one_x/attitude"
echo "  rostopic echo -n 1 /one_x/ins_status"
echo "停止1X：在本终端按 Ctrl+C"
echo

wait "$ROSLAUNCH_PID"
