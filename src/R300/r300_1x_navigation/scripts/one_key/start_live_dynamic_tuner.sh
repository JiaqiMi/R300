#!/usr/bin/env bash
# Start a minimal Web UI that attaches to an already-running move_base stack.
# It does not start/stop navigation and does not modify YAML files.

set -Eeuo pipefail

WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8072}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-30}"
DYNAMIC_SERVERS="${DYNAMIC_SERVERS:-}"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }

[[ -f /opt/ros/noetic/setup.bash ]] || { error "未找到 ROS Noetic"; exit 1; }
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
[[ -f "$WS/devel/setup.bash" ]] || { error "未找到 $WS/devel/setup.bash，请先编译工作空间"; exit 1; }
# shellcheck disable=SC1090
source "$WS/devel/setup.bash"

PKG_PATH="$(rospack find "$PKG" 2>/dev/null || true)"
[[ -n "$PKG_PATH" ]] || { error "ROS 找不到功能包 $PKG"; exit 1; }
NODE="$PKG_PATH/scripts/live_dynamic_reconfigure_web.py"
[[ -f "$NODE" ]] || { error "未找到 $NODE"; exit 1; }
[[ -x "$NODE" ]] || chmod +x "$NODE"

if ! rosparam get /rosdistro >/dev/null 2>&1; then
  error "ROS master 未运行。请先启动：./start_r300_vision_nav.sh --no-rviz"
  exit 1
fi

info "等待 DWA dynamic_reconfigure 服务……"
for ((i=0; i<WAIT_TIMEOUT; i++)); do
  if rosservice list 2>/dev/null | grep -qx '/move_base/DWAPlannerROS/set_parameters'; then
    ok "检测到 /move_base/DWAPlannerROS"
    break
  fi
  if (( i == WAIT_TIMEOUT - 1 )); then
    error "等待 ${WAIT_TIMEOUT}s 后仍未发现 DWA 动态服务。请确认 move_base 已启动。"
    exit 1
  fi
  sleep 1
done

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${WEB_PORT}$"; then
  error "端口 $WEB_PORT 已被占用。可改用：WEB_PORT=8072 ./start_live_dynamic_tuner.sh"
  exit 1
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -n "$IP" ]] || IP="工控机IP"

printf '\n'
ok "在线调参页面即将启动"
printf '  浏览器地址：\033[1;36mhttp://%s:%s\033[0m\n' "$IP" "$WEB_PORT"
printf '  停止页面：当前终端按 Ctrl+C（不会停止导航）\n'
printf '  说明：网页只修改运行参数，不写 YAML。\n\n'

ARGS=("--host" "$WEB_HOST" "--port" "$WEB_PORT")
if [[ -n "$DYNAMIC_SERVERS" ]]; then
  ARGS+=("--servers" "$DYNAMIC_SERVERS")
fi

exec python3 "$NODE" "${ARGS[@]}"
