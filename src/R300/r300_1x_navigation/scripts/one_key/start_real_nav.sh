#!/usr/bin/env bash
# Start pure real-vehicle DWA navigation.
# It shares the odometry adapter, waypoint services/status and navigation core
# with visual navigation, but loads config/subject1_dwa.yaml by default.

set -Eeuo pipefail

WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
LAUNCH_FILE="${R300_REAL_LAUNCH:-subject1_waypoint_nav.launch}"

CAN_PORT="${CAN_PORT:-can0}"
CAN_BITRATE="${CAN_BITRATE:-500000}"
WAYPOINT_FILE="${WAYPOINT_FILE:-}"
MAX_GOAL_DIST="${MAX_GOAL_DIST:-5000.0}"

LAUNCH_BASE="${LAUNCH_BASE:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-false}"
ODOM_PATH="${ODOM_PATH:-true}"
SETUP_CAN="${SETUP_CAN:-true}"
# Backward compatibility: AUTO_START=true is treated as AUTO_RUN=true.
AUTO_RUN="${AUTO_RUN:-${AUTO_START:-false}}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
LOG_DIR="${LOG_DIR:-$WS/log/real_nav}"

DWA_ODOM_TOPIC="${DWA_ODOM_TOPIC:-/subject1/dwa_odom}"
DWA_MAX_YAW_RATE_RADPS="${DWA_MAX_YAW_RATE_RADPS:-0.70}"
REAL_DWA_CONFIG="${REAL_DWA_CONFIG:-}"

ROSLAUNCH_PID=""
ODOM_PATH_PID=""
STOP_REQUESTED="false"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }

usage() {
  cat <<'USAGE'
用法：
  ./start_real_nav.sh [选项]

选项：
  --run              全链路就绪后启动航点
  --rviz             启动 RViz
  --no-rviz          不启动 RViz
  --no-base          不启动底盘，适合台架检查
  --no-path          不启动 /one_x/path 辅助节点
  --setup-can        启动前重新配置 CAN
  --no-setup-can     不重新配置 CAN，仅检查接口是否 UP
  --waypoints PATH   指定航点 YAML
  --dwa-config PATH  指定纯实车 DWA YAML，默认 config/subject1_dwa.yaml
  --timeout SEC      单项检查超时，默认 60 秒
  -h, --help         显示帮助

环境变量：
  R300_WS, CAN_PORT, CAN_BITRATE, WAYPOINT_FILE, MAX_GOAL_DIST,
  LAUNCH_BASE, LAUNCH_RVIZ, ODOM_PATH, SETUP_CAN, AUTO_RUN,
  READY_TIMEOUT, DWA_ODOM_TOPIC, DWA_MAX_YAW_RATE_RADPS, REAL_DWA_CONFIG
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) AUTO_RUN="true"; shift ;;
    --rviz) LAUNCH_RVIZ="true"; shift ;;
    --no-rviz) LAUNCH_RVIZ="false"; shift ;;
    --no-base) LAUNCH_BASE="false"; shift ;;
    --no-path) ODOM_PATH="false"; shift ;;
    --setup-can) SETUP_CAN="true"; shift ;;
    --no-setup-can) SETUP_CAN="false"; shift ;;
    --waypoints)
      [[ $# -ge 2 ]] || { error "--waypoints 后缺少路径"; exit 2; }
      WAYPOINT_FILE="$2"; shift 2 ;;
    --dwa-config)
      [[ $# -ge 2 ]] || { error "--dwa-config 后缺少路径"; exit 2; }
      REAL_DWA_CONFIG="$2"; shift 2 ;;
    --timeout)
      [[ $# -ge 2 ]] || { error "--timeout 后缺少秒数"; exit 2; }
      READY_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) error "未知参数：$1"; usage; exit 2 ;;
  esac
done

publish_zero_cmd() {
  if command -v rostopic >/dev/null 2>&1 && \
     rostopic list 2>/dev/null | grep -qx '/subject1/cmd_vel_raw'; then
    timeout 2 rostopic pub -1 /subject1/cmd_vel_raw geometry_msgs/Twist \
      "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
      >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local rc=$?
  trap - INT TERM EXIT

  if [[ "$STOP_REQUESTED" == "false" ]]; then
    STOP_REQUESTED="true"
    info "正在停止纯实车导航……"
    publish_zero_cmd
  fi

  if [[ -n "$ODOM_PATH_PID" ]] && kill -0 "$ODOM_PATH_PID" 2>/dev/null; then
    kill -INT "$ODOM_PATH_PID" 2>/dev/null || true
  fi

  if [[ -n "$ROSLAUNCH_PID" ]] && kill -0 "$ROSLAUNCH_PID" 2>/dev/null; then
    kill -INT "$ROSLAUNCH_PID" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$ROSLAUNCH_PID" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$ROSLAUNCH_PID" 2>/dev/null; then
      warn "roslaunch 未及时退出，发送 TERM。"
      kill -TERM "$ROSLAUNCH_PID" 2>/dev/null || true
    fi
  fi

  [[ -n "$ODOM_PATH_PID" ]] && wait "$ODOM_PATH_PID" 2>/dev/null || true
  [[ -n "$ROSLAUNCH_PID" ]] && wait "$ROSLAUNCH_PID" 2>/dev/null || true
  exit "$rc"
}
trap cleanup INT TERM EXIT

[[ -f /opt/ros/noetic/setup.bash ]] || {
  error "未找到 ROS Noetic"; exit 1;
}
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
[[ -f "$WS/devel/setup.bash" ]] || {
  error "未找到 $WS/devel/setup.bash，请先编译工作空间"; exit 1;
}
# shellcheck disable=SC1090
source "$WS/devel/setup.bash"

PKG_PATH="$(rospack find "$PKG" 2>/dev/null || true)"
[[ -n "$PKG_PATH" ]] || { error "找不到 ROS 包 $PKG"; exit 1; }
[[ -f "$PKG_PATH/launch/$LAUNCH_FILE" ]] || {
  error "未找到 $PKG_PATH/launch/$LAUNCH_FILE"; exit 1;
}
[[ -x "$PKG_PATH/scripts/dwa_odom_adapter.py" ]] || \
  chmod +x "$PKG_PATH/scripts/dwa_odom_adapter.py"
[[ -x "$PKG_PATH/scripts/waypoint_executor.py" ]] || \
  chmod +x "$PKG_PATH/scripts/waypoint_executor.py"

# 纯实车默认使用 subject1_dwa.yaml；也允许通过环境变量或命令行覆盖。
REAL_DWA_CONFIG="${REAL_DWA_CONFIG:-$PKG_PATH/config/subject1_dwa.yaml}"
[[ -f "$REAL_DWA_CONFIG" ]] || {
  error "纯实车 DWA 配置不存在：$REAL_DWA_CONFIG"; exit 1;
}
REAL_DWA_CONFIG="$(readlink -f "$REAL_DWA_CONFIG")"

if [[ -n "$WAYPOINT_FILE" ]]; then
  [[ -f "$WAYPOINT_FILE" ]] || {
    error "航点文件不存在：$WAYPOINT_FILE"; exit 1;
  }
  WAYPOINT_FILE="$(readlink -f "$WAYPOINT_FILE")"
fi

if ! rosnode list >/dev/null 2>&1; then
  error "ROS master 未运行。请先启动：./start_1x.sh"
  exit 1
fi
rosnode list 2>/dev/null | grep -qx '/one_x_serial_driver' || {
  error "未发现 /one_x_serial_driver。请先启动：./start_1x.sh"
  exit 1
}
if rosnode list 2>/dev/null | grep -qx '/move_base'; then
  error "已有 /move_base 正在运行，请先停止旧导航。"
  exit 1
fi

info "等待独立1X原始解析数据……"
timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/ins_fix >/dev/null 2>&1 || {
  error "没有收到 /one_x/ins_fix"; exit 1;
}

info "等待1X原点设置服务……"
for _ in $(seq 1 $((READY_TIMEOUT * 5))); do
  rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' && break
  sleep 0.2
done
rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' || {
  error "未发现 /one_x/set_current_origin"; exit 1;
}

info "以当前最新1X位置建立本次导航ENU原点……"
ORIGIN_RESPONSE="$(rosservice call /one_x/set_current_origin "{}" 2>&1 || true)"
printf '%s\n' "$ORIGIN_RESPONSE"
grep -Eq 'success:[[:space:]]*(True|true)' <<<"$ORIGIN_RESPONSE" || {
  error "设置导航原点失败"; exit 1;
}

timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/origin >/dev/null 2>&1 || {
  error "设置原点后没有收到 /one_x/origin"; exit 1;
}
timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/odom >/dev/null 2>&1 || {
  error "设置原点后没有收到 /one_x/odom"; exit 1;
}

TF_CHECK="$(mktemp)"
timeout 4 rosrun tf tf_echo odom base_link >"$TF_CHECK" 2>&1 || true
if ! grep -q 'Translation' "$TF_CHECK"; then
  error "设置原点后仍缺少 TF：odom -> base_link"
  cat "$TF_CHECK" >&2 || true
  rm -f "$TF_CHECK"
  exit 1
fi
rm -f "$TF_CHECK"
ok "1X原点、里程计和TF已就绪"

can_is_up() {
  local flags
  flags="$(ip -o link show "$CAN_PORT" 2>/dev/null |
    sed -n 's/.*<\([^>]*\)>.*/\1/p')"
  [[ ",$flags," == *,UP,* ]]
}

if [[ "$LAUNCH_BASE" == "true" ]]; then
  ip link show "$CAN_PORT" >/dev/null 2>&1 || {
    error "未找到 CAN 接口：$CAN_PORT"; exit 1;
  }
  if [[ "$SETUP_CAN" == "true" ]]; then
    info "配置 $CAN_PORT，bitrate=$CAN_BITRATE"
    sudo ip link set "$CAN_PORT" down >/dev/null 2>&1 || true
    sudo ip link set "$CAN_PORT" type can bitrate "$CAN_BITRATE"
    sudo ip link set "$CAN_PORT" up
  fi
  can_is_up || { error "$CAN_PORT 尚未处于 UP 状态"; exit 1; }
  ok "CAN 接口可用：$CAN_PORT"
else
  warn "LAUNCH_BASE=false：本次不启动底盘。"
fi

ROSLAUNCH_ARGS=(
  "$PKG" "$LAUNCH_FILE"
  "can_port:=$CAN_PORT"
  "launch_base:=$LAUNCH_BASE"
  "launch_lidar:=false"
  "launch_rviz:=$LAUNCH_RVIZ"
  "auto_start:=false"
  "max_goal_distance_from_origin_m:=$MAX_GOAL_DIST"
  "dwa_odom_topic:=$DWA_ODOM_TOPIC"
  "dwa_max_yaw_rate_radps:=$DWA_MAX_YAW_RATE_RADPS"
  "dwa_config_file:=$REAL_DWA_CONFIG"
  "local_costmap_config_file:=$PKG_PATH/config/subject1_local_costmap.yaml"
)
[[ -n "$WAYPOINT_FILE" ]] && ROSLAUNCH_ARGS+=("waypoint_file:=$WAYPOINT_FILE")

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/r300_real_nav_$(date +%Y%m%d_%H%M%S).log"

info "启动纯实车导航，使用独立纯实车DWA参数和统一里程计适配器"
info "DWA配置：$REAL_DWA_CONFIG"
info "DWA反馈话题：$DWA_ODOM_TOPIC"
info "最大航点距离：$MAX_GOAL_DIST m"
info "日志文件：$LOG_FILE"

roslaunch "${ROSLAUNCH_ARGS[@]}" > >(tee "$LOG_FILE") 2>&1 &
ROSLAUNCH_PID=$!

if [[ "$ODOM_PATH" == "true" ]]; then
  rosrun "$PKG" odom_to_path.py \
    _odom_topic:=/one_x/odom \
    _path_topic:=/one_x/path \
    _max_points:=5000 >/dev/null 2>&1 &
  ODOM_PATH_PID=$!
fi

ensure_launch_alive() {
  kill -0 "$ROSLAUNCH_PID" 2>/dev/null || {
    error "主 launch 已退出，请查看：$LOG_FILE"; exit 1;
  }
}

wait_message() {
  local topic="$1"
  local description="$2"
  ensure_launch_alive
  info "等待：$description（$topic）"
  if timeout "$READY_TIMEOUT" rostopic echo -n 1 "$topic" >/dev/null 2>&1; then
    ok "$description已就绪"
  else
    error "$description在 ${READY_TIMEOUT}s 内没有数据：$topic"
    exit 1
  fi
}

wait_service() {
  local service="$1"
  local description="$2"
  info "等待：$description（$service）"
  for _ in $(seq 1 $((READY_TIMEOUT * 5))); do
    rosservice list 2>/dev/null | grep -qx "$service" && {
      ok "$description已就绪"; return 0;
    }
    ensure_launch_alive
    sleep 0.2
  done
  error "$description在 ${READY_TIMEOUT}s 内没有出现：$service"
  exit 1
}

check_topic_type() {
  local topic="$1"
  local expected="$2"
  local actual
  actual="$(rostopic type "$topic" 2>/dev/null || true)"
  [[ "$actual" == "$expected" ]] || {
    error "$topic 类型错误，期望 $expected，实际 ${actual:-<未知>}"; exit 1;
  }
}

wait_message "$DWA_ODOM_TOPIC" "DWA适配后里程计"
check_topic_type "$DWA_ODOM_TOPIC" nav_msgs/Odometry
wait_message /move_base/local_costmap/costmap "局部代价地图"
wait_service /move_base/DWAPlannerROS/set_parameters "DWA动态调参服务"

for service in \
  /subject1/start_waypoints \
  /subject1/pause_waypoints \
  /subject1/resume_waypoints \
  /subject1/skip_waypoint \
  /subject1/cancel_waypoints
do
  wait_service "$service" "航点服务"
done

wait_message /subject1/waypoint_status "航点状态与进度"
check_topic_type /subject1/waypoint_status std_msgs/String

LOADED_DWA_ODOM="$(rosparam get /move_base/DWAPlannerROS/odom_topic 2>/dev/null || true)"
ADAPTER_OUTPUT="$(rosparam get /dwa_odom_adapter/output_odom_topic 2>/dev/null || true)"
LOADED_MAX_DIST="$(rosparam get /waypoint_executor/max_goal_distance_from_origin_m 2>/dev/null || true)"

[[ "$LOADED_DWA_ODOM" == "$DWA_ODOM_TOPIC" ]] || {
  error "DWA odom_topic错误：${LOADED_DWA_ODOM:-<空>}，期望 $DWA_ODOM_TOPIC"; exit 1;
}
[[ "$ADAPTER_OUTPUT" == "$DWA_ODOM_TOPIC" ]] || {
  error "dwa_odom_adapter输出错误：${ADAPTER_OUTPUT:-<空>}，期望 $DWA_ODOM_TOPIC"; exit 1;
}

ok "纯导航与视觉导航已统一使用 $DWA_ODOM_TOPIC"
info "实际加载最大航点距离：${LOADED_MAX_DIST:-unknown} m"
info "当前航点状态："
rostopic echo -n 1 /subject1/waypoint_status 2>/dev/null || true

if [[ "$AUTO_RUN" == "true" ]]; then
  [[ "$LAUNCH_BASE" == "true" ]] || {
    error "--run 不能与 --no-base 同时使用"; exit 1;
  }
  warn "全链路已就绪，即将启动航点任务。"
  rosservice call /subject1/start_waypoints "{}"
else
  ok "纯实车导航已就绪，但车辆尚未自动执行航点。"
  echo "启动：rosservice call /subject1/start_waypoints \"{}\""
fi

echo
echo "暂停：rosservice call /subject1/pause_waypoints \"{}\""
echo "恢复：rosservice call /subject1/resume_waypoints \"{}\""
echo "跳过：rosservice call /subject1/skip_waypoint \"{}\""
echo "取消：rosservice call /subject1/cancel_waypoints \"{}\""
echo "状态：watch -n 1 'rostopic echo -n 1 /subject1/waypoint_status'"
echo "停止导航：当前终端按 Ctrl+C（独立1X继续运行）"
echo

wait "$ROSLAUNCH_PID"
