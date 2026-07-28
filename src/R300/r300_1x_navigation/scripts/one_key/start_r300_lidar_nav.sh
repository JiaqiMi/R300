#!/usr/bin/env bash
# =============================================================================
# R300：订阅已运行的 1X + 航点 + move_base/DWA + 雷达高程图避障
#
# 当前雷达 costmap 链路：
#   /elevation_mapping/elevation_map_raw
#     -> lidar_obstacle_scan_node.py
#     -> /r300_lidar/obstacle_scan
#     -> VisionSnapshotLayer（复用通用快照层，按障碍 TTL 保持）
#     -> inflation_layer -> DWA
#
# 本脚本不启动 1X，也不启动 MID-360 / FAST-LIO / 高程图。
# 请先分别启动：
#   1) ./start_1x.sh
#   2) single_lidar_elevation 感知栈
# =============================================================================

set -Eeuo pipefail

# ------------------------------ 默认参数 -------------------------------------
WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
LAUNCH_FILE="${R300_LIDAR_LAUNCH:-subject1_waypoint_lidar_nav.launch}"

CAN_PORT="${CAN_PORT:-can0}"
CAN_BITRATE="${CAN_BITRATE:-500000}"

WAYPOINT_FILE="${WAYPOINT_FILE:-}"
MAX_GOAL_DIST="${MAX_GOAL_DIST:-5000.0}"

ELEVATION_TOPIC="${ELEVATION_TOPIC:-/elevation_mapping/elevation_map_raw}"
OBSTACLE_SCAN_TOPIC="${OBSTACLE_SCAN_TOPIC:-/r300_lidar/obstacle_scan}"
ACTIVE_SCAN_TOPIC="/r300_lidar/active_obstacle_scan"
DEBUG_CLOUD_TOPIC="${DEBUG_CLOUD_TOPIC:-/r300_lidar/obstacle_points}"
LIO_MAP_FRAME="${LIO_MAP_FRAME:-odom}"
LIO_BODY_FRAME="${LIO_BODY_FRAME:-body}"

LAUNCH_BASE="${LAUNCH_BASE:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
ODOM_PATH="${ODOM_PATH:-true}"
SETUP_CAN="${SETUP_CAN:-true}"
AUTO_RUN="${AUTO_RUN:-false}"

READY_TIMEOUT="${READY_TIMEOUT:-60}"
LIDAR_HOLD_TIME_S="${LIDAR_HOLD_TIME_S:-}"
LOG_DIR="${LOG_DIR:-$WS/log/lidar_nav}"

# Web 包装层预先完成 sudo -v 后会设置该变量；普通终端运行时保持 false，
# 由 sudo 自己正常询问密码。
NONINTERACTIVE_SUDO="${R300_NONINTERACTIVE_SUDO:-false}"

ROSLAUNCH_PID=""
ODOM_PATH_PID=""
STOP_REQUESTED="false"

# ------------------------------ 输出函数 -------------------------------------
info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }

usage() {
  cat <<'USAGE'
用法：
  ./start_r300_lidar_nav.sh [选项]

选项：
  --run                    全链路就绪后立即启动航点
  --no-rviz                不启动 RViz
  --no-base                不启动底盘，适合台架检查
  --no-path                不启动 /one_x/path 辅助节点
  --setup-can              启动前重新配置 CAN（默认）
  --no-setup-can           不配置 CAN，仅检查接口是否 UP
  --waypoints PATH         指定航点 YAML
  --elevation-topic NAME   指定高程图话题
  --scan-topic NAME        指定雷达虚拟 LaserScan 话题
  --lio-map-frame NAME     FAST-LIO 地图坐标系，默认 odom
  --lio-body-frame NAME    FAST-LIO 车体坐标系，默认 body
  --timeout SEC            单项检查超时，默认 60 秒
  -h, --help               显示帮助

环境变量：
  R300_WS, CAN_PORT, CAN_BITRATE, WAYPOINT_FILE, MAX_GOAL_DIST,
  ELEVATION_TOPIC, OBSTACLE_SCAN_TOPIC,
  DEBUG_CLOUD_TOPIC, LIO_MAP_FRAME, LIO_BODY_FRAME,
  READY_TIMEOUT, LIDAR_HOLD_TIME_S,
  LAUNCH_RVIZ, LAUNCH_BASE, ODOM_PATH, SETUP_CAN, AUTO_RUN
USAGE
}

# ------------------------------ 参数解析 -------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      AUTO_RUN="true"; shift ;;
    --no-rviz)
      LAUNCH_RVIZ="false"; shift ;;
    --no-base)
      LAUNCH_BASE="false"; shift ;;
    --no-path)
      ODOM_PATH="false"; shift ;;
    --setup-can)
      SETUP_CAN="true"; shift ;;
    --no-setup-can)
      SETUP_CAN="false"; shift ;;
    --waypoints)
      [[ $# -ge 2 ]] || { error "--waypoints 后缺少路径"; exit 2; }
      WAYPOINT_FILE="$2"; shift 2 ;;
    --elevation-topic)
      [[ $# -ge 2 ]] || { error "--elevation-topic 后缺少话题名"; exit 2; }
      ELEVATION_TOPIC="$2"; shift 2 ;;
    --scan-topic)
      [[ $# -ge 2 ]] || { error "--scan-topic 后缺少话题名"; exit 2; }
      OBSTACLE_SCAN_TOPIC="$2"; shift 2 ;;
    --lio-map-frame)
      [[ $# -ge 2 ]] || { error "--lio-map-frame 后缺少坐标系名称"; exit 2; }
      LIO_MAP_FRAME="$2"; shift 2 ;;
    --lio-body-frame)
      [[ $# -ge 2 ]] || { error "--lio-body-frame 后缺少坐标系名称"; exit 2; }
      LIO_BODY_FRAME="$2"; shift 2 ;;
    --timeout)
      [[ $# -ge 2 ]] || { error "--timeout 后缺少秒数"; exit 2; }
      READY_TIMEOUT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      error "未知参数：$1"; usage; exit 2 ;;
  esac
done

for variable_name in \
  ELEVATION_TOPIC OBSTACLE_SCAN_TOPIC DEBUG_CLOUD_TOPIC
do
  value="${!variable_name}"
  [[ "$value" == /* ]] || printf -v "$variable_name" '/%s' "$value"
done

if ! [[ "$READY_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  error "READY_TIMEOUT 必须是正数：$READY_TIMEOUT"
  exit 2
fi

# ------------------------------ 清理/急停 ------------------------------------
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
    info "正在停止雷达避障导航系统……"
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

# ------------------------------ ROS 环境 -------------------------------------
[[ -f /opt/ros/noetic/setup.bash ]] || {
  error "未找到 /opt/ros/noetic/setup.bash"; exit 1;
}
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

[[ -f "$WS/devel/setup.bash" ]] || {
  error "未找到 $WS/devel/setup.bash，请先 catkin_make"; exit 1;
}
# shellcheck disable=SC1090
source "$WS/devel/setup.bash"

if ! rospack find "$PKG" >/dev/null 2>&1; then
  error "ROS 找不到功能包 $PKG"; exit 1
fi

PKG_PATH="$(rospack find "$PKG")"
[[ -f "$PKG_PATH/launch/$LAUNCH_FILE" ]] || {
  error "未找到 $PKG_PATH/launch/$LAUNCH_FILE"; exit 1;
}

LIDAR_ADAPTER_NODE="$PKG_PATH/scripts/lidar_obstacle_scan_node.py"
[[ -f "$LIDAR_ADAPTER_NODE" ]] || {
  error "未找到节点：$LIDAR_ADAPTER_NODE"; exit 1;
}
if [[ ! -x "$LIDAR_ADAPTER_NODE" ]]; then
  warn "正在添加执行权限：$LIDAR_ADAPTER_NODE"
  chmod +x "$LIDAR_ADAPTER_NODE"
fi

REQUIRED_PACKAGES=(
  move_base map_server navfn dwa_local_planner costmap_2d
  grid_map_msgs scout_base robot_state_publisher r300_simulation
)
for dependency in "${REQUIRED_PACKAGES[@]}"; do
  rospack find "$dependency" >/dev/null 2>&1 || {
    error "缺少 ROS 功能包：$dependency"; exit 1;
  }
done
python3 - <<'PY' >/dev/null 2>&1 || {
import numpy
PY
  error "Python3 缺少 numpy；请安装 python3-numpy"
  exit 1
}
ok "ROS 环境和雷达导航依赖检查通过"

# ------------------------------ 文件与外部链路 -------------------------------
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
  error "未发现 /one_x_serial_driver。请先在另一终端运行：./start_1x.sh"
  exit 1
}

if rosnode list 2>/dev/null | grep -qx '/move_base'; then
  error "检测到已有 /move_base。请先停止纯实车、视觉或旧雷达导航。"
  exit 1
fi

info "等待独立 1X 原始解析数据……"
if ! timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/ins_fix >/dev/null 2>&1; then
  error "独立 1X 没有发布 /one_x/ins_fix，请检查 start_1x.sh 和串口。"
  exit 1
fi
ok "独立 1X 解析数据正常"

info "等待外部雷达高程图：$ELEVATION_TOPIC"
if ! timeout "$READY_TIMEOUT" rostopic echo -n 1 "$ELEVATION_TOPIC" >/dev/null 2>&1; then
  error "没有收到雷达高程图：$ELEVATION_TOPIC"
  error "请先启动 single_lidar_elevation（MID-360 + FAST-LIO + 高程图）。"
  exit 1
fi
ELEVATION_TYPE="$(rostopic type "$ELEVATION_TOPIC" 2>/dev/null || true)"
if [[ "$ELEVATION_TYPE" != "grid_map_msgs/GridMap" ]]; then
  error "高程图消息类型不正确：${ELEVATION_TYPE:-<未知>}"
  error "期望：grid_map_msgs/GridMap"
  exit 1
fi
ok "外部高程图数据正常：$ELEVATION_TYPE"

info "检查 FAST-LIO TF：$LIO_MAP_FRAME -> $LIO_BODY_FRAME"
LIO_TF_CHECK="$(mktemp)"
timeout 5 rosrun tf tf_echo "$LIO_MAP_FRAME" "$LIO_BODY_FRAME" \
  >"$LIO_TF_CHECK" 2>&1 || true
if ! grep -q 'Translation' "$LIO_TF_CHECK"; then
  error "缺少 FAST-LIO TF：$LIO_MAP_FRAME -> $LIO_BODY_FRAME"
  cat "$LIO_TF_CHECK" >&2 || true
  rm -f "$LIO_TF_CHECK"
  exit 1
fi
rm -f "$LIO_TF_CHECK"
ok "FAST-LIO TF 可用"

# ------------------------------ 建立 1X 导航原点 ----------------------------
info "等待 1X 原点设置服务……"
for _ in $(seq 1 "$(awk -v t="$READY_TIMEOUT" 'BEGIN {print int(t * 5)}')"); do
  rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' && break
  sleep 0.2
done
rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' || {
  error "未发现 /one_x/set_current_origin；请确认已替换并重新编译 1X 驱动。"
  exit 1
}

info "以当前最新 1X 位置建立本次导航 ENU 原点……"
ORIGIN_RESPONSE="$(rosservice call /one_x/set_current_origin "{}" 2>&1 || true)"
printf '%s\n' "$ORIGIN_RESPONSE"
if ! grep -Eq 'success:[[:space:]]*(True|true)' <<<"$ORIGIN_RESPONSE"; then
  error "1X 导航原点设置失败。"
  exit 1
fi

if ! timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/origin >/dev/null 2>&1; then
  error "设置原点后未收到 /one_x/origin"
  exit 1
fi
if ! timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/odom >/dev/null 2>&1; then
  error "设置原点后未收到 /one_x/odom"
  exit 1
fi

TF_CHECK="$(mktemp)"
timeout 4 rosrun tf tf_echo odom base_link >"$TF_CHECK" 2>&1 || true
if ! grep -q 'Translation' "$TF_CHECK"; then
  error "设置原点后仍缺少 TF：odom -> base_link"
  cat "$TF_CHECK" >&2 || true
  rm -f "$TF_CHECK"
  exit 1
fi
rm -f "$TF_CHECK"
ok "本次导航坐标系已由当前 1X 结果建立"

# ------------------------------ CAN ------------------------------------------
can_is_up() {
  local flags
  flags="$(ip -o link show "$CAN_PORT" 2>/dev/null |
    sed -n 's/.*<\([^>]*\)>.*/\1/p')"
  [[ ",$flags," == *,UP,* ]]
}

run_privileged() {
  if [[ "$EUID" -eq 0 ]]; then
    "$@"
  elif [[ "$NONINTERACTIVE_SUDO" == "true" ]]; then
    sudo -n "$@"
  else
    sudo "$@"
  fi
}

if [[ "$LAUNCH_BASE" == "true" ]]; then
  ip link show "$CAN_PORT" >/dev/null 2>&1 || {
    error "未找到 CAN 接口：$CAN_PORT"; exit 1;
  }

  if [[ "$SETUP_CAN" == "true" ]]; then
    info "配置 $CAN_PORT，bitrate=$CAN_BITRATE"
    run_privileged ip link set "$CAN_PORT" down >/dev/null 2>&1 || true
    run_privileged ip link set "$CAN_PORT" type can bitrate "$CAN_BITRATE"
    run_privileged ip link set "$CAN_PORT" up
  fi

  can_is_up || {
    error "$CAN_PORT 尚未处于 UP 状态"; exit 1;
  }
  ok "CAN 接口可用：$CAN_PORT"
else
  warn "LAUNCH_BASE=false：本次不启动底盘。"
fi

# ------------------------------ 启动参数 -------------------------------------
ROSLAUNCH_ARGS=(
  "$PKG" "$LAUNCH_FILE"
  "can_port:=$CAN_PORT"
  "launch_base:=$LAUNCH_BASE"
  "launch_rviz:=$LAUNCH_RVIZ"
  "elevation_topic:=$ELEVATION_TOPIC"
  "obstacle_scan_topic:=$OBSTACLE_SCAN_TOPIC"
  "debug_cloud_topic:=$DEBUG_CLOUD_TOPIC"
  "lio_map_frame:=$LIO_MAP_FRAME"
  "lio_body_frame:=$LIO_BODY_FRAME"
  "auto_start:=false"
  "max_goal_distance_from_origin_m:=$MAX_GOAL_DIST"
)
[[ -n "$WAYPOINT_FILE" ]] && ROSLAUNCH_ARGS+=("waypoint_file:=$WAYPOINT_FILE")

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/r300_lidar_nav_$(date +%Y%m%d_%H%M%S).log"

info "工作空间：$WS"
info "启动入口：$PKG/$LAUNCH_FILE"
info "定位来源：外部已运行 /one_x_serial_driver"
info "外部高程图：$ELEVATION_TOPIC"
info "雷达障碍扫描：$OBSTACLE_SCAN_TOPIC"
info "雷达层调试扫描：$ACTIVE_SCAN_TOPIC"
info "FAST-LIO TF：$LIO_MAP_FRAME -> $LIO_BODY_FRAME"
if [[ -n "$LIDAR_HOLD_TIME_S" ]]; then
  info "雷达快照层期望保持时间：${LIDAR_HOLD_TIME_S}s"
else
  info "雷达快照层保持时间：自动读取 YAML 实际加载值"
fi
info "日志文件：$LOG_FILE"

roslaunch "${ROSLAUNCH_ARGS[@]}" > >(tee "$LOG_FILE") 2>&1 &
ROSLAUNCH_PID=$!

for _ in $(seq 1 50); do
  rosnode list >/dev/null 2>&1 && break
  kill -0 "$ROSLAUNCH_PID" 2>/dev/null || {
    error "roslaunch 已提前退出，请查看：$LOG_FILE"; exit 1;
  }
  sleep 0.2
done
rosnode list >/dev/null 2>&1 || {
  error "ROS Master 未在预期时间内启动"; exit 1;
}

if [[ "$ODOM_PATH" == "true" ]]; then
  rosrun "$PKG" odom_to_path.py \
    _odom_topic:=/one_x/odom \
    _path_topic:=/one_x/path \
    _max_points:=5000 \
    >/dev/null 2>&1 &
  ODOM_PATH_PID=$!
fi

# ------------------------------ 就绪函数 -------------------------------------
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
    ensure_launch_alive
    error "$description在 ${READY_TIMEOUT}s 内没有数据：$topic"
    exit 1
  fi
}

check_topic_type() {
  local topic="$1"
  local expected="$2"
  local description="$3"
  local actual

  actual="$(rostopic type "$topic" 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    error "$description消息类型不匹配：$topic"
    error "期望：$expected"
    error "实际：${actual:-<未知>}"
    exit 1
  fi
  ok "$description消息类型正确：$actual"
}

wait_service() {
  local service="$1"
  local description="$2"
  local loops
  loops="$(awk -v t="$READY_TIMEOUT" 'BEGIN {print int(t * 5)}')"

  info "等待：$description（$service）"
  for _ in $(seq 1 "$loops"); do
    rosservice list 2>/dev/null | grep -qx "$service" && {
      ok "$description已就绪"; return 0;
    }
    ensure_launch_alive
    sleep 0.2
  done
  error "$description在 ${READY_TIMEOUT}s 内没有出现：$service"
  exit 1
}

check_lio_tf() {
  local tf_log
  ensure_launch_alive
  tf_log="$(mktemp)"
  timeout 4 rosrun tf tf_echo "$LIO_MAP_FRAME" "$LIO_BODY_FRAME" \
    >"$tf_log" 2>&1 || true
  ensure_launch_alive
  if grep -q "Translation" "$tf_log"; then
    ok "FAST-LIO TF 可用：$LIO_MAP_FRAME -> $LIO_BODY_FRAME"
  else
    error "FAST-LIO TF 丢失：$LIO_MAP_FRAME -> $LIO_BODY_FRAME"
    cat "$tf_log" >&2 || true
    rm -f "$tf_log"
    exit 1
  fi
  rm -f "$tf_log"
}

# ------------------------------ 全链路检查 -----------------------------------
wait_message /one_x/odom "1X 惯导里程计"
wait_message /subject1/dwa_odom "DWA 适配后里程计"
check_topic_type /subject1/dwa_odom nav_msgs/Odometry "DWA 适配后里程计"

wait_message "$ELEVATION_TOPIC" "外部雷达高程图"
check_topic_type "$ELEVATION_TOPIC" grid_map_msgs/GridMap "雷达高程图"
check_lio_tf

wait_message "$OBSTACLE_SCAN_TOPIC" "雷达高程图虚拟 LaserScan"
check_topic_type "$OBSTACLE_SCAN_TOPIC" sensor_msgs/LaserScan "雷达虚拟 LaserScan"

wait_message /move_base/local_costmap/costmap "雷达局部代价地图"
wait_service /subject1/start_waypoints "航点启动服务"

# ------------------------------ 参数一致性 -----------------------------------
ADAPTER_ELEVATION="$(
  rosparam get /lidar_obstacle_scan_node/elevation_topic 2>/dev/null || true
)"
ADAPTER_SCAN="$(
  rosparam get /lidar_obstacle_scan_node/scan_topic 2>/dev/null || true
)"
ADAPTER_LIO_MAP="$(
  rosparam get /lidar_obstacle_scan_node/lio_map_frame 2>/dev/null || true
)"
ADAPTER_LIO_BODY="$(
  rosparam get /lidar_obstacle_scan_node/lio_body_frame 2>/dev/null || true
)"

if [[ "$ADAPTER_ELEVATION" != "$ELEVATION_TOPIC" ||
      "$ADAPTER_SCAN" != "$OBSTACLE_SCAN_TOPIC" ||
      "$ADAPTER_LIO_MAP" != "$LIO_MAP_FRAME" ||
      "$ADAPTER_LIO_BODY" != "$LIO_BODY_FRAME" ]]; then
  error "雷达障碍适配节点参数不一致"
  error "elevation=${ADAPTER_ELEVATION:-<空>}，期望 $ELEVATION_TOPIC"
  error "scan=${ADAPTER_SCAN:-<空>}，期望 $OBSTACLE_SCAN_TOPIC"
  error "lio_map=${ADAPTER_LIO_MAP:-<空>}，期望 $LIO_MAP_FRAME"
  error "lio_body=${ADAPTER_LIO_BODY:-<空>}，期望 $LIO_BODY_FRAME"
  exit 1
fi
ok "雷达障碍适配节点订阅与坐标系参数正确"

LIDAR_LAYER_NS="/move_base/local_costmap/lidar_snapshot_layer"
SNAPSHOT_TOPIC="$(rosparam get "${LIDAR_LAYER_NS}/topic" 2>/dev/null || true)"
SNAPSHOT_HOLD="$(rosparam get "${LIDAR_LAYER_NS}/hold_time_s" 2>/dev/null || true)"
SNAPSHOT_ACTIVE_SCAN="$(
  rosparam get "${LIDAR_LAYER_NS}/active_scan_topic" 2>/dev/null || true
)"
PLUGINS="$(rosparam get /move_base/local_costmap/plugins 2>/dev/null || true)"

if [[ "$PLUGINS" != *"lidar_snapshot_layer"* ]]; then
  error "move_base 未加载 lidar_snapshot_layer 插件"
  error "plugins=${PLUGINS:-<空>}"
  exit 1
fi
if [[ "$SNAPSHOT_TOPIC" != "$OBSTACLE_SCAN_TOPIC" ]]; then
  error "雷达快照层订阅话题错误：${SNAPSHOT_TOPIC:-<空>}"
  error "期望：$OBSTACLE_SCAN_TOPIC"
  exit 1
fi
if ! awk -v actual="$SNAPSHOT_HOLD" '
  BEGIN { exit !(actual ~ /^[0-9]+([.][0-9]+)?$/ && actual > 0.0) }
'; then
  error "雷达快照层 hold_time_s 未正确加载：${SNAPSHOT_HOLD:-<空>}"
  exit 1
fi
if [[ -n "$LIDAR_HOLD_TIME_S" ]]; then
  if ! awk -v actual="$SNAPSHOT_HOLD" -v expected="$LIDAR_HOLD_TIME_S" '
    BEGIN {
      if (!(expected ~ /^[0-9]+([.][0-9]+)?$/) || expected <= 0.0) exit 2
      diff = actual - expected; if (diff < 0) diff = -diff
      exit !(diff < 0.000001)
    }
  '; then
    error "雷达快照层 hold_time_s 参数不一致：$SNAPSHOT_HOLD"
    error "显式期望值：${LIDAR_HOLD_TIME_S}s"
    exit 1
  fi
fi
ok "move_base 已加载雷达快照层，障碍保持 ${SNAPSHOT_HOLD}s"

if ! rostopic info "$OBSTACLE_SCAN_TOPIC" 2>/dev/null | grep -q '/move_base'; then
  error "/move_base 未直接订阅 $OBSTACLE_SCAN_TOPIC"
  exit 1
fi
ok "/move_base 正在直接订阅：$OBSTACLE_SCAN_TOPIC"

if [[ -n "$SNAPSHOT_ACTIVE_SCAN" ]]; then
  if [[ "$SNAPSHOT_ACTIVE_SCAN" != "$ACTIVE_SCAN_TOPIC" ]]; then
    error "雷达活动障碍扫描话题不一致：$SNAPSHOT_ACTIVE_SCAN"
    error "期望：$ACTIVE_SCAN_TOPIC"
    exit 1
  fi
  wait_message "$SNAPSHOT_ACTIVE_SCAN" "雷达层活动障碍调试扫描"
  check_topic_type "$SNAPSHOT_ACTIVE_SCAN" sensor_msgs/LaserScan \
    "雷达层活动障碍调试扫描"
fi

DWA_MAX_VEL="$(
  rosparam get /move_base/DWAPlannerROS/max_vel_x 2>/dev/null || echo unknown
)"
CONTROLLER_FREQUENCY="$(
  rosparam get /move_base/controller_frequency 2>/dev/null || echo unknown
)"
INFLATION_RADIUS="$(
  rosparam get /move_base/local_costmap/inflation_layer/inflation_radius \
    2>/dev/null || echo unknown
)"
DWA_ODOM_TOPIC_LOADED="$(
  rosparam get /move_base/DWAPlannerROS/odom_topic 2>/dev/null || true
)"
ADAPTER_OUTPUT_TOPIC="$(
  rosparam get /dwa_odom_adapter/output_odom_topic 2>/dev/null || true
)"
WAYPOINT_MAX_DIST="$(
  rosparam get /waypoint_executor/max_goal_distance_from_origin_m 2>/dev/null || true
)"

if [[ "$DWA_ODOM_TOPIC_LOADED" != "/subject1/dwa_odom" ||
      "$ADAPTER_OUTPUT_TOPIC" != "/subject1/dwa_odom" ]]; then
  error "DWA 里程计链路不一致"
  error "DWA odom_topic=${DWA_ODOM_TOPIC_LOADED:-<空>}"
  error "adapter output=${ADAPTER_OUTPUT_TOPIC:-<空>}"
  exit 1
fi
ok "雷达导航使用统一 DWA 里程计：/subject1/dwa_odom"

info "DWA 最大前进速度：$DWA_MAX_VEL m/s"
info "move_base 控制频率：$CONTROLLER_FREQUENCY Hz"
info "局部地图膨胀半径：$INFLATION_RADIUS m"
info "最大航点距离：${WAYPOINT_MAX_DIST:-unknown} m"
info "当前航点状态："
rostopic echo -n 1 /subject1/waypoint_status 2>/dev/null || true

# ------------------------------ 是否开始运动 ---------------------------------
if [[ "$AUTO_RUN" == "true" ]]; then
  [[ "$LAUNCH_BASE" == "true" ]] || {
    error "--run 不能与 --no-base 同时使用"; exit 1;
  }
  warn "全链路已就绪，即将启动航点任务。"
  rosservice call /subject1/start_waypoints "{}"
else
  ok "整套雷达避障导航已就绪，但车辆尚未自动执行航点。"
  echo
  echo "启动航点：rosservice call /subject1/start_waypoints \"{}\""
fi

echo
echo "暂停：rosservice call /subject1/pause_waypoints \"{}\""
echo "恢复：rosservice call /subject1/resume_waypoints \"{}\""
echo "取消：rosservice call /subject1/cancel_waypoints \"{}\""
echo "停止导航：在本终端按 Ctrl+C"
echo "说明：停止本脚本不会停止独立 1X 和雷达高程感知栈。"
echo

wait "$ROSLAUNCH_PID"
