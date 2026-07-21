#!/usr/bin/env bash
# =============================================================================
# R300：惯导 + 航点 + move_base/DWA + 外部视觉检测话题避障
#
# 当前视觉 costmap 链路：
#   /r300_vision/detections
#     -> /r300_vision/obstacle_scan
#     -> VisionSnapshotLayer（odom中按配置保持障碍，每周期整层重建）
#     -> inflation_layer -> DWA
#
# 本脚本不启动相机、不启动检测网络。
# =============================================================================

set -Eeuo pipefail

# ------------------------------ 默认参数 -------------------------------------
WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
LAUNCH_FILE="${R300_VISION_LAUNCH:-subject1_waypoint_vision_nav.launch}"

INS_PORT="${INS_PORT:-/dev/ttyACM0}"
INS_BAUD="${INS_BAUD:-460800}"
CAN_PORT="${CAN_PORT:-can0}"
CAN_BITRATE="${CAN_BITRATE:-500000}"

WAYPOINT_FILE="${WAYPOINT_FILE:-}"
MAX_GOAL_DIST="${MAX_GOAL_DIST:-5000.0}"

DETECTIONS_TOPIC="${DETECTIONS_TOPIC:-/r300_vision/detections}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
CAMERA_FRAME="${CAMERA_FRAME:-}"

OBSTACLE_SCAN_TOPIC="${OBSTACLE_SCAN_TOPIC:-/r300_vision/obstacle_scan}"
ACTIVE_SCAN_TOPIC="${ACTIVE_SCAN_TOPIC:-/r300_vision/active_obstacle_scan}"

LAUNCH_BASE="${LAUNCH_BASE:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
ODOM_PATH="${ODOM_PATH:-true}"
SETUP_CAN="${SETUP_CAN:-true}"
AUTO_RUN="${AUTO_RUN:-false}"

READY_TIMEOUT="${READY_TIMEOUT:-60}"
# 一键脚本期望 VisionSnapshotLayer 加载的保持时间，默认 5 秒。
# 仅用于启动自检；真正保持时间仍由 local costmap YAML 中 hold_time_s 决定。
VISION_HOLD_TIME_S="${VISION_HOLD_TIME_S:-}"
LOG_DIR="${LOG_DIR:-$WS/log/vision_nav}"

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
  ./start_r300_vision_nav.sh [选项]

选项：
  --run                    全链路就绪后启动航点
  --no-rviz                不启动 RViz
  --no-base                不启动底盘，适合台架检查
  --no-path                不启动 /one_x/path 辅助节点
  --setup-can              启动前重新配置 CAN
  --waypoints PATH         指定航点 YAML
  --detections-topic NAME  指定检测结果话题
  --camera-info-topic NAME 指定相机内参话题
  --camera-frame NAME      手工指定检测坐标系
  --timeout SEC            单项检查超时，默认 60 秒
  -h, --help               显示帮助

环境变量：
  R300_WS, INS_PORT, INS_BAUD, CAN_PORT, CAN_BITRATE,
  WAYPOINT_FILE, MAX_GOAL_DIST, DETECTIONS_TOPIC,
  CAMERA_INFO_TOPIC, CAMERA_FRAME, OBSTACLE_SCAN_TOPIC,
  ACTIVE_SCAN_TOPIC, READY_TIMEOUT, VISION_HOLD_TIME_S,
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
    --waypoints)
      [[ $# -ge 2 ]] || { error "--waypoints 后缺少路径"; exit 2; }
      WAYPOINT_FILE="$2"; shift 2 ;;
    --detections-topic)
      [[ $# -ge 2 ]] || { error "--detections-topic 后缺少话题名"; exit 2; }
      DETECTIONS_TOPIC="$2"; shift 2 ;;
    --camera-info-topic)
      [[ $# -ge 2 ]] || { error "--camera-info-topic 后缺少话题名"; exit 2; }
      CAMERA_INFO_TOPIC="$2"; shift 2 ;;
    --camera-frame)
      [[ $# -ge 2 ]] || { error "--camera-frame 后缺少坐标系名称"; exit 2; }
      CAMERA_FRAME="$2"; shift 2 ;;
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
  DETECTIONS_TOPIC CAMERA_INFO_TOPIC OBSTACLE_SCAN_TOPIC ACTIVE_SCAN_TOPIC
do
  value="${!variable_name}"
  [[ "$value" == /* ]] || printf -v "$variable_name" '/%s' "$value"
done

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
    info "正在停止视觉导航系统……"
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

VISION_ADAPTER_NODE="$PKG_PATH/scripts/vision_obstacle_layer_node.py"
for node_file in "$VISION_ADAPTER_NODE"; do
  [[ -f "$node_file" ]] || { error "未找到节点：$node_file"; exit 1; }
  if [[ ! -x "$node_file" ]]; then
    warn "正在添加执行权限：$node_file"
    chmod +x "$node_file"
  fi
done

REQUIRED_PACKAGES=(
  move_base map_server navfn dwa_local_planner costmap_2d
  r300_vision_msgs scout_base robot_state_publisher r300_simulation
)
for dependency in "${REQUIRED_PACKAGES[@]}"; do
  rospack find "$dependency" >/dev/null 2>&1 || {
    error "缺少 ROS 功能包：$dependency"; exit 1;
  }
done
ok "ROS 环境和导航依赖检查通过"

# ------------------------------ 文件与硬件 -----------------------------------
if [[ -n "$WAYPOINT_FILE" ]]; then
  [[ -f "$WAYPOINT_FILE" ]] || {
    error "航点文件不存在：$WAYPOINT_FILE"; exit 1;
  }
  WAYPOINT_FILE="$(readlink -f "$WAYPOINT_FILE")"
fi

[[ -e "$INS_PORT" ]] || {
  error "惯导串口不存在：$INS_PORT"; exit 1;
}
[[ -r "$INS_PORT" && -w "$INS_PORT" ]] || {
  error "当前用户没有 $INS_PORT 的读写权限"; exit 1;
}
ok "惯导串口可用：$INS_PORT，baud=$INS_BAUD"

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
  "ins_serial_port:=$INS_PORT"
  "ins_baudrate:=$INS_BAUD"
  "can_port:=$CAN_PORT"
  "launch_base:=$LAUNCH_BASE"
  "launch_rviz:=$LAUNCH_RVIZ"
  "detections_topic:=$DETECTIONS_TOPIC"
  "camera_info_topic:=$CAMERA_INFO_TOPIC"
  "auto_start:=false"
  "max_goal_distance_from_origin_m:=$MAX_GOAL_DIST"
)
[[ -n "$WAYPOINT_FILE" ]] &&
  ROSLAUNCH_ARGS+=("waypoint_file:=$WAYPOINT_FILE")

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/r300_vision_nav_$(date +%Y%m%d_%H%M%S).log"

info "工作空间：$WS"
info "启动入口：$PKG/$LAUNCH_FILE"
info "外部检测话题：$DETECTIONS_TOPIC"
info "外部相机内参：$CAMERA_INFO_TOPIC"
info "视觉障碍扫描：$OBSTACLE_SCAN_TOPIC"
info "视觉层调试扫描：$ACTIVE_SCAN_TOPIC"
if [[ -n "$VISION_HOLD_TIME_S" ]]; then
  info "VisionSnapshotLayer 期望保持时间：${VISION_HOLD_TIME_S}s"
else
  info "VisionSnapshotLayer 保持时间：自动读取 YAML 实际加载值"
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

  if timeout "$READY_TIMEOUT" \
      rostopic echo -n 1 "$topic" >/dev/null 2>&1; then
    ok "$description已就绪"
  else
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

read_detection_frame() {
  timeout "$READY_TIMEOUT" \
    rostopic echo -n 1 "$DETECTIONS_TOPIC" 2>/dev/null |
    awk -F': ' \
      '/^[[:space:]]*frame_id:/{gsub(/["\047]/, "", $2); print $2; exit}'
}

check_camera_tf() {
  local frame="$CAMERA_FRAME"
  local tf_log

  if [[ -z "$frame" ]]; then
    frame="$(read_detection_frame || true)"
  fi

  [[ -n "$frame" ]] || {
    error "无法从检测消息读取 frame_id"; exit 1;
  }

  tf_log="$(mktemp)"
  timeout 4 rosrun tf tf_echo base_link "$frame" \
    >"$tf_log" 2>&1 || true

  if grep -q "Translation" "$tf_log"; then
    ok "TF 可用：base_link -> $frame"
    CAMERA_FRAME="$frame"
  else
    error "缺少 TF：base_link -> $frame"
    cat "$tf_log" >&2 || true
    rm -f "$tf_log"
    exit 1
  fi
  rm -f "$tf_log"
}

# ------------------------------ 全链路检查 -----------------------------------
wait_message /one_x/odom "1X 惯导里程计"

wait_message "$CAMERA_INFO_TOPIC" "外部相机内参"
check_topic_type \
  "$CAMERA_INFO_TOPIC" sensor_msgs/CameraInfo "相机内参"

wait_message "$DETECTIONS_TOPIC" "外部视觉识别结果"
check_topic_type \
  "$DETECTIONS_TOPIC" \
  r300_vision_msgs/DetectedObjectArray \
  "视觉识别结果"
check_camera_tf

wait_message "$OBSTACLE_SCAN_TOPIC" "视觉虚拟 LaserScan"
check_topic_type \
  "$OBSTACLE_SCAN_TOPIC" sensor_msgs/LaserScan "视觉虚拟 LaserScan"

wait_message \
  /move_base/local_costmap/costmap \
  "视觉局部代价地图"
wait_service \
  /subject1/start_waypoints \
  "航点启动服务"

# ------------------------------ 参数一致性 -----------------------------------
ADAPTER_DETECTIONS="$(
  rosparam get /vision_obstacle_layer_node/detections_topic \
    2>/dev/null || true
)"
ADAPTER_CAMERA_INFO="$(
  rosparam get /vision_obstacle_layer_node/camera_info_topic \
    2>/dev/null || true
)"

if [[ "$ADAPTER_DETECTIONS" != "$DETECTIONS_TOPIC" ||
      "$ADAPTER_CAMERA_INFO" != "$CAMERA_INFO_TOPIC" ]]; then
  error "视觉适配节点订阅参数不一致"
  error "detections=${ADAPTER_DETECTIONS:-<空>}，期望 $DETECTIONS_TOPIC"
  error "camera_info=${ADAPTER_CAMERA_INFO:-<空>}，期望 $CAMERA_INFO_TOPIC"
  exit 1
fi
ok "视觉适配节点订阅参数正确"

VISION_LAYER_NS="/move_base/local_costmap/vision_snapshot_layer"
SNAPSHOT_TOPIC="$(
  rosparam get "${VISION_LAYER_NS}/topic" 2>/dev/null || true
)"
SNAPSHOT_HOLD="$(
  rosparam get "${VISION_LAYER_NS}/hold_time_s" 2>/dev/null || true
)"
SNAPSHOT_ACTIVE_SCAN="$(
  rosparam get "${VISION_LAYER_NS}/active_scan_topic" 2>/dev/null || true
)"
PLUGINS="$(
  rosparam get /move_base/local_costmap/plugins 2>/dev/null || true
)"

if [[ "$PLUGINS" != *"vision_snapshot_layer"* ]]; then
  error "move_base 未加载 vision_snapshot_layer 插件"
  error "plugins=${PLUGINS:-<空>}"
  exit 1
fi

if [[ "$SNAPSHOT_TOPIC" != "$OBSTACLE_SCAN_TOPIC" ]]; then
  error "VisionSnapshotLayer 订阅话题错误：${SNAPSHOT_TOPIC:-<空>}"
  error "期望：$OBSTACLE_SCAN_TOPIC"
  exit 1
fi

# 确认YAML实际加载的是有效正数。
if ! awk -v actual="$SNAPSHOT_HOLD" '
  BEGIN {
    exit !(actual ~ /^[0-9]+([.][0-9]+)?$/ && actual > 0.0)
  }
'; then
  error "VisionSnapshotLayer hold_time_s 未正确加载：${SNAPSHOT_HOLD:-<空>}"
  error "请检查 /move_base/local_costmap/vision_snapshot_layer/hold_time_s"
  exit 1
fi

# 默认接受YAML中的实际值。
# 只有显式设置VISION_HOLD_TIME_S时才严格比较。
if [[ -n "$VISION_HOLD_TIME_S" ]]; then
  if ! awk -v actual="$SNAPSHOT_HOLD" -v expected="$VISION_HOLD_TIME_S" '
    BEGIN {
      if (!(expected ~ /^[0-9]+([.][0-9]+)?$/) || expected <= 0.0) {
        exit 2
      }

      diff = actual - expected
      if (diff < 0) {
        diff = -diff
      }

      exit !(diff < 0.000001)
    }
  '; then
    error "VisionSnapshotLayer hold_time_s 参数不一致：${SNAPSHOT_HOLD}"
    error "显式期望值：${VISION_HOLD_TIME_S}s"
    exit 1
  fi
fi

ok "move_base 已加载 VisionSnapshotLayer，odom 中保持 ${SNAPSHOT_HOLD}s"


if ! rostopic info "$OBSTACLE_SCAN_TOPIC" 2>/dev/null | grep -q '/move_base'; then
  error "/move_base 未直接订阅 $OBSTACLE_SCAN_TOPIC"
  exit 1
fi
ok "/move_base 正在直接订阅：$OBSTACLE_SCAN_TOPIC"

if [[ -n "$SNAPSHOT_ACTIVE_SCAN" ]]; then
  wait_message "$SNAPSHOT_ACTIVE_SCAN" "视觉层活动障碍调试扫描"
  check_topic_type "$SNAPSHOT_ACTIVE_SCAN" sensor_msgs/LaserScan \
    "视觉层活动障碍调试扫描"
fi

DWA_MAX_VEL="$(
  rosparam get /move_base/DWAPlannerROS/max_vel_x \
    2>/dev/null || echo unknown
)"
CONTROLLER_FREQUENCY="$(
  rosparam get /move_base/controller_frequency \
    2>/dev/null || echo unknown
)"
INFLATION_RADIUS="$(
  rosparam get \
    /move_base/local_costmap/inflation_layer/inflation_radius \
    2>/dev/null || echo unknown
)"

info "检测坐标系：$CAMERA_FRAME"
info "DWA 最大前进速度：$DWA_MAX_VEL m/s"
info "move_base 控制频率：$CONTROLLER_FREQUENCY Hz"
info "局部地图膨胀半径：$INFLATION_RADIUS m"

# ------------------------------ 是否开始运动 ---------------------------------
if [[ "$AUTO_RUN" == "true" ]]; then
  [[ "$LAUNCH_BASE" == "true" ]] || {
    error "--run 不能与 --no-base 同时使用"; exit 1;
  }
  warn "全链路已就绪，即将启动航点任务。"
  rosservice call /subject1/start_waypoints "{}"
else
  ok "整套视觉导航已就绪，但车辆尚未自动执行航点。"
  echo
  echo "启动航点：rosservice call /subject1/start_waypoints \"{}\""
fi

echo
echo "暂停：rosservice call /subject1/pause_waypoints \"{}\""
echo "恢复：rosservice call /subject1/resume_waypoints \"{}\""
echo "取消：rosservice call /subject1/cancel_waypoints \"{}\""
echo "停止导航：在本终端按 Ctrl+C"
echo

wait "$ROSLAUNCH_PID"