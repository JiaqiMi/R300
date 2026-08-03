#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 科目一比赛完整一键启动（稳定版 v3）
#
# 启动顺序：
#   0. 固定并确认同一个 ROS Master
#   1. 直接启动 1X 底层 launch（不再嵌套调用 start_1x.sh）
#   2. 启动 D435i + 常规双 YOLO + Web
#   3. 启动 MID-360 + FAST-LIO + elevation_mapping，RViz关闭
#   4. 启动完整版雷达避障导航（左右牌 + 自主泊车），RViz关闭并自动开跑
#
# 使用：
#   chmod +x ~/r300_ws/scripts/start_subject1_competition.sh
#   ~/r300_ws/scripts/start_subject1_competition.sh
#
# 停止：
#   当前终端按 Ctrl+C。紧急情况优先按车辆物理急停。
#
# 可覆盖环境变量：
#   R300_WS=/home/explorer/r300_ws
#   INS_PORT=/dev/ttyACM0
#   INS_BAUD=460800
#   REQUIRE_VALID_FIX=true
#   VALID_FIX_TIMEOUT=600
#   VISION_READY_TIMEOUT=180
#   LIDAR_READY_TIMEOUT=240
#   NAV_READY_TIMEOUT=300

set -Eeuo pipefail

VERSION="v3-direct-1x-launch-20260803"

# =============================================================================
# 配置
# =============================================================================
WS="${R300_WS:-$HOME/r300_ws}"
INS_PORT="${INS_PORT:-/dev/ttyACM0}"
INS_BAUD="${INS_BAUD:-460800}"
REQUIRE_VALID_FIX="${REQUIRE_VALID_FIX:-true}"

MASTER_TIMEOUT="${MASTER_TIMEOUT:-20}"
ONE_X_READY_TIMEOUT="${ONE_X_READY_TIMEOUT:-120}"
VALID_FIX_TIMEOUT="${VALID_FIX_TIMEOUT:-600}"
VISION_READY_TIMEOUT="${VISION_READY_TIMEOUT:-180}"
LIDAR_READY_TIMEOUT="${LIDAR_READY_TIMEOUT:-240}"
NAV_READY_TIMEOUT="${NAV_READY_TIMEOUT:-300}"

ROS_MASTER_URI_FIXED="${ROS_MASTER_URI_FIXED:-http://127.0.0.1:11311}"

START_VISION="$WS/scripts/start_r300.sh"
START_LIDAR="$WS/scripts/start_single_lidar_elevation.sh"
START_NAV="$WS/src/R300/r300_1x_navigation/scripts/one_key/start_r300_lidar_nav.sh"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WS/log/subject1_competition/$RUN_ID"
mkdir -p "$LOG_DIR"

LOG_MASTER="$LOG_DIR/00_roscore.log"
LOG_1X="$LOG_DIR/01_one_x.log"
LOG_VISION="$LOG_DIR/02_vision.log"
LOG_LIDAR="$LOG_DIR/03_lidar_elevation.log"
LOG_NAV="$LOG_DIR/04_lidar_navigation.log"

PID_MASTER=""
PID_1X=""
PID_VISION=""
PID_LIDAR=""
PID_NAV=""

OWN_MASTER=false
OWN_1X=false
OWN_VISION=false
OWN_LIDAR=false
OWN_NAV=false
STOPPING=false

# =============================================================================
# 输出
# =============================================================================
info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }
step()  { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$*"; }

show_tail() {
  local title="$1" file="$2" lines="${3:-100}"
  error "$title 最近日志："
  if [[ -f "$file" ]]; then
    tail -n "$lines" "$file" >&2 || true
  else
    error "日志不存在：$file"
  fi
}

alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

node_exists() {
  local node="$1"
  rosnode list 2>/dev/null | grep -Fxq "$node"
}

service_exists() {
  local service="$1"
  rosservice list 2>/dev/null | grep -Fxq "$service"
}

start_bg() {
  local __pid_var="$1" log_file="$2"
  shift 2
  : >"$log_file"
  "$@" >"$log_file" 2>&1 &
  local child_pid=$!
  printf -v "$__pid_var" '%s' "$child_pid"
}

check_owner() {
  local name="$1" pid="${2:-}" log_file="$3"
  [[ -z "$pid" ]] && return 0
  if ! alive "$pid"; then
    error "$name 进程已经提前退出。"
    show_tail "$name" "$log_file"
    return 1
  fi
}

wait_node() {
  local node="$1" timeout_s="$2" name="$3" pid="${4:-}" log_file="$5"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    node_exists "$node" && return 0
    check_owner "$name" "$pid" "$log_file" || return 1
    sleep 0.5
  done
  error "等待节点超时：$node（${timeout_s}s）"
  show_tail "$name" "$log_file"
  return 1
}

wait_service() {
  local service="$1" timeout_s="$2" name="$3" pid="${4:-}" log_file="$5"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    service_exists "$service" && return 0
    check_owner "$name" "$pid" "$log_file" || return 1
    sleep 0.5
  done
  error "等待服务超时：$service（${timeout_s}s）"
  show_tail "$name" "$log_file"
  return 1
}

topic_type_is() {
  local topic="$1" expected="$2"
  [[ "$(rostopic type "$topic" 2>/dev/null || true)" == "$expected" ]]
}

wait_topic_messages() {
  local topic="$1" expected_type="$2" count="$3" timeout_s="$4"
  local name="$5" pid="${6:-}" log_file="$7"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if topic_type_is "$topic" "$expected_type"; then
      if timeout 8 rostopic echo -n "$count" "$topic" >/dev/null 2>&1; then
        return 0
      fi
    fi
    check_owner "$name" "$pid" "$log_file" || return 1
    sleep 0.5
  done

  error "等待话题消息超时：$topic，期望类型=$expected_type"
  show_tail "$name" "$log_file"
  return 1
}

wait_tf() {
  local parent="$1" child="$2" timeout_s="$3" name="$4" pid="${5:-}" log_file="$6"
  local deadline=$((SECONDS + timeout_s))
  local tmp
  tmp="$(mktemp)"

  while (( SECONDS < deadline )); do
    : >"$tmp"
    timeout 3 rosrun tf tf_echo "$parent" "$child" >"$tmp" 2>&1 || true
    if grep -q "Translation" "$tmp"; then
      rm -f "$tmp"
      return 0
    fi
    check_owner "$name" "$pid" "$log_file" || {
      rm -f "$tmp"
      return 1
    }
    sleep 1
  done

  rm -f "$tmp"
  error "等待 TF 超时：$parent -> $child"
  show_tail "$name" "$log_file"
  return 1
}

wait_waypoint_running() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))
  local status

  while (( SECONDS < deadline )); do
    check_owner "雷达避障导航" "$PID_NAV" "$LOG_NAV" || return 1
    status="$(timeout 4 rostopic echo -n 1 /subject1/waypoint_status 2>/dev/null || true)"
    if grep -Eq 'state=RUNNING|data:.*state=RUNNING' <<<"$status"; then
      return 0
    fi
    sleep 1
  done

  error "导航已启动，但航点状态未在 ${timeout_s}s 内进入 RUNNING。"
  show_tail "雷达避障导航" "$LOG_NAV"
  return 1
}

wait_valid_fix() {
  if [[ "${REQUIRE_VALID_FIX,,}" != "true" && "$REQUIRE_VALID_FIX" != "1" ]]; then
    warn "跳过非零有效经纬度检查，仅允许台架联调，禁止正式比赛使用。"
    return 0
  fi

  info "等待 /one_x/fix 给出非零有效位置（最多 ${VALID_FIX_TIMEOUT}s）……"
  FIX_TIMEOUT="$VALID_FIX_TIMEOUT" python3 - <<'PY'
import math
import os
import sys
import time

import rospy
from sensor_msgs.msg import NavSatFix

timeout_s = float(os.environ.get("FIX_TIMEOUT", "600"))
deadline = time.monotonic() + timeout_s
last_print = 0.0

rospy.init_node("subject1_competition_wait_fix",
                anonymous=True, disable_signals=True)

while not rospy.is_shutdown() and time.monotonic() < deadline:
    try:
        msg = rospy.wait_for_message("/one_x/fix", NavSatFix, timeout=2.0)
    except Exception:
        continue

    lat = float(msg.latitude)
    lon = float(msg.longitude)
    status = int(msg.status.status)

    valid = (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
        and abs(lat) > 1.0e-9
        and abs(lon) > 1.0e-9
        and status >= 0
    )

    if valid:
        print(f"[ OK ] 1X有效位置：lat={lat:.10f}, "
              f"lon={lon:.10f}, status={status}")
        sys.exit(0)

    now = time.monotonic()
    if now - last_print >= 5.0:
        print(f"[WARN] 1X位置暂不可用：lat={lat:.10f}, "
              f"lon={lon:.10f}, status={status}，继续等待……",
              flush=True)
        last_print = now

print("[ERR ] 等待1X有效位置超时。请检查GNSS天线、室外环境和1X状态。",
      file=sys.stderr)
sys.exit(1)
PY
}

# =============================================================================
# 停止
# =============================================================================
descendants_postorder() {
  local root="$1" child
  while read -r child; do
    [[ -n "$child" ]] || continue
    descendants_postorder "$child"
    printf '%s\n' "$child"
  done < <(pgrep -P "$root" 2>/dev/null || true)
}

signal_tree() {
  local sig="$1" root="$2" child
  while read -r child; do
    [[ -n "$child" ]] && kill -"$sig" "$child" 2>/dev/null || true
  done < <(descendants_postorder "$root")
  kill -"$sig" "$root" 2>/dev/null || true
}

stop_component() {
  local name="$1" pid="${2:-}"
  [[ -n "$pid" ]] || return 0
  alive "$pid" || return 0

  info "停止 $name（PID=$pid）……"
  kill -TERM "$pid" 2>/dev/null || true

  for _ in {1..40}; do
    alive "$pid" || return 0
    sleep 0.25
  done

  warn "$name 未在10秒内退出，发送INT到完整进程树。"
  signal_tree INT "$pid"
  for _ in {1..24}; do
    alive "$pid" || return 0
    sleep 0.25
  done

  warn "$name 仍未退出，发送TERM到完整进程树。"
  signal_tree TERM "$pid"
  for _ in {1..16}; do
    alive "$pid" || return 0
    sleep 0.25
  done

  warn "$name 仍未退出，最终发送KILL。"
  signal_tree KILL "$pid"
}

emergency_stop() {
  timeout 2 rosservice call /subject1/cancel_waypoints "{}" \
    >/dev/null 2>&1 || true
  timeout 2 rostopic pub -1 /move_base/cancel actionlib_msgs/GoalID \
    "{stamp: {secs: 0, nsecs: 0}, id: ''}" \
    >/dev/null 2>&1 || true
  timeout 2 rostopic pub -1 /subject1/cmd_vel_raw geometry_msgs/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    >/dev/null 2>&1 || true
}

cleanup() {
  local rc=$?
  trap - INT TERM EXIT

  if [[ "$STOPPING" == "false" ]]; then
    STOPPING=true
    echo
    warn "正在安全停止科目一整套系统……"
    emergency_stop

    [[ "$OWN_NAV" == "true" ]] && stop_component "雷达避障导航" "$PID_NAV"
    [[ "$OWN_LIDAR" == "true" ]] && stop_component "雷达和高程图" "$PID_LIDAR"
    [[ "$OWN_VISION" == "true" ]] && stop_component "相机和常规视觉" "$PID_VISION"
    [[ "$OWN_1X" == "true" ]] && stop_component "1X惯导" "$PID_1X"
    [[ "$OWN_MASTER" == "true" ]] && stop_component "ROS Master" "$PID_MASTER"

    ok "停止流程完成。日志目录：$LOG_DIR"
  fi
  exit "$rc"
}
trap cleanup INT TERM EXIT

# =============================================================================
# 环境与预检
# =============================================================================
[[ -f /opt/ros/noetic/setup.bash ]] || {
  error "未安装ROS Noetic：/opt/ros/noetic/setup.bash不存在"
  exit 1
}
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

[[ -f "$WS/devel/setup.bash" ]] || {
  error "工作空间尚未编译：$WS/devel/setup.bash不存在"
  exit 1
}
# shellcheck disable=SC1090
source "$WS/devel/setup.bash"

export R300_WS="$WS"
export ROS_MASTER_URI="$ROS_MASTER_URI_FIXED"
export PYTHONUNBUFFERED=1

for cmd in roscore roslaunch rosnode rostopic rosservice rosrun rosparam timeout python3; do
  command -v "$cmd" >/dev/null 2>&1 || {
    error "缺少命令：$cmd"
    exit 1
  }
done

for file in "$START_VISION" "$START_LIDAR" "$START_NAV"; do
  [[ -f "$file" ]] || {
    error "缺少启动脚本：$file"
    exit 1
  }
  [[ -x "$file" ]] || chmod +x "$file"
done

[[ -e "$INS_PORT" ]] || {
  error "1X串口不存在：$INS_PORT"
  exit 1
}
[[ -r "$INS_PORT" && -w "$INS_PORT" ]] || {
  error "当前用户无权读写1X串口：$INS_PORT"
  error "请检查dialout用户组或udev规则。"
  exit 1
}

if node_exists /move_base; then
  error "检测到已有 /move_base。请先停止旧导航后再启动。"
  exit 1
fi

step "科目一比赛完整一键启动"
echo "脚本版本：$VERSION"
echo "工作空间：$WS"
echo "日志目录：$LOG_DIR"
echo "ROS Master：$ROS_MASTER_URI"
echo "1X串口：$INS_PORT @ $INS_BAUD"
echo "最终模式：左右牌=开启，自主泊车=开启，RViz=关闭，自动开跑=开启"

# =============================================================================
# 0/4：固定ROS Master
# =============================================================================
step "0/4 固定 ROS Master"

if rosparam get /run_id >/dev/null 2>&1; then
  ok "检测到现有ROS Master，直接复用。"
else
  start_bg PID_MASTER "$LOG_MASTER" roscore -p 11311
  OWN_MASTER=true
  info "已启动ROS Master，PID=$PID_MASTER"

  deadline=$((SECONDS + MASTER_TIMEOUT))
  while (( SECONDS < deadline )); do
    if rosparam get /run_id >/dev/null 2>&1 &&
       node_exists /rosout; then
      break
    fi
    check_owner "ROS Master" "$PID_MASTER" "$LOG_MASTER" || exit 1
    sleep 0.5
  done

  rosparam get /run_id >/dev/null 2>&1 || {
    error "ROS Master未在${MASTER_TIMEOUT}s内就绪。"
    show_tail "ROS Master" "$LOG_MASTER"
    exit 1
  }
  node_exists /rosout || {
    error "ROS Master已响应，但/rosout未注册。"
    show_tail "ROS Master" "$LOG_MASTER"
    exit 1
  }
  ok "ROS Master已经稳定就绪。"
fi

# =============================================================================
# 1/4：直接启动1X底层launch
# =============================================================================
step "1/4 启动 1X 惯导"

if node_exists /one_x_serial_driver; then
  warn "检测到已有 /one_x_serial_driver，将验证后复用。"
else
  start_bg PID_1X "$LOG_1X" \
    roslaunch --wait r300_1x_navigation one_x_localization_only.launch \
      serial_port:="$INS_PORT" \
      baudrate:="$INS_BAUD" \
      publish_full_attitude:=false \
      origin_mode:=deferred \
      origin_set_max_age_s:=0.50
  OWN_1X=true
  info "1X底层launch已启动，PID=$PID_1X，日志：$LOG_1X"
fi

wait_node /one_x_serial_driver "$ONE_X_READY_TIMEOUT" \
  "1X惯导" "$PID_1X" "$LOG_1X"

wait_topic_messages /one_x/ins_fix sensor_msgs/NavSatFix 2 \
  "$ONE_X_READY_TIMEOUT" "1X惯导" "$PID_1X" "$LOG_1X"

wait_topic_messages /one_x/fix sensor_msgs/NavSatFix 2 \
  "$ONE_X_READY_TIMEOUT" "1X惯导" "$PID_1X" "$LOG_1X"

wait_service /one_x/set_current_origin "$ONE_X_READY_TIMEOUT" \
  "1X惯导" "$PID_1X" "$LOG_1X"

wait_valid_fix
ok "1X原始解析、位置话题和原点服务均已就绪；当前尚未设置导航原点。"

# =============================================================================
# 2/4：视觉
# =============================================================================
step "2/4 启动 D435i、常规识别和 Web"

if node_exists /r300_dual_yolo_depth_node; then
  warn "检测到常规双YOLO已运行，将验证后复用。"
else
  start_bg PID_VISION "$LOG_VISION" \
    env ROS_MASTER_URI="$ROS_MASTER_URI" R300_WS="$WS" \
    bash "$START_VISION" web
  OWN_VISION=true
  info "视觉启动脚本已启动，PID=$PID_VISION，日志：$LOG_VISION"
fi

wait_node /r300_dual_yolo_depth_node "$VISION_READY_TIMEOUT" \
  "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_node /r300_web_video_server "$VISION_READY_TIMEOUT" \
  "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /camera/color/image_raw sensor_msgs/Image 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /camera/aligned_depth_to_color/image_raw sensor_msgs/Image 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /camera/color/camera_info sensor_msgs/CameraInfo 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /r300_vision/annotated_image sensor_msgs/Image 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /r300_vision/detections r300_vision_msgs/DetectedObjectArray 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

ok "D435i、常规识别、左右牌检测输入和Web视频均已就绪。"

# =============================================================================
# 3/4：雷达和高程图
# =============================================================================
step "3/4 启动 MID-360、FAST-LIO 和高程图"

if topic_type_is /elevation_mapping/elevation_map_raw grid_map_msgs/GridMap &&
   timeout 5 rostopic echo -n 2 /elevation_mapping/elevation_map_raw \
     >/dev/null 2>&1; then
  warn "检测到现有高程图持续发布，将验证后复用雷达感知栈。"
else
  start_bg PID_LIDAR "$LOG_LIDAR" \
    env ROS_MASTER_URI="$ROS_MASTER_URI" R300_WS="$WS" \
        RVIZ=0 RESTART_EXISTING=1 \
    bash "$START_LIDAR"
  OWN_LIDAR=true
  info "雷达启动脚本已启动，PID=$PID_LIDAR，日志：$LOG_LIDAR"
fi

wait_topic_messages /cloud_registered_body sensor_msgs/PointCloud2 3 \
  "$LIDAR_READY_TIMEOUT" "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

wait_topic_messages /Odometry nav_msgs/Odometry 3 \
  "$LIDAR_READY_TIMEOUT" "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

wait_tf odom body "$LIDAR_READY_TIMEOUT" \
  "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

wait_topic_messages /elevation_mapping/elevation_map_raw grid_map_msgs/GridMap 3 \
  "$LIDAR_READY_TIMEOUT" "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

# 让TF缓存和高程图再稳定两秒，避免启动即刻的外推问题。
sleep 2
check_owner "雷达感知" "$PID_LIDAR" "$LOG_LIDAR" || exit 1

ok "MID-360、FAST-LIO、odom->body TF和高程图均已完全就绪。"

# =============================================================================
# 4/4：完整版雷达避障导航
# =============================================================================
step "4/4 启动完整版雷达避障导航并自动开跑"

start_bg PID_NAV "$LOG_NAV" \
  env ROS_MASTER_URI="$ROS_MASTER_URI" R300_WS="$WS" \
  bash "$START_NAV" \
    --sign-guidance \
    --auto-parking \
    --no-rviz \
    --run
OWN_NAV=true
info "雷达导航脚本已启动，PID=$PID_NAV，日志：$LOG_NAV"

for node in \
  /move_base \
  /scout_base_node \
  /waypoint_executor \
  /lidar_obstacle_scan_node \
  /direction_sign_local_goal \
  /r300_autonomous_parking \
  /one_x_alignment_gate
do
  wait_node "$node" "$NAV_READY_TIMEOUT" \
    "雷达避障导航" "$PID_NAV" "$LOG_NAV"
done

wait_topic_messages /subject1/dwa_odom nav_msgs/Odometry 2 \
  "$NAV_READY_TIMEOUT" "雷达避障导航" "$PID_NAV" "$LOG_NAV"

wait_topic_messages /r300_lidar/obstacle_scan sensor_msgs/LaserScan 2 \
  "$NAV_READY_TIMEOUT" "雷达避障导航" "$PID_NAV" "$LOG_NAV"

wait_service /subject1/start_waypoints "$NAV_READY_TIMEOUT" \
  "雷达避障导航" "$PID_NAV" "$LOG_NAV"

wait_waypoint_running "$NAV_READY_TIMEOUT"

ok "科目一完整系统已启动，航点任务正在自动运行。"

echo
echo "常用状态："
echo "  rostopic echo /subject1/waypoint_status"
echo "  rostopic echo /subject1/direction_sign/state"
echo "  rostopic echo /subject1/autonomous_parking/state"
echo
echo "暂停：rosservice call /subject1/pause_waypoints \"{}\""
echo "恢复：rosservice call /subject1/resume_waypoints \"{}\""
echo "跳过：rosservice call /subject1/skip_waypoint \"{}\""
echo "取消：rosservice call /subject1/cancel_waypoints \"{}\""
echo
echo "日志目录：$LOG_DIR"
echo "停止整套系统：本终端按 Ctrl+C"
echo

# 导航脚本应在比赛期间持续运行；若它异常退出，自动触发整套清理。
while alive "$PID_NAV"; do
  sleep 2
done

error "雷达避障导航进程已退出，整套系统将安全停止。"
show_tail "雷达避障导航" "$LOG_NAV"
exit 1
