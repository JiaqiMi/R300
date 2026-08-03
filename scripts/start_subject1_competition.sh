#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 科目一比赛完整一键启动 v5
#
# 设计原则：
#   1. 不修改、不复制替换任何原有分别启动脚本；
#   2. 严格调用已经验证可单独运行的原脚本：
#        start_1x.sh
#        start_r300.sh web
#        start_single_lidar_elevation.sh
#        start_r300_lidar_nav.sh
#   3. 仅在本一键脚本及其子进程环境中，安全包装
#      `rosnode list`、`rosservice list`、`rostopic list`。
#      解决子脚本启用 pipefail 时，`list | grep -q` 因 SIGPIPE
#      偶发将“已经存在的节点”误判为“不存在”的问题。
#   4. 不激活、不切换Python/YOLO虚拟环境，继承当前终端环境。
#
# 启动顺序：
#   1X -> D435i/常规视觉/Web -> MID-360/FAST-LIO/高程图
#      -> 完整雷达避障导航（左右牌+自主泊车，无RViz，自动开跑）
#
# 使用：
#   chmod +x ~/r300_ws/scripts/start_subject1_competition.sh
#   ~/r300_ws/scripts/start_subject1_competition.sh
#
# 停止：
#   当前终端按 Ctrl+C。紧急情况优先使用车辆物理急停。

set -Eeuo pipefail

VERSION="v5-original-scripts-only-20260803"

# =============================================================================
# 配置
# =============================================================================
WS="${R300_WS:-$HOME/r300_ws}"
INS_PORT="${INS_PORT:-/dev/ttyACM0}"
INS_BAUD="${INS_BAUD:-460800}"
REQUIRE_VALID_FIX="${REQUIRE_VALID_FIX:-true}"

ONE_X_READY_TIMEOUT="${ONE_X_READY_TIMEOUT:-150}"
VALID_FIX_TIMEOUT="${VALID_FIX_TIMEOUT:-600}"
VISION_READY_TIMEOUT="${VISION_READY_TIMEOUT:-240}"
LIDAR_READY_TIMEOUT="${LIDAR_READY_TIMEOUT:-300}"
NAV_READY_TIMEOUT="${NAV_READY_TIMEOUT:-360}"

START_1X="$WS/src/R300/r300_1x_navigation/scripts/one_key/start_1x.sh"
START_VISION="$WS/scripts/start_r300.sh"
START_LIDAR="$WS/scripts/start_single_lidar_elevation.sh"
START_NAV="$WS/src/R300/r300_1x_navigation/scripts/one_key/start_r300_lidar_nav.sh"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WS/log/subject1_competition/$RUN_ID"
mkdir -p "$LOG_DIR"

LOG_1X="$LOG_DIR/01_one_x.log"
LOG_VISION="$LOG_DIR/02_vision.log"
LOG_LIDAR="$LOG_DIR/03_lidar_elevation.log"
LOG_NAV="$LOG_DIR/04_lidar_navigation.log"

PID_1X=""
PID_VISION=""
PID_LIDAR=""
PID_NAV=""

OWN_1X=false
OWN_VISION=false
OWN_LIDAR=false
OWN_NAV=false
STOPPING=false
WRAPPER_DIR=""

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

# 这些函数本身也使用“先完整获取列表，再精确匹配”的安全方式。
node_exists() {
  local node="$1" snapshot
  snapshot="$(rosnode list 2>/dev/null)" || return 1
  grep -Fx "$node" <<<"$snapshot" >/dev/null
}

service_exists() {
  local service="$1" snapshot
  snapshot="$(rosservice list 2>/dev/null)" || return 1
  grep -Fx "$service" <<<"$snapshot" >/dev/null
}

topic_type_is() {
  local topic="$1" expected="$2"
  [[ "$(rostopic type "$topic" 2>/dev/null || true)" == "$expected" ]]
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
    error "$name 启动脚本已经提前退出。"
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
  local parent="$1" child="$2" timeout_s="$3"
  local name="$4" pid="${5:-}" log_file="$6"
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
  error "等待TF超时：$parent -> $child"
  show_tail "$name" "$log_file"
  return 1
}

wait_valid_fix() {
  if [[ "${REQUIRE_VALID_FIX,,}" != "true" &&
        "$REQUIRE_VALID_FIX" != "1" ]]; then
    warn "已跳过非零有效经纬度检查；仅允许台架联调。"
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

rospy.init_node(
    "subject1_competition_wait_fix",
    anonymous=True,
    disable_signals=True,
)

while not rospy.is_shutdown() and time.monotonic() < deadline:
    try:
        msg = rospy.wait_for_message(
            "/one_x/fix",
            NavSatFix,
            timeout=2.0,
        )
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
        print(
            f"[ OK ] 1X有效位置：lat={lat:.10f}, "
            f"lon={lon:.10f}, status={status}"
        )
        sys.exit(0)

    now = time.monotonic()
    if now - last_print >= 5.0:
        print(
            f"[WARN] 1X位置暂不可用：lat={lat:.10f}, "
            f"lon={lon:.10f}, status={status}，继续等待……",
            flush=True,
        )
        last_print = now

print(
    "[ERR ] 等待1X有效位置超时。请检查GNSS天线、室外环境和1X状态。",
    file=sys.stderr,
)
sys.exit(1)
PY
}

wait_waypoint_running() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))
  local status

  while (( SECONDS < deadline )); do
    check_owner "雷达避障导航" "$PID_NAV" "$LOG_NAV" || return 1

    status="$(
      timeout 4 rostopic echo -n 1 \
        /subject1/waypoint_status 2>/dev/null || true
    )"

    if grep -Eq 'state=RUNNING|data:.*state=RUNNING' <<<"$status"; then
      return 0
    fi
    sleep 1
  done

  error "导航已经启动，但航点状态未在${timeout_s}s内进入RUNNING。"
  show_tail "雷达避障导航" "$LOG_NAV"
  return 1
}

# =============================================================================
# 仅在当前一键脚本的子进程环境中包装ROS list命令
# =============================================================================
install_safe_ros_list_wrappers() {
  local real_rosnode real_rosservice real_rostopic wrapper

  real_rosnode="$(command -v rosnode)"
  real_rosservice="$(command -v rosservice)"
  real_rostopic="$(command -v rostopic)"

  [[ -n "$real_rosnode" && -n "$real_rosservice" &&
     -n "$real_rostopic" ]] || {
    error "无法定位rosnode/rosservice/rostopic"
    return 1
  }

  WRAPPER_DIR="$(mktemp -d /tmp/r300_ros_safe_list.XXXXXX)"
  wrapper="$WRAPPER_DIR/ros_safe_wrapper.py"

  cat >"$wrapper" <<'PY'
#!/usr/bin/env python3
"""Transparent ROS CLI wrapper.

Only the `list` subcommand is buffered completely before writing to stdout.
This prevents an upstream SIGPIPE when a child shell uses:

    set -o pipefail
    rosnode list | grep -q ...

All other subcommands are exec'd directly to the original ROS executable.
"""

import os
import subprocess
import sys

name = os.path.basename(sys.argv[0])
env_name = {
    "rosnode": "R300_REAL_ROSNODE",
    "rosservice": "R300_REAL_ROSSERVICE",
    "rostopic": "R300_REAL_ROSTOPIC",
}.get(name)

if not env_name:
    print(f"Unsupported wrapper name: {name}", file=sys.stderr)
    sys.exit(127)

real = os.environ.get(env_name)
if not real:
    print(f"Missing environment variable: {env_name}", file=sys.stderr)
    sys.exit(127)

args = sys.argv[1:]

if args and args[0] == "list":
    result = subprocess.run(
        [real] + args,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )

    try:
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        # `grep -q` may close its input immediately after finding a match.
        # Redirect stdout before interpreter shutdown so this wrapper still
        # returns the real ROS command's result instead of SIGPIPE/exit 141.
        try:
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, sys.stdout.fileno())
        except OSError:
            pass

    sys.exit(result.returncode)

os.execv(real, [real] + args)
PY

  chmod +x "$wrapper"
  ln -s "$wrapper" "$WRAPPER_DIR/rosnode"
  ln -s "$wrapper" "$WRAPPER_DIR/rosservice"
  ln -s "$wrapper" "$WRAPPER_DIR/rostopic"

  export R300_REAL_ROSNODE="$real_rosnode"
  export R300_REAL_ROSSERVICE="$real_rosservice"
  export R300_REAL_ROSTOPIC="$real_rostopic"
  export PATH="$WRAPPER_DIR:$PATH"

  ok "已为本次一键启动安装安全ROS列表包装。原始脚本未被修改。"
}

# =============================================================================
# 停止与清理
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

  info "停止$name（PID=$pid）……"
  kill -INT "$pid" 2>/dev/null || true

  for _ in {1..40}; do
    alive "$pid" || return 0
    sleep 0.25
  done

  warn "$name未在10秒内退出，停止完整子进程树。"
  signal_tree INT "$pid"

  for _ in {1..24}; do
    alive "$pid" || return 0
    sleep 0.25
  done

  signal_tree TERM "$pid"
  for _ in {1..16}; do
    alive "$pid" || return 0
    sleep 0.25
  done

  signal_tree KILL "$pid"
}

emergency_stop() {
  timeout 2 rosservice call \
    /subject1/cancel_waypoints "{}" >/dev/null 2>&1 || true

  timeout 2 rostopic pub -1 \
    /move_base/cancel \
    actionlib_msgs/GoalID \
    "{stamp: {secs: 0, nsecs: 0}, id: ''}" \
    >/dev/null 2>&1 || true

  timeout 2 rostopic pub -1 \
    /subject1/cmd_vel_raw \
    geometry_msgs/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0},
      angular: {x: 0.0, y: 0.0, z: 0.0}}" \
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

    [[ "$OWN_NAV" == "true" ]] &&
      stop_component "雷达避障导航" "$PID_NAV"

    [[ "$OWN_LIDAR" == "true" ]] &&
      stop_component "雷达和高程图" "$PID_LIDAR"

    [[ "$OWN_VISION" == "true" ]] &&
      stop_component "相机和常规视觉" "$PID_VISION"

    [[ "$OWN_1X" == "true" ]] &&
      stop_component "1X惯导" "$PID_1X"

    [[ -n "$WRAPPER_DIR" && -d "$WRAPPER_DIR" ]] &&
      rm -rf "$WRAPPER_DIR"

    ok "停止流程完成。日志目录：$LOG_DIR"
  fi

  exit "$rc"
}
trap cleanup INT TERM EXIT

# =============================================================================
# 环境与文件检查
# =============================================================================
[[ -f /opt/ros/noetic/setup.bash ]] || {
  error "未安装ROS Noetic"
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
export PYTHONUNBUFFERED=1

for cmd in rosnode rosservice rostopic rosrun timeout python3; do
  command -v "$cmd" >/dev/null 2>&1 || {
    error "缺少命令：$cmd"
    exit 1
  }
done

for file in "$START_1X" "$START_VISION" "$START_LIDAR" "$START_NAV"; do
  [[ -f "$file" ]] || {
    error "缺少原始启动脚本：$file"
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
  exit 1
}

# 必须在定位真实ROS命令后再改变PATH。
install_safe_ros_list_wrappers

if node_exists /move_base; then
  error "检测到已有/move_base，请先停止旧导航。"
  exit 1
fi

step "科目一比赛完整一键启动"
echo "脚本版本：$VERSION"
echo "工作空间：$WS"
echo "日志目录：$LOG_DIR"
echo "1X串口：$INS_PORT @ $INS_BAUD"
echo "原始分别启动脚本：保持不变"
echo "最终模式：左右牌=开启，自主泊车=开启，RViz=关闭，自动开跑=开启"

# =============================================================================
# 1/4：调用原始start_1x.sh
# =============================================================================
step "1/4 调用原始 start_1x.sh"

if node_exists /one_x_serial_driver; then
  warn "检测到已有/one_x_serial_driver，将验证并复用。"
else
  start_bg PID_1X "$LOG_1X" \
    env \
      R300_WS="$WS" \
      INS_PORT="$INS_PORT" \
      INS_BAUD="$INS_BAUD" \
    bash "$START_1X"

  OWN_1X=true
  info "原始start_1x.sh已启动，PID=$PID_1X，日志：$LOG_1X"
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

ok "原始start_1x.sh运行正常；1X位置和原点服务已就绪，尚未设置导航原点。"

# =============================================================================
# 2/4：调用原始start_r300.sh web
# =============================================================================
step "2/4 调用原始 start_r300.sh web"

if node_exists /r300_dual_yolo_depth_node; then
  warn "检测到常规双YOLO已运行，将验证并复用。"
else
  start_bg PID_VISION "$LOG_VISION" \
    env R300_WS="$WS" \
    bash "$START_VISION" web

  OWN_VISION=true
  info "原始start_r300.sh已启动，PID=$PID_VISION，日志：$LOG_VISION"
fi

wait_node /r300_dual_yolo_depth_node "$VISION_READY_TIMEOUT" \
  "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_node /r300_web_video_server "$VISION_READY_TIMEOUT" \
  "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /camera/color/image_raw sensor_msgs/Image 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages \
  /camera/aligned_depth_to_color/image_raw sensor_msgs/Image 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /camera/color/camera_info sensor_msgs/CameraInfo 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages /r300_vision/annotated_image sensor_msgs/Image 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

wait_topic_messages \
  /r300_vision/detections r300_vision_msgs/DetectedObjectArray 2 \
  "$VISION_READY_TIMEOUT" "视觉系统" "$PID_VISION" "$LOG_VISION"

ok "原始视觉脚本运行正常；D435i、常规识别、左右牌输入和Web均已就绪。"

# =============================================================================
# 3/4：调用原始雷达和高程图脚本
# =============================================================================
step "3/4 调用原始 start_single_lidar_elevation.sh"

if topic_type_is \
     /elevation_mapping/elevation_map_raw grid_map_msgs/GridMap &&
   timeout 5 rostopic echo -n 2 \
     /elevation_mapping/elevation_map_raw >/dev/null 2>&1; then

  warn "检测到高程图已经持续发布，将验证并复用现有雷达感知栈。"
else
  start_bg PID_LIDAR "$LOG_LIDAR" \
    env \
      R300_WS="$WS" \
      RVIZ=0 \
      RESTART_EXISTING=1 \
    bash "$START_LIDAR"

  OWN_LIDAR=true
  info "原始雷达脚本已启动，PID=$PID_LIDAR，日志：$LOG_LIDAR"
fi

wait_topic_messages /cloud_registered_body sensor_msgs/PointCloud2 3 \
  "$LIDAR_READY_TIMEOUT" "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

wait_topic_messages /Odometry nav_msgs/Odometry 3 \
  "$LIDAR_READY_TIMEOUT" "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

wait_tf odom body "$LIDAR_READY_TIMEOUT" \
  "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

wait_topic_messages \
  /elevation_mapping/elevation_map_raw grid_map_msgs/GridMap 3 \
  "$LIDAR_READY_TIMEOUT" "雷达感知" "$PID_LIDAR" "$LOG_LIDAR"

sleep 2
check_owner "雷达感知" "$PID_LIDAR" "$LOG_LIDAR" || exit 1

ok "原始雷达脚本运行正常；MID-360、FAST-LIO、TF和高程图均已就绪。"

# =============================================================================
# 4/4：调用原始完整版雷达导航脚本
# =============================================================================
step "4/4 调用原始完整版 start_r300_lidar_nav.sh"

start_bg PID_NAV "$LOG_NAV" \
  env R300_WS="$WS" \
  bash "$START_NAV" \
    --sign-guidance \
    --auto-parking \
    --no-rviz \
    --run

OWN_NAV=true
info "原始雷达导航脚本已启动，PID=$PID_NAV，日志：$LOG_NAV"

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

while alive "$PID_NAV"; do
  sleep 2
done

error "雷达避障导航脚本已经退出，整套系统将安全停止。"
show_tail "雷达避障导航" "$LOG_NAV"
exit 1
