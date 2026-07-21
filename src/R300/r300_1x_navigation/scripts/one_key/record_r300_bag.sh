#!/usr/bin/env bash
set -Eeuo pipefail

# 一键录制 R300/1X 导航分析 bag。
#
# 用法：
#   ./record_r300_bag.sh
#   OUT=~/bags/test_01 ./record_r300_bag.sh
#   WAIT_TIMEOUT=20 ./record_r300_bag.sh
#
# 说明：
# - 使用 LZ4，兼顾压缩速度与工控机负载；
# - 新增的 1X 原始分析话题按每个有效串口帧发布，通常约 100 Hz；
# - 不再录制 /one_x/fix、/one_x/heading_deg、/one_x/pos_compare；
# - 不录制 /one_x/path，避免整段历史 Path 高频重复写入导致 bag 快速膨胀。

WS="${R300_WS:-$HOME/r300_ws}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-15}"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }

[[ -f /opt/ros/noetic/setup.bash ]] || {
  error "未找到 ROS Noetic"
  exit 1
}
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

if [[ -f "$WS/devel/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "$WS/devel/setup.bash"
else
  error "未找到 $WS/devel/setup.bash，请先编译工作空间"
  exit 1
fi

if ! rosparam get /rosdistro >/dev/null 2>&1; then
  error "ROS master 未运行，请先启动实时导航或 1X 定位"
  exit 1
fi

OUT="${OUT:-$HOME/bags/r300_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT%.bag}"
mkdir -p "$(dirname "$OUT")"

# 这些话题必须存在，否则通常说明新驱动没有重新编译或没有重启。
REQUIRED_TOPICS=(
  /one_x/odom
  /one_x/ins_fix
  /one_x/gps_fix
  /one_x/imu
  /one_x/ins_imu
  /one_x/ins_status
  /one_x/attitude
  /one_x/vel
  /one_x/update_flag
)

info "等待新的 1X 高频分析话题……"
for ((i = 0; i < WAIT_TIMEOUT; ++i)); do
  missing=()
  topic_list="$(rostopic list 2>/dev/null || true)"
  for topic in "${REQUIRED_TOPICS[@]}"; do
    if ! grep -qx "$topic" <<<"$topic_list"; then
      missing+=("$topic")
    fi
  done

  if ((${#missing[@]} == 0)); then
    ok "1X 高频分析话题已全部就绪"
    break
  fi

  if ((i == WAIT_TIMEOUT - 1)); then
    error "等待 ${WAIT_TIMEOUT}s 后仍缺少以下话题："
    printf '  %s\n' "${missing[@]}" >&2
    error "请确认已重新 catkin_make，并重启 one_x_serial_driver"
    exit 1
  fi
  sleep 1
done

TOPICS=(
  # 控制、里程计与定位
  /subject1/cmd_vel_raw
  /one_x/odom
  /subject1/dwa_odom
  /one_x/origin

  # INS/GNSS 原始分析数据（有效串口帧频率，通常约 100 Hz）
  /one_x/ins_fix
  /one_x/gps_fix
  /one_x/imu
  /one_x/ins_imu
  /one_x/ins_status
  /one_x/attitude
  /one_x/vel
  /one_x/update_flag
  /one_x/diagnostics

  # move_base 规划与任务状态
  /move_base/NavfnROS/plan
  /move_base/DWAPlannerROS/local_plan
  /move_base/current_goal
  /move_base/status
  /move_base/result

  # 视觉障碍链路（不存在时 rosbag 会等待话题出现）
  /r300_vision/obstacle_scan
  /r300_vision/active_obstacle_scan

  # 坐标变换
  /tf
  /tf_static
)

printf '\n'
info "rosbag 输出：$OUT.bag"
info "压缩方式：LZ4"
info "录制话题："
printf '  %s\n' "${TOPICS[@]}"
printf '\n'
info "按 Ctrl+C 正常结束录制并写入 bag 索引"

# exec 让 Ctrl+C 直接交给 rosbag，确保正常关闭 .bag.active 文件。
exec rosbag record \
  --lz4 \
  --buffsize=512 \
  -O "$OUT" \
  "${TOPICS[@]}"
