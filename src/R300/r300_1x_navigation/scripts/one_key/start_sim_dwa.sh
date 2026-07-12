#!/usr/bin/env bash
set -euo pipefail

# 一键启动：DWA 闭环仿真
# 用法：
#   ./start_sim_dwa.sh
#   RVIZ=false ./start_sim_dwa.sh
#   DRIFT_Y_MPS=0.03 YAW_NOISE_DEG=0.5 ./start_sim_dwa.sh

WS="${R300_WS:-$HOME/r300_ws}"
PKG="${R300_PKG:-r300_1x_navigation}"
RVIZ="${RVIZ:-true}"
ODOM_PATH="${ODOM_PATH:-true}"

SIM_MAX_V="${SIM_MAX_V:-1.5}"
SIM_MAX_W="${SIM_MAX_W:-0.6}"
SIM_ACC_V="${SIM_ACC_V:-0.8}"
SIM_ACC_W="${SIM_ACC_W:-1.2}"
SIM_TAU_V="${SIM_TAU_V:-0.20}"
SIM_TAU_W="${SIM_TAU_W:-0.15}"

POS_NOISE="${POS_NOISE:-0.0}"
YAW_NOISE_DEG="${YAW_NOISE_DEG:-0.0}"
DRIFT_X_MPS="${DRIFT_X_MPS:-0.0}"
DRIFT_Y_MPS="${DRIFT_Y_MPS:-0.0}"
YAW_DRIFT_DEGPS="${YAW_DRIFT_DEGPS:-0.0}"
JUMP_PERIOD_S="${JUMP_PERIOD_S:-0.0}"
JUMP_STD_M="${JUMP_STD_M:-0.0}"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

if ! rospack find "$PKG" >/dev/null 2>&1; then
  echo "[ERROR] 找不到 ROS 包 $PKG，请检查 R300_WS=$WS 是否正确。"
  exit 1
fi

cleanup() {
  echo "[INFO] 停止仿真相关节点..."
  rosnode kill /odom_to_path >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$ODOM_PATH" == "true" ]]; then
  rosrun "$PKG" odom_to_path.py _odom_topic:=/one_x/odom _path_topic:=/one_x/path _max_points:=5000 &
  sleep 0.5
fi

roslaunch "$PKG" subject1_dwa_sim.launch \
  use_rviz:="$RVIZ" \
  sim_max_v:="$SIM_MAX_V" \
  sim_max_w:="$SIM_MAX_W" \
  sim_acc_lim_v:="$SIM_ACC_V" \
  sim_acc_lim_w:="$SIM_ACC_W" \
  sim_tau_v:="$SIM_TAU_V" \
  sim_tau_w:="$SIM_TAU_W" \
  pos_noise_std:="$POS_NOISE" \
  yaw_noise_std_deg:="$YAW_NOISE_DEG" \
  drift_x_mps:="$DRIFT_X_MPS" \
  drift_y_mps:="$DRIFT_Y_MPS" \
  yaw_drift_degps:="$YAW_DRIFT_DEGPS" \
  jump_period_s:="$JUMP_PERIOD_S" \
  jump_std_m:="$JUMP_STD_M"
