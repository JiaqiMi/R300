#!/usr/bin/env bash
set -euo pipefail

# 一键录包：记录调参和实车诊断核心话题。
# 用法：
#   ./record_r300_bag.sh
#   OUT=~/bags/test1 ./record_r300_bag.sh

source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi

mkdir -p "$HOME/bags"
OUT="${OUT:-$HOME/bags/r300_$(date +%Y%m%d_%H%M%S)}"

echo "[INFO] rosbag 输出：$OUT.bag"
rosbag record -O "$OUT" \
  /subject1/cmd_vel_raw \
  /one_x/odom \
  /one_x/path \
  /one_x/fix \
  /one_x/ins_fix \
  /one_x/gps_fix \
  /one_x/heading_deg \
  /one_x/pos_compare \
  /move_base/NavfnROS/plan \
  /move_base/DWAPlannerROS/local_plan \
  /move_base/current_goal \
  /tf \
  /tf_static
