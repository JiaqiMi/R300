#!/usr/bin/env bash
set -euo pipefail

# 一键停止当前导航/仿真相关节点。
source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi

NODES=(
  /move_base
  /waypoint_executor
  /one_x_serial_driver
  /scout_base_node
  /sim_r300_odom_node
  /subject1_blank_map
  /sim_map_to_odom
  /subject1_map_to_odom
  /subject1_base_to_livox
  /odom_to_path
  /rviz
)

for n in "${NODES[@]}"; do
  rosnode kill "$n" >/dev/null 2>&1 || true
done

echo "[INFO] 已尝试停止 R300 导航/仿真相关节点。"
