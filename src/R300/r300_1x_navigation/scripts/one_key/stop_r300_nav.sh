#!/usr/bin/env bash
set -Eeuo pipefail

# Stop navigation-related nodes while keeping the standalone 1X parser alive.
# Use --with-1x only when the user also wants to stop the serial parser.

STOP_1X=false
if [[ "${1:-}" == "--with-1x" ]]; then
  STOP_1X=true
elif [[ $# -gt 0 ]]; then
  echo "用法：$0 [--with-1x]" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi

NODES=(
  /move_base
  /waypoint_executor
  /dwa_odom_adapter
  /vision_obstacle_layer_node
  /vision_costmap_scan_node
  /scout_base_node
  /robot_state_publisher
  /sim_r300_odom_node
  /subject1_blank_map
  /sim_map_to_odom
  /subject1_map_to_odom
  /subject1_base_to_livox
  /odom_to_path
  /rviz
)

if [[ "$STOP_1X" == "true" ]]; then
  NODES+=(/one_x_serial_driver)
fi

for n in "${NODES[@]}"; do
  rosnode kill "$n" >/dev/null 2>&1 || true
done

if [[ "$STOP_1X" == "true" ]]; then
  echo "[INFO] 已停止导航节点，并请求停止1X解析。"
else
  echo "[INFO] 已停止导航节点；独立1X解析保持运行。"
  echo "[INFO] 如需同时停止1X：./stop_r300_nav.sh --with-1x"
fi
