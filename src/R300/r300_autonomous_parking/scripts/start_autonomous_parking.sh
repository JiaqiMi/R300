#!/usr/bin/env bash
set -euo pipefail

WS="${R300_WS:-$HOME/r300_ws}"
MODEL1_PATH="${PARKING_MODEL1_PATH:-}"
MODEL2_PATH="${PARKING_MODEL2_PATH:-}"
AUTO_START="${PARKING_AUTO_START_AFTER_WAYPOINTS:-false}"

if [[ -f "$HOME/venvs/yolo26/bin/activate" ]]; then
  # parking_manager启动的视觉子进程会继承该Python环境。
  source "$HOME/venvs/yolo26/bin/activate"
fi
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

ARGS=("auto_start_after_waypoints:=$AUTO_START")
[[ -n "$MODEL1_PATH" ]] && ARGS+=("model1_path:=$MODEL1_PATH")
[[ -n "$MODEL2_PATH" ]] && ARGS+=("model2_path:=$MODEL2_PATH")

exec roslaunch r300_autonomous_parking autonomous_parking.launch "${ARGS[@]}" "$@"
