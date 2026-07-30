#!/usr/bin/env bash
set -euo pipefail

WS="${R300_WS:-$HOME/r300_ws}"
MODEL1_PATH="${PARKING_MODEL1_PATH:-}"
MODEL2_PATH="${PARKING_MODEL2_PATH:-}"

# 若调用终端已经激活 conda yolo26，则保持当前环境。
# 兼容原先通过 ~/venvs/yolo26 启动的部署方式。
if [[ -f "$HOME/venvs/yolo26/bin/activate" ]]; then
  source "$HOME/venvs/yolo26/bin/activate"
fi

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

# ROS Noetic 系统 Python 包（如 PyKDL）在虚拟环境中仍然可见。
export PYTHONPATH="/usr/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$WS/devel/lib/python3/dist-packages:${PYTHONPATH:-}"

ARGS=()
[[ -n "$MODEL1_PATH" ]] && ARGS+=("model1_path:=$MODEL1_PATH")
[[ -n "$MODEL2_PATH" ]] && ARGS+=("model2_path:=$MODEL2_PATH")

exec roslaunch r300_autonomous_parking autonomous_parking.launch "${ARGS[@]}" "$@"
