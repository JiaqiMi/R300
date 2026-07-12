#!/usr/bin/env bash
set -euo pipefail

# 一键发送 base_link 坐标系下的目标点。
# 用法：
#   ./send_goal_base.sh forward 10
#   ./send_goal_base.sh back 5
#   ./send_goal_base.sh left  3
#   ./send_goal_base.sh right 3
#   ./send_goal_base.sh xy 10 -2

source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi

MODE="${1:-forward}"
A="${2:-10}"
B="${3:-0}"

case "$MODE" in
  forward)
    X="$A"; Y="0.0" ;;
  back|backward)
    X="-$A"; Y="0.0" ;;
  left)
    X="0.0"; Y="$A" ;;
  right)
    X="0.0"; Y="-$A" ;;
  xy)
    X="$A"; Y="$B" ;;
  *)
    echo "用法：$0 forward 10 | back 5 | left 3 | right 3 | xy 10 -2"
    exit 1 ;;
esac

echo "[INFO] 发送 base_link 目标：x=$X, y=$Y"
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
"{header: {frame_id: 'base_link'}, pose: {position: {x: $X, y: $Y, z: 0.0}, orientation: {w: 1.0}}}"
