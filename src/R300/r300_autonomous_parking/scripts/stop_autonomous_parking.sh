#!/usr/bin/env bash
set -u
source /opt/ros/noetic/setup.bash
[[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]] && source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
rosservice call /subject1/autonomous_parking/reset "{}" >/dev/null 2>&1 || true
rosnode kill /r300_autonomous_parking >/dev/null 2>&1 || true
