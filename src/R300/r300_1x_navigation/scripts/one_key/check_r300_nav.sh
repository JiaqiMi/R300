#!/usr/bin/env bash
set -euo pipefail

# 一键检查：当前导航链路、参数、话题订阅是否正常。

source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi

echo "========== ROS nodes =========="
rosnode list | sort || true

echo "\n========== /subject1/cmd_vel_raw =========="
rostopic info /subject1/cmd_vel_raw || true

echo "\n========== /one_x/odom =========="
rostopic info /one_x/odom || true

echo "\n========== /map =========="
rostopic info /map || true

echo "\n========== move_base params =========="
rosparam get /move_base/controller_frequency || true
rosparam get /move_base/planner_frequency || true
rosparam get /move_base/DWAPlannerROS/odom_topic || true
rosparam get /move_base/DWAPlannerROS/max_vel_x || true
rosparam get /move_base/DWAPlannerROS/max_vel_trans || true
rosparam get /move_base/DWAPlannerROS/max_vel_theta || true
rosparam get /move_base/DWAPlannerROS/acc_lim_x || true
rosparam get /move_base/DWAPlannerROS/acc_lim_theta || true

echo "\n========== TF odom->base_link =========="
(timeout 5 rosrun tf tf_echo odom base_link) || true
