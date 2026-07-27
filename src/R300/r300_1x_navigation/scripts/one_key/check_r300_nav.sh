#!/usr/bin/env bash
set -euo pipefail

# Check the navigation chain shared by pure and visual real-vehicle modes.

source /opt/ros/noetic/setup.bash
if [[ -f "${R300_WS:-$HOME/r300_ws}/devel/setup.bash" ]]; then
  source "${R300_WS:-$HOME/r300_ws}/devel/setup.bash"
fi

echo "========== ROS nodes =========="
rosnode list | sort || true

echo
echo "========== velocity command =========="
rostopic info /subject1/cmd_vel_raw || true

echo
echo "========== localization odometry =========="
rostopic info /one_x/odom || true

echo
echo "========== DWA feedback odometry =========="
rostopic info /subject1/dwa_odom || true
rosparam get /dwa_odom_adapter/input_odom_topic || true
rosparam get /dwa_odom_adapter/output_odom_topic || true
rosparam get /dwa_odom_adapter/max_yaw_rate_radps || true

echo
echo "========== move_base params =========="
rosparam get /move_base/controller_frequency || true
rosparam get /move_base/planner_frequency || true
rosparam get /move_base/DWAPlannerROS/odom_topic || true
rosparam get /move_base/DWAPlannerROS/max_vel_x || true
rosparam get /move_base/DWAPlannerROS/max_vel_trans || true
rosparam get /move_base/DWAPlannerROS/max_vel_theta || true
rosparam get /move_base/DWAPlannerROS/acc_lim_x || true
rosparam get /move_base/DWAPlannerROS/acc_lim_theta || true

echo
echo "========== waypoint services =========="
for service in \
  /subject1/start_waypoints \
  /subject1/pause_waypoints \
  /subject1/resume_waypoints \
  /subject1/skip_waypoint \
  /subject1/cancel_waypoints
do
  if rosservice list 2>/dev/null | grep -qx "$service"; then
    echo "[OK] $service"
  else
    echo "[MISS] $service"
  fi
done

echo
echo "========== waypoint status/progress =========="
rostopic echo -n 1 /subject1/waypoint_status || true
rosparam get /waypoint_executor/max_goal_distance_from_origin_m || true

echo
echo "========== local costmap =========="
rosparam get /move_base/local_costmap/global_frame || true
rosparam get /move_base/local_costmap/width || true
rosparam get /move_base/local_costmap/height || true
rosparam get /move_base/local_costmap/resolution || true
rosparam get /move_base/local_costmap/plugins || true

echo
echo "========== TF odom->base_link =========="
(timeout 5 rosrun tf tf_echo odom base_link) || true
