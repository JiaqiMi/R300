cmake_minimum_required(VERSION 3.0.2)
project(r300_autonomous_parking)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  message_filters
  move_base_msgs
  nav_msgs
  r300_vision_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_geometry_msgs
  tf2_ros
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/autonomous_parking_node.py
    scripts/parking_perception_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  PROGRAMS
    scripts/start_autonomous_parking.sh
    scripts/stop_autonomous_parking.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
