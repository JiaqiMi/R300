cmake_minimum_required(VERSION 3.0.2)
project(r300_yolo_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
  r300_vision_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
  r300_vision_msgs
)

catkin_install_python(
  PROGRAMS
    scripts/yolo_depth_node.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    launch
    config
    models
  DESTINATION
    ${CATKIN_PACKAGE_SHARE_DESTINATION}
)