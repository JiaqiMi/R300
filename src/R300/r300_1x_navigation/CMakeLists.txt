cmake_minimum_required(VERSION 3.0.2)
project(r300_1x_navigation)

add_compile_options(-std=c++11 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  diagnostic_msgs
  geometry_msgs
  costmap_2d
  pluginlib
  move_base_msgs
  message_generation
  nav_msgs
  roscpp
  rospy
  r300_vision_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_geometry_msgs
  tf2_ros
  tf2
)

add_message_files(
  FILES
  InsImu.msg
  Attitude.msg
  Velocity.msg
  UpdateFlag.msg
  InsStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS actionlib costmap_2d diagnostic_msgs geometry_msgs message_runtime move_base_msgs nav_msgs pluginlib roscpp rospy r300_vision_msgs sensor_msgs std_msgs std_srvs tf tf2 tf2_geometry_msgs tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(one_x_serial_driver src/one_x_serial_driver.cpp)
add_dependencies(one_x_serial_driver ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(one_x_serial_driver ${catkin_LIBRARIES})

add_executable(geodetic_goal_sender src/geodetic_goal_sender.cpp)
target_link_libraries(geodetic_goal_sender ${catkin_LIBRARIES})

add_executable(cmd_vel_guard src/cmd_vel_guard.cpp)
target_link_libraries(cmd_vel_guard ${catkin_LIBRARIES})

add_library(r300_vision_snapshot_layer src/vision_snapshot_layer.cpp)
target_link_libraries(r300_vision_snapshot_layer ${catkin_LIBRARIES})
add_dependencies(r300_vision_snapshot_layer ${catkin_EXPORTED_TARGETS})

install(TARGETS one_x_serial_driver geodetic_goal_sender cmd_vel_guard
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS r300_vision_snapshot_layer
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES vision_snapshot_layer_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch config maps include/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

# catkin_install_python(PROGRAMS
#   scripts/waypoint_executor.py
#   scripts/dwa_odom_adapter.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# 仿真和工具+实际
catkin_install_python(PROGRAMS
  scripts/waypoint_executor.py
  scripts/dwa_odom_adapter.py
  scripts/odom_to_path.py
  scripts/sim_blank_map_node.py
  scripts/sim_r300_odom_node.py
  scripts/ins_trajectory_viewer.py
  scripts/ultrasonic_basic_node.py
  scripts/ultrasonic_filter_node.py
  scripts/ultrasonic_estop_node.py
  scripts/vision_obstacle_layer_node.py
  scripts/vision_clear_scan_node.py
  scripts/costmap_web_viewer.py
  scripts/dwa_web_tuner.py
  scripts/vision_costmap_scan_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/one_key/start_r300_vision_nav.sh
  scripts/one_key/start_dwa_web_tuner.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install(FILES VISUAL_AVOIDANCE_GUIDE.md
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
