cmake_minimum_required(VERSION 3.0.2)
project(r300_1x_navigation)

add_compile_options(-std=c++11 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  diagnostic_msgs
  geometry_msgs
  move_base_msgs
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  std_srvs
  tf
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS actionlib diagnostic_msgs geometry_msgs move_base_msgs nav_msgs roscpp sensor_msgs std_msgs std_srvs tf
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(one_x_serial_driver src/one_x_serial_driver.cpp)
target_link_libraries(one_x_serial_driver ${catkin_LIBRARIES})

add_executable(geodetic_goal_sender src/geodetic_goal_sender.cpp)
target_link_libraries(geodetic_goal_sender ${catkin_LIBRARIES})

add_executable(cmd_vel_guard src/cmd_vel_guard.cpp)
target_link_libraries(cmd_vel_guard ${catkin_LIBRARIES})

install(TARGETS one_x_serial_driver geodetic_goal_sender cmd_vel_guard
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config maps include/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

catkin_install_python(PROGRAMS
  scripts/waypoint_executor.py
  scripts/dwa_odom_adapter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)