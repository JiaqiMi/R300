cmake_minimum_required(VERSION 3.0.2)
project(r300_web_dashboard)

find_package(catkin REQUIRED COMPONENTS rospy)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/dashboard_server.py
  scripts/lidar_web_adapter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(PROGRAMS
  scripts/web_start_lidar_nav.sh
  scripts/web_stop_lidar_nav.sh
  scripts/web_start_lidar_elevation.sh
  scripts/web_start_lidar_display.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch www config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
