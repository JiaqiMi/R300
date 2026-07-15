cmake_minimum_required(VERSION 3.0.2)
project(r300_web_dashboard)

find_package(catkin REQUIRED COMPONENTS rospy)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/dashboard_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch www config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
