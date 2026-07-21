cmake_minimum_required(VERSION 3.0.2)
project(single_lidar_elevation)
find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/livox_imu_transform.py
  scripts/livox_odom_transform.py
  scripts/custom_msg_crop.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
