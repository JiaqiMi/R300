#!/bin/bash
source /home/amov/cartographer_ws/devel_isolated/cartographer_ros/setup.bash
source /home/amov/amovcar/devel/setup.bash
#r300_cartphrapher_slam
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_bringup r300_bringup.launch; exec bash"' \
--tab -e 'bash -c "sleep 4;roslaunch amovcar_sensor amovcar_lidar.launch; exec bash"' \
--tab -e 'bash -c "sleep 5;roslaunch r300_function r300_cartographer_slam.launch; exec bash"' \
--tab -e 'bash -c "sleep 6;roslaunch r300_function r300_cartographer_rviz.launch; exec bash"' \
