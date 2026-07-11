#!/bin/bash
source /home/amov/cartographer_ws/devel_isolated/cartographer_ros/setup.bash
source /home/amov/amovcar/devel/setup.bash
#r300_navigation_indoor
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch mavros apm.launch; exec bash"' \
--tab -e 'bash -c "sleep 4;roslaunch r300_bringup r300_bringup_indoor.launch; exec bash"' \
--tab -e 'bash -c "sleep 7;roslaunch amovcar_sensor amovcar_lidar.launch; exec bash"' \
--tab -e 'bash -c "sleep 8;roslaunch r300_function r300_navigation_indoor.launch; exec bash"' \


