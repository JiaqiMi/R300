#!/bin/bash
source /home/amov/cartographer_ws/devel_isolated/cartographer_ros/setup.bash
source /home/amov/amovcar/devel/setup.bash
#r300_rtabmap
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 4;roslaunch amovcar_sensor amovcar_t265.launch; exec bash"' \
--tab -e 'bash -c "sleep 5;roslaunch amovcar_sensor amovcar_d435i.launch; exec bash"' \
--tab -e 'bash -c "sleep 6;roslaunch r300_function r300_rtabmap.launch; exec bash"' \
