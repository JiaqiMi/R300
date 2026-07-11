#!/bin/bash
source /home/amov/amovcar/devel/setup.bash
#r300_rtabmap
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch mavros apm.launch; exec bash"' \
--tab -e 'bash -c "sleep 4;roslaunch r300_bringup r300_bringup_indoor.launch; exec bash"' \
--tab -e 'bash -c "sleep 5;roslaunch r300_function  r300_d435i.launch; exec bash"' \
--tab -e 'bash -c "sleep 6;roslaunch r300_function r300_odom_rtabmap.launch; exec bash"' \
