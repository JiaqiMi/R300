#!bin/bash
rosservice call /finish_trajectory 0
rosservice call /write_state "{filename: '/home/amov/amovcar/src/R300/r300_simulation/maps/map.pbstream'}"
rosrun cartographer_ros cartographer_pbstream_to_ros_map -map_filestem=/home/amov/amovcar/src/R300/r300_simulation/maps/map -pbstream_filename=/home/amov/amovcar/src/R300/r300_simulation/maps/map.pbstream -resolution=0.05