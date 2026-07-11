#!/bin/bash

#This is a map saver for cartographer slam
#Default save to /home/amov/amovcar/src/R300/r300_function/maps/
#And the default name is r300_map.map

#Note
    # If you want to save different maps, you need to CHANGE the name of map
    # Prevents previous maps from being overwritten.

rosservice call /finish_trajectory 0
rosservice call /write_state "{filename: '/home/amov/amovcar/src/R300/r300_function/maps/r300_map.pbstream'}"
rosrun cartographer_ros cartographer_pbstream_to_ros_map -map_filestem=/home/amov/amovcar/src/R300/r300_function/maps/r300_map -pbstream_filename=/home/amov/amovcar/src/R300/r300_function/maps/r300_map.pbstream -resolution=0.05
