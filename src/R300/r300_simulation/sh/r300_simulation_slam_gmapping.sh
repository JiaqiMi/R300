#r300_simulation_slam_gmapping
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 1;roslaunch r300_simulation r300_2dlidar_simulation.launch; exec bash"' \
--tab -e 'bash -c "sleep 2;roslaunch r300_function r300_teleop_key.launch; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_function r300_gmapping.launch; exec bash"' \
--tab -e 'bash -c "sleep 4;roslaunch r300_function r300_slam_rviz.launch; exec bash"' \
