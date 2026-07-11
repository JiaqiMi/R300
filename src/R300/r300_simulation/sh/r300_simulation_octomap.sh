#r300_simulation_octomap
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 1;roslaunch r300_simulation r300_3dlidar_simulation.launch; exec bash"' \
--tab -e 'bash -c "sleep 2;roslaunch r300_function r300_teleop_key.launch; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_function r300_octomap.launch; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_function r300_octomap_rviz.launch; exec bash"' \