#r300_simulation_slam_cartographer
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 1;roslaunch r300_simulation r300_cartographer_slam_gazebo.launch; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_simulation r300_cartographer_slam_sim.launch; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_simulation r300_cartographer_slam_rviz.launch; exec bash"' \
--tab -e 'bash -c "sleep 3;roslaunch r300_function r300_teleop_key.launch; exec bash"' \
