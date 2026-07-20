R300 Web Dashboard UI v9 - 两行总览版

改动：
1. 第一行：摄像头 / local_costmap+DWA / 卫星轨迹图。
2. 第二行：雷达俯视图 / 车辆状态 / 航点控制日志 / 视觉检测信息日志。
3. 宽屏下总共两行，方便外场同时观察主要信息。
4. 继续放大节点启动日志和视觉检测日志显示高度。

使用：
cd ~/r300_ws
unzip -o r300_web_dashboard_ui_v9_2rows.zip -d ~/r300_ws
~/r300_ws/src/R300/r300_web_dashboard/scripts/apply_web_ui_patch.sh

重启：
pkill -f dashboard_server
pkill -f rosbridge_websocket
cd ~/r300_ws
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
roslaunch r300_web_dashboard r300_web_dashboard.launch

浏览器 Ctrl+F5 强制刷新。
