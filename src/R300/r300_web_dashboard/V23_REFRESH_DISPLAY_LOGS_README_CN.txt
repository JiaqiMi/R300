R300 Web v23：刷新恢复、分区日志、点云/高程按需传输
========================================================

本版本基于已经修复以下问题的版本继续修改：
1. 停止 1X 调用 stop_1x.sh；
2. 纯实车/雷达导航后台使用 sudo 密码 1234 配置 CAN；
3. 点云固定米制显示范围，不再随每帧 bounds 忽大忽小；
4. 已有视觉/雷达两种避障导航入口保持不变。

本次修改
--------

一、雷达感知与 Web 显示拆分

“启动雷达感知 / 高程计算”只启动：
  MID-360 -> FAST-LIO -> GPU 高程图

它不会启动 r300_lidar_web_adapter，也不会持续向浏览器发送点云/高程 JSON。
雷达避障仍可使用：
  /elevation_mapping/elevation_map_raw
  /r300_lidar/obstacle_scan
  /move_base/local_costmap/costmap

需要查看时点击“显示点云 / 高程”，单独启动：
  /r300_lidar_web_adapter
  /r300_web/lidar_points_json
  /r300_web/elevation_json

关闭显示只停止 Web 适配器，不停止 MID-360、FAST-LIO、高程计算或雷达避障。

二、端口

8090：网页 HTTP
9090：ROSBridge WebSocket；costmap、路径、状态、点云 JSON、高程 JSON都走此端口
8080：web_video_server；仅视觉 MJPEG 图像走此端口

点云和高程图没有新增独立端口。

三、刷新恢复

页面刷新不会停止后台 ROS 节点。
浏览器会：
- 自动重新连接 rosbridge 9090；
- 自动重新订阅核心话题；
- 若“点云 / 高程显示”后台仍在运行，自动恢复两路显示话题订阅；
- 视觉 MJPEG 加入失败后自动重试；
- 页面从后台切回时检查并恢复连接。

四、负载优化

- 点云/高程显示默认不传输；
- 点云显示降为 3000 点、1 Hz；
- 高程显示降为 100×100、1 Hz；
- costmap 底图只在新 OccupancyGrid 到达时重新生成，并按原始网格缓存；
- odom、路径、scan 重绘复用缓存，不再重复计算整张 760×520 图像。

这些修改只影响 Web 显示与传输，不修改导航、避障、高程算法、TF、DWA参数或启动顺序。

五、日志位置

视觉卡片下方：相机 / YOLO / web_video_server 日志
代价地图下方：1X、纯实车、视觉避障、雷达避障日志
点云卡片下方：雷达感知、高程计算、Web显示适配器日志
航点控制区域只保留服务调用日志。

六、建议使用顺序

视觉避障：
  启动相机/视觉 -> 启动1X -> 启动视觉避障/代价地图

雷达避障：
  启动1X -> 启动雷达感知/高程计算 -> 启动雷达避障/代价地图

查看点云和高程时再点击：
  显示点云/高程

不查看时点击：
  关闭点云/高程显示

七、替换后编译

cd ~/r300_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg r300_web_dashboard
source ~/r300_ws/devel/setup.bash

roslaunch r300_web_dashboard r300_web_dashboard.launch

浏览器 Ctrl+F5 一次。
