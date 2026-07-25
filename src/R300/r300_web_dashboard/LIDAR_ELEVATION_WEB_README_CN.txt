R300 Web v19：MID-360 点云 + GPU 高程图显示
============================================================

保持不变：
1. 相机/视觉启动逻辑不变。
2. ①启动导航（1X）和②启动代价地图仍然完全分开。
3. 目标类型、相机三维坐标、目标经纬度回传与 CSV 记录保持不变。
4. 不修改 single_lidar_elevation、FAST-LIO、elevation_mapping_cupy 或导航程序。

新增：
- 左侧大框：/cloud_registered_body 配准点云，最多 6000 点、2 Hz Web 显示。
- 右侧大框：/elevation_mapping/elevation_map_raw 的 elevation 层，最多 150×150、1 Hz Web 显示。
- 两个框与视频框规格相近，点云可拖拽旋转/滚轮缩放，高程图可拖拽平移/滚轮缩放。
- “启动雷达 / 点云 / 高程”按钮一次共同启动：
    roslaunch single_lidar_elevation single_lidar_elevation.launch rviz:=false
  同时启动 Web 轻量显示适配节点。
- “停止雷达 / 点云 / 高程”一次共同停止上述整组进程。

Web 轻量话题：
- /r300_web/lidar_points_json   std_msgs/String
- /r300_web/elevation_json      std_msgs/String

原始感知话题未修改：
- /cloud_registered
- /cloud_registered_body
- /Odometry
- /elevation_mapping/elevation_map_raw

启动前提：
- single_lidar_elevation 已放入当前 catkin 工作空间并完成编译。
- rospack find single_lidar_elevation 能找到该包。
- MID-360 网口和 IP 已按原模块 README 配置。
- GPU/CuPy/PyTorch 等依赖已按原模块 README 安装。
