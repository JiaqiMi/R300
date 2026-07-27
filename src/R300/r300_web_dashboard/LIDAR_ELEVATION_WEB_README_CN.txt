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

v20 更新：高程框改为"车头朝上"视图
------------------------------------------------------------
背景：高程图（GridMap）是 odom 轴对齐的，网格只随车"位置"平移、不随车
"朝向"旋转。v19 前端把它按固定方向绘制且车辆箭头恒朝上，实际是
"odom 轴朝上"的视图却标注"前方 x+"——车原地转圈时画面零响应（人相对
车头的方位不更新），车绕障碍物转圈时因位置平移看起来又是对的。

改法：
- 适配节点经 TF 查询 odom->body（FAST-LIO 树）的 yaw，写入
  /r300_web/elevation_json 的 robot_x/robot_y/robot_yaw 字段（version: 2）。
  注意：不能改用 /one_x/odom 的朝向，那是 1X 惯导的另一棵 odom 树，
  两树 yaw 相差任意常量。
- 前端将整幅高程图绕图心旋转 yaw（圆形视窗，四角不越界），车辆箭头
  保持朝上=车头；图边缘的 "x+" 角标指示 odom +x 方向，供与 rviz 对照。
- TF 未就绪（启动前 3~5 秒）时降级为原"odom x+ 朝上"视图并明确标注，
  车辆只画位置点、不画方向箭头，避免误导。

预期行为：原地转圈时整幅地图绕车图标反向旋转（障碍物转到车头方向时
显示在图标上方）；绕障碍物转圈时与 v19 一致。圆形视窗半径=地图半宽
（12m 图约 6m），四角约 2.5m 对角数据不再显示。
