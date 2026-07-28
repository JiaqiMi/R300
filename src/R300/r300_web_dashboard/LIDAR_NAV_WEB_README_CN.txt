本版本以“已修复三个问题”的 Web 包为基线，只新增雷达高程图避障导航接入。

保留且确认存在的三个修复：
1. 停止 1X 调用 stop_1x.sh；
2. 纯实车导航后台使用 sudo 密码 1234，停止调用 stop_r300_nav.sh；
3. 点云固定 ±8m 显示范围，不随每帧 bounds 跳大跳小。

新增内容：
- ②C“雷达避障 / 代价地图”启动、停止按钮；
- 启动 subject1_waypoint_lidar_nav.launch；
- 启动前检查 1X、高程图、已有 move_base，并设置当前 1X ENU 原点；
- 后台配置 can0，默认 sudo 密码由 R300_SUDO_PASS 提供，未设置时使用 1234；
- 纯实车、视觉避障、雷达避障三模式互斥；
- 订阅 /r300_lidar/obstacle_scan 与 /r300_lidar/active_obstacle_scan；
- 在现有 costmap 画布叠加雷达本帧障碍与 TTL 活动障碍；
- 原 /move_base/local_costmap/costmap、DWA 路径、点云、高程图和目标回传逻辑不变。

雷达模式顺序：
1. 启动 1X；
2. 启动雷达 / 点云 / 高程；
3. 等待高程图出现；
4. 启动雷达避障 / 代价地图；
5. 点击开始航点。
