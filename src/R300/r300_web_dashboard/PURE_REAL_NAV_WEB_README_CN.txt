R300 Web：纯实车导航按钮说明
================================

本版本只修改 r300_web_dashboard，不修改 r300_1x_navigation 内任何程序。

网页启动逻辑：
1. ① 启动 1X 惯导
   cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
   ./start_1x.sh

2. 1X 数据稳定后，在以下两种导航模式中二选一：

   ②A 纯实车导航（不接入视觉障碍）
   ./start_real_nav.sh --no-rviz

   ②B 视觉避障导航 / 代价地图
   ./start_r300_vision_nav.sh --no-rviz

纯实车导航按钮不会自动执行航点。待页面状态显示“纯实车导航：运行中”且日志显示全链路就绪后，
在网页“航点控制”区域点击“开始航点”，等价于：
   rosservice call /subject1/start_waypoints "{}"

安全限制：
- ②A 与 ②B 不能同时运行，因为两者都会启动 move_base、底盘控制和航点执行器。
- Web 会在其中一个模式运行时拒绝启动另一个模式。
- 停止 ②A 或 ②B 不会停止独立运行的 1X。
