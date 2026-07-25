R300 Web 目标类型与经纬度回传（v18）

本版以 v17 导航/代价地图分步启动版为基础，仅增加 Web 目标回传和记录功能。
导航、代价地图、视觉检测节点、惯导节点及其启动脚本逻辑均未修改。

数据来源：
1. /r300_vision/detections：目标类型、置信度、相机光学坐标。
2. /r300_vision/target_point：当前选中目标三维点。
3. /one_x/fix 或 /one_x/gps_fix：车辆经纬度。
4. /one_x/heading_deg：车辆航向（北0°、东90°）；无该话题时可由 /one_x/odom yaw 换算兜底。

相机坐标约定：x向右、y向下、z向前。Web 端将前向/右向距离按车辆航向换算成北向/东向偏移，再用 WGS84 局部曲率估算目标经纬度。

网页显示：
- 每个识别目标的类型、置信度、相机三维坐标、估算经纬度。
- 当前 /r300_vision/target_point 对应的目标类型和经纬度。
- 可下载浏览器 CSV。
- 可开始/停止工控机本地 CSV 记录。

本地保存目录：
~/.ros/r300_web_dashboard/targets/

注意：目标经纬度是融合相机深度、车辆经纬度和航向得到的估算值。误差受相机安装偏移、目标深度误差、时间同步、惯导航向与GNSS误差影响。相机不在车辆定位中心时，可编辑 www/config.json 中 target_geolocation 的 camera_forward_offset_m 和 camera_right_offset_m。
