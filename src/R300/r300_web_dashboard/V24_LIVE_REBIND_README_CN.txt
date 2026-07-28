R300 Web v24：只修复页面动态恢复，不改变任何启动/停止逻辑。

修改范围仅 www/app.js：
1. 点击“启动相机/视觉”后，自动重复请求 8080 MJPEG，直到画面出现，无需刷新页面。
2. 点击纯实车/视觉避障/雷达避障后，自动重新绑定 costmap、路径和障碍话题；
   当导航后台运行但 5 秒未收到 costmap 时，仅重新订阅 rosbridge，不重启 ROS 节点。
3. 点云/高程显示订阅保持独立，启动相机不会主动关闭、停止或重新启动雷达显示。

重要：若导航日志出现“高程图输入超时，已停发 scan”，说明
/elevation_mapping/elevation_map_raw 本身停止发布。这不是网页刷新或 rosbridge 问题，
常见原因是 Jetson 上双 YOLO 与 elevation_mapping_cupy 同时占用 GPU 后进程异常或饥饿。
本版本不会擅自重启高程节点，也不会修改 YOLO/高程算法参数。
