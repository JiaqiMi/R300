R300 Web Dashboard v15 INS-only-web patch

本包只包含：
- r300_web_dashboard/www/index.html
- r300_web_dashboard/www/app.js
- r300_web_dashboard/www/style.css

不包含 config.json，不修改端口/IP/视频参数，不修改 dashboard_server.py，不修改 one_x_serial_driver.cpp。

新增：
1. 惯导/GPS面板：读取 /one_x/gps_fix、/one_x/fix、/one_x/ins_status。
2. 显示惯导接收的 GPS 经纬高。
3. GPS首次有效后计时。
4. 显示 work_state/navigation_mode/position_reference/velocity_reference/INS有效/故障。
5. 复位按钮通过 ROSBridge 发布 std_msgs/String 到 /one_x/command_hex。

注意：复位按钮需要底层已有节点订阅 /one_x/command_hex 并处理该报文；本包不改串口驱动，所以不会破坏当前 Web 连接。
