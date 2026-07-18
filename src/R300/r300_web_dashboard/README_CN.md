# R300 Web 上位机

这个包用于把 R300 当前 ROS 系统中的定位、视觉、路径、局部代价地图、激光雷达和航点控制集中显示到浏览器中。

## 1. 功能

第一版包含：

- 实时视觉检测画面：`/r300_vision/annotated_image`
- 车辆状态：`/one_x/odom`、`/one_x/fix`、`/one_x/gps_fix`、`/one_x/heading_deg`
- 控制指令：`/subject1/cmd_vel_raw`
- 路径显示：`/move_base/NavfnROS/plan`、`/move_base/DWAPlannerROS/local_plan`
- 代价地图显示：`/move_base/local_costmap/costmap`
- 激光与视觉障碍俯视图：`/scan`、`/r300_vision/obstacle_scan`、`/r300_vision/active_obstacle_scan`
- 检测目标列表：`/r300_vision/detections`、`/r300_vision/target_point`
- 航点服务按钮：开始、暂停、恢复、跳过、取消

第一版不会直接发布 `/cmd_vel`，避免和 `move_base` 抢底盘控制。

## 2. 放到哪里

将整个文件夹放到：

```bash
~/r300_ws/src/R300/r300_web_dashboard
```

目录应类似：

```text
~/r300_ws/src/R300/r300_web_dashboard/
├── package.xml
├── CMakeLists.txt
├── launch/
├── scripts/
├── www/
└── config/
```

## 3. 安装依赖

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-rosbridge-server \
  ros-noetic-web-video-server \
  ros-noetic-tf2-web-republisher
```

## 4. 编译

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg r300_web_dashboard
source devel/setup.bash
```

也可以直接完整编译：

```bash
cd ~/r300_ws
catkin_make
source devel/setup.bash
```

## 5. 启动

先启动你原来的 R300 系统，例如：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_r300_vision_nav.sh --no-rviz
```

再启动上位机：

```bash
roslaunch r300_web_dashboard r300_web_dashboard.launch
```

浏览器打开：

```text
http://工控机IP:8090
```

查看工控机 IP：

```bash
hostname -I
```

例如：

```text
http://192.168.1.107:8090
```

## 6. 端口说明

默认端口：

```text
8090：上位机网页
9090：rosbridge websocket
8080：web_video_server 图像流
```

修改端口：

```bash
roslaunch r300_web_dashboard r300_web_dashboard.launch \
  dashboard_port:=8091 \
  rosbridge_port:=9091 \
  video_port:=8081
```

## 7. 修改话题

网页真正读取的话题配置在：

```text
r300_web_dashboard/www/config.json
```

例如视觉图像话题不是 `/r300_vision/annotated_image`，就改：

```json
"video": {
  "topic": "/r300_vision/annotated_image"
}
```

如果 costmap 话题不同，改：

```json
"costmap": "/move_base/local_costmap/costmap"
```

如果速度话题后续又加回 `cmd_vel_guard`，可以把：

```json
"cmd_vel": "/subject1/cmd_vel_raw"
```

改为：

```json
"cmd_vel": "/subject1/cmd_vel_safe"
```

## 8. 分步调试

### 8.1 检查 rosbridge

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

浏览器控制台没有 websocket 报错即可。

### 8.2 检查视频服务

```bash
rosrun web_video_server web_video_server _port:=8080 _address:=0.0.0.0
```

浏览器直接访问：

```text
http://工控机IP:8080/stream?topic=/r300_vision/annotated_image&type=mjpeg
```

### 8.3 检查关键话题

```bash
rostopic hz /one_x/odom
rostopic hz /subject1/cmd_vel_raw
rostopic hz /move_base/local_costmap/costmap
rostopic hz /r300_vision/annotated_image
rostopic hz /scan
```

## 9. 页面看不到数据的常见原因

1. `rosbridge_server` 没启动；
2. 浏览器访问的不是工控机 IP；
3. 话题名和 `www/config.json` 不一致；
4. 只启动了上位机，没有启动导航或视觉系统；
5. 防火墙或网络不通；
6. `/move_base/local_costmap/costmap` 发布频率很低，需要等几秒；
7. 图像页面黑屏，多数是 `web_video_server` 没启动或图像话题不存在。

## 10. 后续可以继续加的功能

- 动态目标状态：行人停车、跟车、超车、恢复
- 障碍物经纬度上报列表
- rosbag 开始/停止按钮
- 清除 costmap 按钮
- 一键启动多个 launch 的按钮
- 点云 `PointCloud2` 三维显示
- 登录密码与操作权限

## 11. 安全说明

第一版只做服务调用和状态显示，不直接发布底盘速度。实车测试时仍需要保留遥控器、急停和人工接管。

## v4 猕猴桃主题与网页启动节点

本版本将界面改为绿色猕猴桃主题，标题改为“别打了我是酱油”，并增加两个网页启动按钮：

- 启动相机/视觉：执行 `~/r300_ws/scripts/start_r300.sh web`
- 启动点云/代价地图：执行 `cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./start_r300_vision_nav.sh --no-rviz`

启动 Web：

```bash
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
roslaunch r300_web_dashboard r300_web_dashboard.launch
```

浏览器打开：

```text
http://100.106.189.126:8090
```

注意：本版本为了快速测试，在本地 Web 服务中处理了点云/导航脚本的 sudo 密码输入。该方式只适合实验室内网调试。长期使用建议改成 sudoers 中只对指定脚本配置 NOPASSWD。

## v5：网页按钮启动点云/代价地图

本版将网页按钮改为调用 wrapper 脚本：

- `scripts/web_start_camera.sh`：启动 `~/r300_ws/scripts/start_r300.sh web`
- `scripts/web_start_nav.sh`：启动 `start_r300_vision_nav.sh --no-rviz`

点云/导航脚本需要 sudo 时，`web_start_nav.sh` 会用密码 `1234` 做 `sudo -S -v`，并保持 sudo 缓存，避免后台无交互终端时启动失败。

按钮日志在：

```bash
~/.ros/r300_web_dashboard/web_start_camera.log
~/.ros/r300_web_dashboard/web_start_nav.log
```
