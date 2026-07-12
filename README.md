# Self-Driving Car
---

## 项目介绍


无人车挑战赛

---

## 系统介绍

### 系统架构

pass

---

### 2. 运行环境

- Ubuntu 20.04
- ROS Noetic
- Python 3.8
- NVIDIA Jetson Orin
- CUDA 11.4
- Intel RealSense D435i
- PyTorch CUDA
- Ultralytics YOLO

---

## 惯性导航与控制系统

惯性导航与控制系统部分，用于在R300无人车上接入自研1X惯导/GPS，替代原飞控位姿输入，并基于ROS `move_base + DWA`实现定点导航、多航点导航和仿真调参。

系统当前采用纯 DWA 控制链路，`move_base` 输出的速度指令直接发送到底盘驱动节点，不再使用预对准、转头接管、`cmd_vel_guard` 或 `dwa_odom_adapter` 等中间控制模块。

---

### 1.系统架构

整体导航与控制链路如下：

```text
1X INS/GPS
    ↓
one_x_serial_driver
    ↓
/one_x/odom  +  odom → base_link
    ↓
move_base + DWA
    ↓
/subject1/cmd_vel_raw
    ↓
scout_base_node
    ↓
R300 底盘
```

仿真调参链路如下：

```text
sim_r300_odom_node
    ↓
/one_x/odom  +  odom → base_link
    ↓
move_base + DWA
    ↓
/subject1/cmd_vel_raw
    ↓
sim_r300_odom_node
```

主要功能包括：

- 解析 1X 惯导 110 字节串口数据；
- 发布 `/one_x/odom` 和 `odom → base_link` TF；
- 发布 INS 经纬度、GPS 经纬度、航向角和位置对比信息；
- 使用 GPS/INS 位置信息建立局部导航坐标；
- 将经纬度航点转换为局部 ENU 目标点；
- 使用 `move_base + DWA` 生成底盘速度指令；
- 支持多航点顺序执行；
- 支持 RViz 空白地图下的 DWA 闭环仿真；
- 支持一键启动、链路检查、目标点测试和 rosbag 数据记录。

---

### 2.项目结构

```text
r300_ws/
├── src/
│   └── R300/
│       └── r300_1x_navigation/
│           ├── config/
│           │   ├── subject1_dwa.yaml
│           │   ├── subject1_move_base.yaml
│           │   ├── subject1_waypoints.yaml
│           │   ├── subject1_costmap_common.yaml
│           │   ├── subject1_global_costmap.yaml
│           │   └── subject1_local_costmap.yaml
│           │
│           ├── launch/
│           │   ├── subject1_waypoint_nav.launch
│           │   ├── subject1_move_base.launch
│           │   ├── subject1_dwa_sim.launch
│           │   └── one_x_localization_only.launch
│           │
│           ├── scripts/
│           │   ├── waypoint_executor.py
│           │   ├── sim_r300_odom_node.py
│           │   ├── sim_blank_map_node.py
│           │   ├── odom_to_path.py
│           │   └── one_key/
│           │       ├── start_real_nav.sh
│           │       ├── start_sim_dwa.sh
│           │       ├── start_localization_only.sh
│           │       ├── send_goal_base.sh
│           │       ├── check_r300_nav.sh
│           │       ├── record_r300_bag.sh
│           │       ├── stop_r300_nav.sh
│           │       └── fix_permissions.sh
│           │
│           ├── src/
│           │   └── one_x_serial_driver.cpp
│           │
│           ├── maps/
│           │   ├── subject1_blank_map.yaml
│           │   └── subject1_blank_map.pgm
│           │
│           ├── CMakeLists.txt
│           └── package.xml
```

---

### 3.主要 ROS 话题

#### 惯导与定位相关话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/one_x/odom` | `nav_msgs/Odometry` | 导航使用的里程计 |
| `/one_x/fix` | `sensor_msgs/NavSatFix` | 当前导航位置 |
| `/one_x/ins_fix` | `sensor_msgs/NavSatFix` | INS 经纬度 |
| `/one_x/gps_fix` | `sensor_msgs/NavSatFix` | GPS 经纬度 |
| `/one_x/heading_deg` | `std_msgs/Float64` | 惯导航向角 |
| `/one_x/pos_compare` | `std_msgs/String` | INS/GPS 位置对比 |
| `/one_x/path` | `nav_msgs/Path` | RViz 轨迹显示 |

#### 控制与规划相关话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/subject1/cmd_vel_raw` | `geometry_msgs/Twist` | DWA 输出速度指令 |
| `/move_base/NavfnROS/plan` | `nav_msgs/Path` | 全局路径 |
| `/move_base/DWAPlannerROS/local_plan` | `nav_msgs/Path` | 局部路径 |
| `/move_base/current_goal` | `geometry_msgs/PoseStamped` | 当前导航目标 |

---

### 4.编译与环境加载

```bash
cd ~/r300_ws
catkin_make
source devel/setup.bash
```

建议加入 `~/.bashrc`：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

---

### 5.常用启动指令

#### 实车导航

```bash
roslaunch r300_1x_navigation subject1_waypoint_nav.launch
```

指定惯导串口和波特率：

```bash
roslaunch r300_1x_navigation subject1_waypoint_nav.launch \
  ins_serial_port:=/dev/ttyACM0 \
  ins_baudrate:=460800 \
  auto_start:=false
```

手动开始航点任务：

```bash
rosservice call /subject1/start_waypoints
```

取消航点任务：

```bash
rosservice call /subject1/cancel_waypoints
```

暂停航点任务：

```bash
rosservice call /subject1/pause_waypoints
```

恢复航点任务：

```bash
rosservice call /subject1/resume_waypoints
```

跳过当前航点：

```bash
rosservice call /subject1/skip_waypoint
```

---

#### 仅启动惯导定位

```bash
roslaunch r300_1x_navigation one_x_localization_only.launch
```

该模式只发布惯导/GPS 数据和 TF，不启动 `move_base`，也不控制车辆。

---

#### DWA 仿真调参

```bash
roslaunch r300_1x_navigation subject1_dwa_sim.launch
```

发送车体正前方 10 m 目标：

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
"{header: {frame_id: 'base_link'}, pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

发送车体后方 5 m 目标：

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
"{header: {frame_id: 'base_link'}, pose: {position: {x: -5.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

---

### 6.航点配置

航点文件：

```text
r300_1x_navigation/config/subject1_waypoints.yaml
```

格式示例：

```yaml
subject1_waypoints:
  waypoints:
    - name: wp_01
      latitude_deg: 38.98663491
      longitude_deg: 117.3418414
      altitude_m: 21.741

    - name: wp_02
      latitude_deg: 38.9866441
      longitude_deg: 117.3419243
      altitude_m: 21.741
```

注意：同一个 `subject1_waypoints` 下只能有一个 `waypoints:` 列表，不能重复写多个 `waypoints:`，否则 YAML 会发生覆盖，只读取最后一组内容。

---

### 7.一键启动脚本

一键脚本位于：

```text
r300_1x_navigation/scripts/one_key/
```

首次使用前增加执行权限：

```bash
chmod +x ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/*.sh
```

如果工作空间路径不是 `~/r300_ws`，可以通过环境变量指定：

```bash
export R300_WS=~/r300_ws
```

---

#### 实车导航一键启动

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_real_nav.sh
```

功能：

```text
启动 1X 惯导串口驱动
启动 move_base + DWA
启动 waypoint_executor 多航点节点
启动 scout_base_node 底盘驱动
建立 /one_x/odom → move_base → /subject1/cmd_vel_raw → scout_base_node 控制链路
```

常用参数：

```bash
INS_PORT=/dev/ttyACM0 \
INS_BAUDRATE=460800 \
AUTO_START=false \
LAUNCH_RVIZ=true \
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_real_nav.sh
```

实车测试时建议 `AUTO_START=false`，确认链路正常后再手动开始航点：

```bash
rosservice call /subject1/start_waypoints
```

---

#### DWA 仿真一键启动

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_sim_dwa.sh
```

功能：

```text
启动空白地图
启动虚拟 R300 里程计节点
启动 move_base + DWA
发布 /one_x/odom 和 odom → base_link
启动 /one_x/path 轨迹显示
用于无实车、无惯导条件下调试 DWA 参数
```

常用参数：

```bash
RVIZ=true \
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_sim_dwa.sh
```

模拟定位漂移和噪声：

```bash
DRIFT_Y_MPS=0.03 \
YAW_NOISE_DEG=0.5 \
JUMP_PERIOD_S=3.0 \
JUMP_STD_M=0.2 \
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_sim_dwa.sh
```

---

#### 仅惯导定位一键启动

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_localization_only.sh
```

功能：

```text
只启动 1X 惯导串口驱动
发布 /one_x/odom、/one_x/fix、/one_x/ins_fix、/one_x/gps_fix、/one_x/heading_deg
发布 odom → base_link TF
不启动 move_base
不启动 scout_base_node
不控制车辆
```

指定串口：

```bash
INS_PORT=/dev/ttyACM0 \
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/start_localization_only.sh
```

---

#### 发送测试目标

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/send_goal_base.sh forward 10
```

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/send_goal_base.sh back 5
```

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/send_goal_base.sh left 3
```

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/send_goal_base.sh right 3
```

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/send_goal_base.sh xy 10 -2
```

该脚本向 `/move_base_simple/goal` 发布 `base_link` 坐标系下的目标点，适合 DWA 参数调试。

---

#### 链路检查脚本

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/check_r300_nav.sh
```

该脚本用于检查：

```text
ROS 节点状态
/subject1/cmd_vel_raw 发布者和订阅者
/one_x/odom 发布者和订阅者
move_base 关键参数
odom → base_link TF
```

实车正常时：

```text
/subject1/cmd_vel_raw:
  Publisher: /move_base
  Subscriber: /scout_base_node
```

仿真正常时：

```text
/subject1/cmd_vel_raw:
  Publisher: /move_base
  Subscriber: /sim_r300_odom_node
```

---

#### rosbag 一键记录

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/record_r300_bag.sh
```

功能：

```text
记录 DWA 控制指令
记录惯导/GPS 定位结果
记录全局路径和局部路径
记录 TF
记录航点目标和轨迹
```

指定输出文件名前缀：

```bash
OUT=~/bags/r300_dwa_test_01 \
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/record_r300_bag.sh
```

---

#### 停止导航相关节点

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/stop_r300_nav.sh
```

功能：

```text
停止 move_base
停止 waypoint_executor
停止 one_x_serial_driver
停止 scout_base_node
停止仿真节点
停止 RViz
清理上一轮测试残留节点
```

---

#### 权限修复脚本

```bash
~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/fix_permissions.sh
```

功能：

```text
为 scripts/ 下的 Python 脚本和 one_key/ 下的 Shell 脚本添加执行权限
避免 roslaunch 或 bash 启动时报 Permission denied
```

---

### 链路检查指令

检查速度指令链路：

```bash
rostopic info /subject1/cmd_vel_raw
```

检查定位输入：

```bash
rostopic info /one_x/odom
```

检查 TF：

```bash
rosrun tf tf_echo odom base_link
```

检查 DWA 参数：

```bash
rosparam get /move_base/DWAPlannerROS/odom_topic
rosparam get /move_base/DWAPlannerROS/max_vel_x
rosparam get /move_base/DWAPlannerROS/max_vel_theta
```

---

### rosbag 推荐记录话题

```bash
rosbag record -O ~/r300_nav_test.bag \
  /subject1/cmd_vel_raw \
  /one_x/odom \
  /one_x/path \
  /one_x/fix \
  /one_x/ins_fix \
  /one_x/gps_fix \
  /one_x/heading_deg \
  /one_x/pos_compare \
  /move_base/NavfnROS/plan \
  /move_base/DWAPlannerROS/local_plan \
  /move_base/current_goal \
  /tf \
  /tf_static
```

---

### 注意事项

- `scout_base_node` 的 `odom_pub` 应保持 `false`，避免和 1X 惯导发布的 `odom → base_link` 冲突；
- 实车控制链路中 `/subject1/cmd_vel_raw` 由 `move_base` 发布，并直接发送给 `scout_base_node`；
- 修改 DWA YAML 后需要重启 `move_base`，或者使用 `rqt_reconfigure` 实时调参；
- 室内纯惯性位置会漂移，适合做 DWA 定性仿真，不适合作为高速闭环定位来源；
- 目标在车身后方且不允许倒车时，纯 DWA 可能选择原地转向或小弧线前进，这是 DWA 采样规划的正常特性；
- 如果要求必须“先原地转正再前进”，需要额外增加一次性预转向逻辑；
- 高速实车测试必须逐级提速，不建议直接使用高速度参数。


## 控制系统

pass

--

## Vision System

基于 **ROS Noetic、Intel RealSense D435i、Ultralytics YOLO 和 NVIDIA Jetson GPU** 实现的目标检测与深度定位系统。

系统支持以下功能：

- RealSense D435i 彩色图像与对齐深度图采集
- YOLO 模型 GPU 推理
- 目标二维检测框发布
- 目标三维位置估计
- 检测结果可视化
- ROS bag 数据录制与回放
- Web 浏览器实时查看检测画面
- 相机、模型、Web 和 rosbag 一键启动

---

### 1. 项目结构

```text
r300_ws/
├── src/
│   └── R300_vision/
│       ├── r300_vision_msgs/
│       │   └── msg/
│       │       ├── DetectedObject.msg
│       │       └── DetectedObjectArray.msg
│       │
│       └── r300_yolo_detector/
│           ├── config/
│           ├── launch/
│           ├── models/
│           ├── scripts/
│           ├── CMakeLists.txt
│           └── package.xml
│
├── scripts/
│   └── start_r300.sh
│
├── build/
├── devel/
└── README.md
```

---



### 3. 项目编译

进入工作空间：

```bash
cd ~/r300_ws
```

清理旧的编译结果：

```bash
rm -rf build devel
```

加载 ROS Noetic 环境：

```bash
source /opt/ros/noetic/setup.bash
```

完整编译工作空间，并保存编译日志：

```bash
catkin_make 2>&1 | tee build_log.txt
```

加载当前工作空间：

```bash
source ~/r300_ws/devel/setup.bash
```

> 建议将下面两行添加到 `~/.bashrc`，避免每次打开终端后重复执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

---

### 4. 独立编译视觉功能包

#### 4.1 编译消息包

```bash
cd ~/r300_ws
catkin_make --pkg r300_vision_msgs
```

#### 4.2 编译目标检测包

```bash
cd ~/r300_ws
catkin_make --pkg r300_yolo_detector
```

编译完成后重新加载工作空间：

```bash
source ~/r300_ws/devel/setup.bash
```

---

### 5. 分步启动系统

分步启动适合调试和排查问题。

#### 5.1 启动 RealSense D435i

打开第一个终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

启动相机，并开启深度对齐与时间同步：

```bash
roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  enable_sync:=true
```

检查相机话题：

```bash
rostopic list | grep camera
```

检查彩色图像频率：

```bash
rostopic hz /camera/color/image_raw
```

检查对齐深度图频率：

```bash
rostopic hz /camera/aligned_depth_to_color/image_raw
```

---

#### 5.2 启动目标检测与深度定位

打开第二个终端：

```bash
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

启动 YOLO 检测节点：

```bash
roslaunch r300_yolo_detector yolo_depth.launch
```

正常启动后应看到类似日志：

```text
CUDA=True
GPU: Orin
Model classes: ...
R300 YOLO depth node started
```

---

#### 5.3 查看检测结果

打开第三个终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rqt_image_view
```

在图像话题列表中选择：

```text
/r300_vision/annotated_image
```

---

### 6. 主要 ROS 话题

#### 6.1 订阅话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | D435i 彩色图像 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 对齐到彩色图的深度图 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 彩色相机内参 |

#### 6.2 发布话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/r300_vision/annotated_image` | `sensor_msgs/Image` | 带检测框和距离信息的图像 |
| `/r300_vision/detections` | `r300_vision_msgs/DetectedObjectArray` | 所有目标检测和位置结果 |
| `/r300_vision/target_point` | `geometry_msgs/PointStamped` | 选中目标的三维位置 |

查看检测结果：

```bash
rostopic echo /r300_vision/detections
```

查看选中目标位置：

```bash
rostopic echo /r300_vision/target_point
```

检查标注图像发布频率：

```bash
rostopic hz /r300_vision/annotated_image
```

---

### 7. 录制检测图像和检测结果

#### 7.1 创建记录目录

```bash
mkdir -p ~/r300_records
cd ~/r300_records
```

#### 7.2 开始录制 rosbag

```bash
rosbag record \
  -O yolo_test_$(date +%Y%m%d_%H%M%S).bag \
  /r300_vision/annotated_image \
  /r300_vision/detections \
  /r300_vision/target_point
```

录制结束时按：

```text
Ctrl + C
```

生成的文件示例：

```text
yolo_test_20260711_183000.bag
```

---

#### 7.3 查看 rosbag 信息

```bash
rosbag info ~/r300_records/yolo_test_20260711_183000.bag
```

请将文件名替换为实际生成的文件名。

查看已有记录：

```bash
ls -lh ~/r300_records
```

---

#### 7.4 回放 rosbag

先启动 ROS Master：

```bash
roscore
```

打开一个新终端，以 `0.5` 倍速循环播放：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

rosbag play \
  ~/r300_records/yolo_test_20260711_183000.bag \
  --loop \
  -r 0.5
```

---

#### 7.5 查看回放画面

再打开一个新终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rqt_image_view
```

选择话题：

```text
/r300_vision/annotated_image
```

---

### 8. Web 实时画面发布

#### 8.1 启动 Web 视频服务器

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

启动服务：

```bash
rosrun web_video_server web_video_server \
  _port:=8080 \
  _address:=0.0.0.0 \
  _server_threads:=2 \
  _ros_threads:=2
```

其中：

- `0.0.0.0` 表示监听所有网卡；
- `8080` 为 Web 服务端口。

不建议固定绑定某个具体 IP，否则工控机 IP 变化后可能导致服务启动失败。

---

#### 8.2 查询工控机 IP

```bash
hostname -I
```

假设工控机 IP 为：

```text
192.168.1.107
```

浏览器访问 Web 首页：

```text
http://192.168.1.107:8080/
```

直接查看检测视频流：

```text
http://192.168.1.107:8080/stream?topic=/r300_vision/annotated_image&type=mjpeg
```

低带宽模式：

```text
http://192.168.1.107:8080/stream?topic=/r300_vision/annotated_image&type=mjpeg&quality=70&width=640&height=480
```

> `127.0.0.1` 只表示当前电脑本机。  
> 在其他电脑浏览器中，应使用工控机的实际局域网 IP。

---

### 9. 一键启动

一键启动脚本位于：

```text
~/r300_ws/scripts/start_r300.sh
```

首次使用前增加执行权限：

```bash
chmod +x ~/r300_ws/scripts/start_r300.sh
```

#### 9.1 启动相机、模型和 Web

```bash
~/r300_ws/scripts/start_r300.sh web
```

启动内容：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ Web视频服务
```

---

#### 9.2 启动相机、模型和 rosbag

```bash
~/r300_ws/scripts/start_r300.sh bag
```

启动内容：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ rosbag录制
```

---

#### 9.3 同时启动 Web 和 rosbag

```bash
~/r300_ws/scripts/start_r300.sh both
```

启动内容：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ Web视频服务
+ rosbag录制
```

停止所有节点并安全结束 rosbag：

```text
Ctrl + C
```

---

### 10. 直接使用总 Launch 文件

总 launch 文件可以启动：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ Web视频服务
```

启动命令：

```bash
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

roslaunch r300_yolo_detector r300_system.launch
```

关闭 Web：

```bash
roslaunch r300_yolo_detector r300_system.launch \
  enable_web:=false
```

修改 Web 端口：

```bash
roslaunch r300_yolo_detector r300_system.launch \
  web_port:=8081
```

---

### 11. 坐标系说明

目标三维位置默认发布在相机光学坐标系中：

```text
camera_color_optical_frame
```

坐标方向定义如下：

| 坐标轴 | 方向 |
|---|---|
| X | 图像右侧 |
| Y | 图像下方 |
| Z | 相机正前方 |

因此：

```text
position.z
```

表示目标距离相机的前向距离。

在正式接入无人车控制系统前，建议通过 TF 将目标位置从：

```text
camera_color_optical_frame
```

转换到：

```text
base_link
```

---

### 12. 常见问题

#### 12.1 找不到 ROS 包

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rospack profile
```

检查：

```bash
rospack find r300_yolo_detector
```

---

#### 12.2 CUDA 不可用

```bash
source ~/venvs/yolo26/bin/activate

python3 - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

#### 12.3 Web 页面无法访问

检查 Web 服务是否运行：

```bash
ss -lntp | grep 8080
```

检查防火墙：

```bash
sudo ufw status
```

如有需要，开放端口：

```bash
sudo ufw allow 8080/tcp
```

---

#### 12.4 Web 页面可以打开但没有图像

检查标注图像是否正常发布：

```bash
rostopic hz /r300_vision/annotated_image
```

检查话题发布者：

```bash
rostopic info /r300_vision/annotated_image
```

---

#### 12.5 rosbag 异常中断

如果出现：

```text
xxx.bag.active
```

重新建立索引：

```bash
rosbag reindex xxx.bag.active
```

修复并生成新文件：

```bash
rosbag fix \
  xxx.bag.active \
  xxx_fixed.bag
```

---

### 13. 推荐运行方式

日常查看检测效果：

```bash
~/r300_ws/scripts/start_r300.sh web
```

正式采集实验数据：

```bash
~/r300_ws/scripts/start_r300.sh both
```

仅启动 ROS 节点进行调试：

```bash
roslaunch r300_yolo_detector r300_system.launch
```

