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

本模块用于在 R300 无人车上接入自研 1X 惯导/GPS，替代原飞控位姿输入，并基于 ROS1 `move_base + DWA` 实现：

- 惯导定位；
- 单点与多航点导航；
- DWA 仿真调参；
- 视觉障碍接入局部代价地图；
- Web 端查看 costmap、全局路径、局部路径和控制指令。

当前控制链路为：

```text
1X INS/GPS → /one_x/odom → move_base + DWA → /subject1/cmd_vel_raw → scout_base_node
```

视觉避障链路为：

```text
/r300_vision/detections → vision_obstacle_layer_node.py
→ /r300_vision/obstacle_scan → VisionSnapshotLayer
→ local_costmap → inflation_layer → DWA
```

系统不再使用 `cmd_vel_guard`、`dwa_odom_adapter`、预转向接管或其他中间控制模块。

---

### 1. 目录结构

```text
r300_1x_navigation/
├── config/
│   ├── subject1_dwa.yaml
│   ├── subject1_move_base.yaml
│   ├── subject1_waypoints.yaml
│   ├── subject1_costmap_common.yaml
│   ├── subject1_global_costmap.yaml
│   ├── subject1_local_costmap.yaml
│   ├── subject1_dwa_vision.yaml
│   ├── subject1_local_costmap_vision.yaml
│   └── subject1_vision_obstacles.yaml
├── launch/
│   ├── subject1_waypoint_nav.launch
│   ├── subject1_move_base.launch
│   ├── subject1_dwa_sim.launch
│   ├── one_x_localization_only.launch
│   ├── subject1_vision_avoidance.launch
│   └── subject1_waypoint_vision_nav.launch
├── scripts/
│   ├── waypoint_executor.py
│   ├── sim_r300_odom_node.py
│   ├── odom_to_path.py
│   ├── vision_obstacle_layer_node.py
│   ├── costmap_web_viewer.py
│   ├── diagnose_vision_costmap.py
│   └── one_key/
│       ├── start_real_nav.sh
│       ├── start_sim_dwa.sh
│       ├── start_localization_only.sh
│       ├── start_r300_vision_nav.sh
│       ├── send_goal_base.sh
│       ├── check_r300_nav.sh
│       ├── record_r300_bag.sh
│       ├── stop_r300_nav.sh
│       └── fix_permissions.sh
├── include/r300_1x_navigation/
│   └── vision_snapshot_layer.hpp
├── src/
│   ├── one_x_serial_driver.cpp
│   └── vision_snapshot_layer.cpp
├── maps/
├── vision_snapshot_layer_plugins.xml
├── CMakeLists.txt
└── package.xml
```

---

### 2. 主要 ROS 话题

#### 定位

| 话题 | 说明 |
|---|---|
| `/one_x/odom` | DWA 使用的里程计 |
| `/one_x/fix` | 当前导航位置 |
| `/one_x/ins_fix` | INS 经纬度 |
| `/one_x/gps_fix` | GPS 经纬度 |
| `/one_x/heading_deg` | 惯导航向角 |
| `/one_x/pos_compare` | INS/GPS 位置对比 |
| `/one_x/path` | 车辆轨迹 |

#### 规划与控制

| 话题 | 说明 |
|---|---|
| `/subject1/cmd_vel_raw` | DWA 输出到底盘的速度指令 |
| `/move_base/NavfnROS/plan` | 全局路径 |
| `/move_base/DWAPlannerROS/local_plan` | DWA 局部路径 |
| `/move_base/current_goal` | 当前目标点 |

#### 视觉避障

| 话题 | 说明 |
|---|---|
| `/r300_vision/detections` | 外部视觉检测结果 |
| `/r300_vision/obstacle_scan` | 视觉障碍几何扫描 |
| `/r300_vision/active_obstacle_scan` | VisionSnapshotLayer 当前有效障碍 |
| `/move_base/local_costmap/costmap` | DWA 使用的局部代价地图 |

---

### 3. 编译

```bash
cd ~/r300_ws && source /opt/ros/noetic/setup.bash && catkin_make --pkg r300_1x_navigation -j4 && source devel/setup.bash
```

首次使用时修复脚本权限：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./fix_permissions.sh
```

---

### 4. 实车导航

进入脚本目录：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
```

启动实车导航：

```bash
./start_real_nav.sh
```

指定惯导串口：

```bash
INS_PORT=/dev/ttyUSB0 ./start_real_nav.sh
```

配置 CAN 后启动：

```bash
SETUP_CAN=true CAN_PORT=can0 ./start_real_nav.sh
```

启动 RViz：

```bash
LAUNCH_RVIZ=true ./start_real_nav.sh
```

启动后自动执行航点：

```bash
AUTO_START=true ./start_real_nav.sh
```

手动开始航点：

```bash
rosservice call /subject1/start_waypoints "{}"
```

暂停、恢复、取消或跳过航点：

```bash
rosservice call /subject1/pause_waypoints "{}"
```

```bash
rosservice call /subject1/resume_waypoints "{}"
```

```bash
rosservice call /subject1/cancel_waypoints "{}"
```

```bash
rosservice call /subject1/skip_waypoint "{}"
```

---

### 5. 仅启动惯导定位

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./start_localization_only.sh
```

指定串口：

```bash
INS_PORT=/dev/ttyUSB0 ./start_localization_only.sh
```

该模式不启动 `move_base` 和底盘，不会控制车辆。

---

### 6. DWA 仿真

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./start_sim_dwa.sh
```

不启动 RViz：

```bash
RVIZ=false ./start_sim_dwa.sh
```

加入位置漂移与航向噪声：

```bash
DRIFT_Y_MPS=0.03 YAW_NOISE_DEG=0.5 ./start_sim_dwa.sh
```

加入周期性位置跳变：

```bash
JUMP_PERIOD_S=3.0 JUMP_STD_M=0.2 ./start_sim_dwa.sh
```

---

### 7. 发送测试目标

目标坐标系为 `base_link`，适合直接测试 DWA。

```bash
./send_goal_base.sh forward 10
```

```bash
./send_goal_base.sh back 5
```

```bash
./send_goal_base.sh left 3
```

```bash
./send_goal_base.sh right 3
```

```bash
./send_goal_base.sh xy 10 -2
```

---

### 8. 航点配置

航点文件：

```text
config/subject1_waypoints.yaml
```

示例：

```yaml
subject1_waypoints:
  waypoints:
    - name: wp_01
      latitude_deg: 38.98663491
      longitude_deg: 117.3418414
      altitude_m: 21.741

    - name: wp_02
      latitude_deg: 38.98664410
      longitude_deg: 117.3419243
      altitude_m: 21.741
```

同一个 `subject1_waypoints` 下只能保留一个 `waypoints:` 列表。

---

### 9. 视觉避障导航

视觉检测节点和相机节点需要先单独启动，并确保以下话题有数据：

```bash
rostopic hz /r300_vision/detections
```

```bash
rostopic hz /camera/color/camera_info
```

启动视觉导航：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./start_r300_vision_nav.sh --no-rviz
```

台架测试、不启动底盘：

```bash
./start_r300_vision_nav.sh --no-base --no-rviz
```

启动 Web costmap 查看器：

```bash
python3 ~/r300_ws/src/R300/r300_1x_navigation/scripts/costmap_web_viewer.py _port:=8088
```

减少小碎片障碍标注：

```bash
python3 ~/r300_ws/src/R300/r300_1x_navigation/scripts/costmap_web_viewer.py _port:=8088 _min_obstacle_cluster_beams:=4 _max_obstacle_labels:=5
```

查看工控机 IP：

```bash
hostname -I
```

浏览器打开：

```text
http://工控机IP:8088
```

例如：

```text
http://192.168.1.107:8088
```

视觉 costmap 诊断：

```bash
python3 ~/r300_ws/src/R300/r300_1x_navigation/scripts/diagnose_vision_costmap.py _duration_s:=20
```

正常状态应满足：

```text
红色视觉障碍点存在
黑色致命障碍存在
灰色膨胀区域存在
视觉障碍簇数量正确
完整 costmap 持续刷新
```

---

### 10. 链路检查

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./check_r300_nav.sh
```

实车正常链路：

```text
/one_x/odom：/one_x_serial_driver → /move_base
/subject1/cmd_vel_raw：/move_base → /scout_base_node
```

仿真正常链路：

```text
/one_x/odom：/sim_r300_odom_node → /move_base
/subject1/cmd_vel_raw：/move_base → /sim_r300_odom_node
```

常用检查：

```bash
rostopic info /subject1/cmd_vel_raw
```

```bash
rostopic info /one_x/odom
```

```bash
rosrun tf tf_echo odom base_link
```

```bash
rosparam get /move_base/DWAPlannerROS
```

---

### 11. 录制 ROS Bag

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./record_r300_bag.sh
```

自定义文件名：

```bash
OUT=~/bags/r300_dwa_test_01 ./record_r300_bag.sh
```

默认记录惯导、路径、速度、目标点和 TF 等关键话题。

---

### 12. 停止系统

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && ./stop_r300_nav.sh
```

每次重新测试前建议先执行一次，避免旧节点和旧 TF 残留。

---

### 13. 推荐测试流程

#### 实车导航

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key && INS_PORT=/dev/ttyACM0 CAN_PORT=can0 ./start_real_nav.sh
```

```bash
./check_r300_nav.sh
```

```bash
rosservice call /subject1/start_waypoints "{}"
```

#### 视觉避障台架测试

```bash
./start_r300_vision_nav.sh --no-base --no-rviz
```

```bash
python3 ~/r300_ws/src/R300/r300_1x_navigation/scripts/costmap_web_viewer.py _port:=8088
```

```bash
./send_goal_base.sh forward 5
```

#### 视觉避障实车测试

```bash
./start_r300_vision_nav.sh --no-rviz
```

```bash
./send_goal_base.sh forward 5
```

首次实车避障测试应使用低速参数，并准备物理急停。

---

### 14. 注意事项

- `scout_base_node` 的 `odom_pub` 应保持 `false`，避免与 1X 惯导 TF 冲突；
- `/subject1/cmd_vel_raw` 由 `move_base` 直接发送给底盘；
- 修改 YAML 后需要重启 `move_base`；
- `always_send_full_costmap: true` 时，`costmap_updates` 为 `0 Hz` 属于正常现象，应观察完整 `/costmap` 频率；
- VisionSnapshotLayer 在 `odom` 坐标系中保持视觉障碍，避免车辆转向时障碍跟随车体漂移；
- 红色点是视觉障碍采样点，DWA真正使用的是黑色致命障碍和灰色膨胀区域；
- 视觉障碍只进入局部代价地图，DWA可能绕行，也可能停车，不会自动重新生成长距离全局路径；
- 室内纯惯性位置会漂移，仅适合低速定性测试；
- 高速实车测试必须逐级提速，并保留物理急停。


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

## Web功能

包含：

- 实时视觉检测画面：`/r300_vision/annotated_image`
- 车辆状态：`/one_x/odom`、`/one_x/fix`、`/one_x/gps_fix`、`/one_x/heading_deg`
- 控制指令：`/subject1/cmd_vel_raw`
- 路径显示：`/move_base/NavfnROS/plan`、`/move_base/DWAPlannerROS/local_plan`
- 代价地图显示：`/move_base/local_costmap/costmap`
- 激光与视觉障碍俯视图：`/scan`、`/r300_vision/obstacle_scan`、`/r300_vision/active_obstacle_scan`
- 检测目标列表：`/r300_vision/detections`、`/r300_vision/target_point`
- 航点服务按钮：开始、暂停、恢复、跳过、取消


## 1. 安装依赖

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-rosbridge-server \
  ros-noetic-web-video-server \
  ros-noetic-tf2-web-republisher
```

## 2. 编译

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg r300_web_dashboard
source devel/setup.bash
```


## 3. 启动

先启动原来的 R300 系统：

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

## 4. 端口说明


```
8090：上位机网页
9090：rosbridge websocket
8080：web_video_server 图像流
```

## 5. 分步调试

### 5.1 检查 rosbridge

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

浏览器控制台没有 websocket 报错即可。

### 5.2 检查视频服务

```bash
rosrun web_video_server web_video_server _port:=8080 _address:=0.0.0.0
```

浏览器直接访问：

```text
http://工控机IP:8080/stream?topic=/r300_vision/annotated_image&type=mjpeg
```

### 5.3 检查关键话题

```bash
rostopic hz /one_x/odom
rostopic hz /subject1/cmd_vel_raw
rostopic hz /move_base/local_costmap/costmap
rostopic hz /r300_vision/annotated_image
rostopic hz /scan
```
### 5.4 检查关键话题
一键启动 Web：

```bash
roslaunch r300_web_dashboard r300_web_dashboard.launch
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

```

