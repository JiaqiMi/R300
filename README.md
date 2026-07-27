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

## 单雷达高程图感知模块（single_lidar_elevation）

自包含感知功能组：单颗 Livox MID-360 → FAST-LIO 里程计 → GPU 高程图，
一条 launch 发布点云 / 里程计 / 高程图话题，与本仓库既有导航、视觉主业务零耦合
（不启动该模块时一切照旧）。所有第三方依赖源码已 vendor 在模块目录内，单独 clone 本仓库即可编译。

- 模块说明：[src/single_lidar_elevation/README.md](src/single_lidar_elevation/README.md)（依赖 / 编译 / 运行）、
  [src/single_lidar_elevation/bringup/README.md](src/single_lidar_elevation/bringup/README.md)（话题 / TF / 与既有导航栈共跑警告，必读）
- 验证记录：[src/single_lidar_elevation/VERIFICATION.md](src/single_lidar_elevation/VERIFICATION.md)
- 一键运行：`roslaunch single_lidar_elevation single_lidar_elevation.launch`
- 注意：整仓 `catkin_make` 现在会连带编译该模块的 11 个 catkin 包（需 PCL、pybind11-catkin、CUDA 等，
  见模块 README 的系统要求）。只做视觉 / 导航开发、不想编译它时可用黑名单跳过：

  ```bash
  catkin_make -DCATKIN_BLACKLIST_PACKAGES="single_lidar_elevation;livox_ros_driver2;fast_lio;elevation_mapping_cupy;elevation_map_msgs;grid_map_core;grid_map_msgs;grid_map_cv;grid_map_sdf;grid_map_ros;grid_map_rviz_plugin"
  ```

---

## 1X 惯导、导航规划与控制（r300_1x_navigation）

该功能包负责 R300 无人车的 1X 惯导解析、经纬度航点导航、`move_base + Navfn + DWA` 路径规划、视觉障碍接入、仿真调参和运行数据记录。

当前推荐架构如下：

```text
独立 1X 解析
/one_x/odom
    ↓
dwa_odom_adapter
    ↓
/subject1/dwa_odom
    ↓
move_base + Navfn + DWA
    ↓
/subject1/cmd_vel_raw
    ↓
scout_base_node
```

视觉避障模式额外增加：

```text
/r300_vision/detections
    ↓
/r300_vision/obstacle_scan
    ↓
VisionSnapshotLayer + InflationLayer
    ↓
local_costmap
    ↓
DWA
```

纯实车导航和视觉避障导航共同使用 1X、航点执行器、`dwa_odom_adapter` 和 `/subject1/dwa_odom`。两者主要区别为：

| 模式 | DWA 参数 | 局部障碍 |
|---|---|---|
| 纯实车导航 | `config/subject1_dwa.yaml` | 不加载视觉障碍层 |
| 视觉避障导航 | `config/subject1_dwa_vision.yaml` | 加载 `VisionSnapshotLayer` |

---

### 1. 编译与权限

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg r300_1x_navigation -DCMAKE_BUILD_TYPE=Release -j4
source ~/r300_ws/devel/setup.bash
```

首次使用或脚本提示 `Permission denied` 时：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
chmod +x *.sh
chmod +x ../*.py
```

---

### 2. 推荐启动顺序

#### 2.1 单独启动 1X

终端 1：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_1x.sh
```

指定串口：

```bash
INS_PORT=/dev/ttyUSB0 ./start_1x.sh
```

1X 默认以 `deferred` 模式运行：先发布原始解析数据，导航启动时再由导航脚本调用 `/one_x/set_current_origin`，以当时最新位置建立 ENU 原点。初始方向仍使用 1X 的真实航向，不会强制归零。

常用检查：

```bash
rostopic hz /one_x/ins_fix
rostopic hz /one_x/ins_imu
rostopic echo -n 1 /one_x/attitude
rostopic echo -n 1 /one_x/ins_status
```

#### 2.2 启动纯实车导航

终端 2：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_real_nav.sh --no-rviz
```

完成检查后自动开始航点：

```bash
./start_real_nav.sh --no-rviz --run
```

纯实车导航默认加载：

```text
config/subject1_dwa.yaml
```

临时指定其他 DWA 文件：

```bash
./start_real_nav.sh \
  --no-rviz \
  --dwa-config ~/r300_ws/src/R300/r300_1x_navigation/config/subject1_dwa_test.yaml
```

#### 2.3 启动视觉避障导航

先确保相机和视觉检测节点已运行：

```bash
rostopic hz /camera/color/camera_info
rostopic hz /r300_vision/detections
```

再启动导航：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_r300_vision_nav.sh --no-rviz
```

完成视觉链路检查后自动开始航点：

```bash
./start_r300_vision_nav.sh --no-rviz --run
```

台架测试、不启动底盘：

```bash
./start_r300_vision_nav.sh --no-base --no-rviz
```

视觉导航默认加载：

```text
config/subject1_dwa_vision.yaml
config/subject1_local_costmap_vision.yaml
```

---

### 3. 航点控制、状态和进度

航点文件：

```text
config/subject1_waypoints.yaml
```

开始、暂停、恢复、跳过和取消：

```bash
rosservice call /subject1/start_waypoints "{}"
rosservice call /subject1/pause_waypoints "{}"
rosservice call /subject1/resume_waypoints "{}"
rosservice call /subject1/skip_waypoint "{}"
rosservice call /subject1/cancel_waypoints "{}"
```

实时查看当前航点、总进度和任务状态：

```bash
watch -n 1 'rostopic echo -n 1 /subject1/waypoint_status'
```

状态中重点关注：

```text
state          IDLE / RUNNING / PAUSED / COMPLETED / FAILED
progress       当前航点/总航点
current        当前航点名称
last_command   START / PAUSE / RESUME / SKIP / CANCEL
transition     GOAL_SENT / WAYPOINT_REACHED / PAUSED / FAILED 等
error          失败原因
```

两种导航模式的最大允许航点距离默认统一为 `5000 m`。运行时检查：

```bash
rosparam get /waypoint_executor/max_goal_distance_from_origin_m
```

---

### 4. 主要话题

#### 1X 原始与导航数据

| 话题 | 说明 |
|---|---|
| `/one_x/ins_fix` | 1X 组合导航 INS 经纬高 |
| `/one_x/gps_fix` | 协议中的 GPS 经纬高 |
| `/one_x/ins_imu` | 原始三轴角速度和加速度 |
| `/one_x/attitude` | 原始 `pitch / roll / heading` |
| `/one_x/vel` | 原始 `Ve / Vn / Vu` |
| `/one_x/update_flag` | Byte 107 更新标志及各 bit 含义 |
| `/one_x/ins_status` | INS 工作状态、导航模式和有效/故障状态 |
| `/one_x/origin` | 当前导航 ENU 原点 |
| `/one_x/odom` | 1X 位置、姿态和速度 |
| `/one_x/diagnostics` | 串口、帧校验和状态诊断 |

#### 规划与控制

| 话题 | 说明 |
|---|---|
| `/subject1/dwa_odom` | DWA 实际使用的速度反馈 |
| `/subject1/cmd_vel_raw` | DWA 输出到底盘的速度指令 |
| `/subject1/waypoint_status` | 航点状态和进度 |
| `/move_base/NavfnROS/plan` | 全局路径 |
| `/move_base/DWAPlannerROS/local_plan` | DWA 局部路径 |
| `/move_base/local_costmap/costmap` | DWA 使用的局部代价地图 |

#### 视觉障碍

| 话题 | 说明 |
|---|---|
| `/r300_vision/detections` | 视觉检测结果 |
| `/r300_vision/obstacle_scan` | 视觉障碍扫描 |
| `/r300_vision/active_obstacle_scan` | 当前仍有效的障碍扫描 |

---

### 5. DWA Web 仿真调参

启动独立仿真调参台：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_dwa_web_tuner.sh
```

浏览器地址以终端打印为准，当前脚本默认通常为：

```text
http://127.0.0.1:8090
```

远程 SSH 使用本地端口转发：

```bash
ssh -L 8090:127.0.0.1:8090 explorer@工控机IP
```

该模式会启动独立仿真环境，提供：

- DWA、`move_base`、局部 costmap 和膨胀层参数调节；
- 单点和多航点任务；
- 人工添加视觉障碍；
- 全局路径、局部路径、轨迹和 costmap 显示；
- 参数、航点和障碍配置导出。

仿真模式不启动 1X、相机、YOLO 和底盘。它与实车导航共用 `/move_base` 等命名空间，**不要与纯实车导航或视觉避障导航同时运行**。

---

### 6. 实时在线调参

先启动纯实车导航或视觉避障导航，再另开终端：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_live_dynamic_tuner.sh
```

指定端口：

```bash
WEB_PORT=8072 ./start_live_dynamic_tuner.sh
```

远程 SSH 转发：

```bash
ssh -L 8072:127.0.0.1:8072 explorer@工控机IP
```

浏览器打开：

```text
http://127.0.0.1:8072
```

使用方法：

1. 点击“读取当前参数”；
2. 修改需要调整的字段；
3. 点击“应用已修改项”；
4. 参数在下一个控制周期或 costmap 更新周期生效；
5. 调试完成后导出 YAML 或 JSON。

在线调参只修改当前 ROS 运行值，**不会自动写回配置文件**。重启导航后会重新加载磁盘中的 YAML。`latch_xy_goal_tolerance` 等非动态参数仍需修改 YAML 后重启。

实车运动过程中优先调整评分权重和采样参数；速度、加速度、控制频率、膨胀半径和地图尺寸应停车后再修改。

---

### 7. 链路检查

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./check_r300_nav.sh
```

关键检查命令：

```bash
rosparam get /move_base/DWAPlannerROS/odom_topic
rosparam get /dwa_odom_adapter/input_odom_topic
rosparam get /dwa_odom_adapter/output_odom_topic
rostopic info /subject1/dwa_odom
rostopic info /subject1/cmd_vel_raw
rosrun tf tf_echo odom base_link
```

正常情况下：

```text
/move_base/DWAPlannerROS/odom_topic = /subject1/dwa_odom
/dwa_odom_adapter/input_odom_topic  = /one_x/odom
/dwa_odom_adapter/output_odom_topic = /subject1/dwa_odom
```

---

### 8. ROS Bag 记录

开始记录：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./record_r300_bag.sh
```

自定义文件名：

```bash
OUT=~/bags/r300_test_01 ./record_r300_bag.sh
```

按 `Ctrl+C` 正常结束并写入 bag 索引。默认保存 1X 原始数据、导航里程计、DWA 反馈、速度指令、规划路径、航点状态、视觉障碍和 TF 等关键话题。

查看记录：

```bash
rosbag info ~/bags/文件名.bag
rqt_bag ~/bags/文件名.bag
```

---

### 9. 停止系统

只停止导航、保留独立 1X：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./stop_r300_nav.sh
```

停止 1X：

```bash
./stop_1x.sh
```

同时停止导航和 1X：

```bash
./stop_r300_nav.sh --with-1x
```

---

### 10. 使用注意

- 先启动 `start_1x.sh`，再启动纯实车或视觉避障导航；
- 纯实车和视觉避障导航不要同时启动；
- `scout_base_node` 的 `odom_pub` 应保持关闭，避免与 1X 的 `odom → base_link` 冲突；
- 两种模式均通过 `dwa_odom_adapter` 使用 `/subject1/dwa_odom`；
- 纯实车和视觉模式分别使用 `subject1_dwa.yaml` 与 `subject1_dwa_vision.yaml`；
- Web 仿真参数与在线运行参数互不共享，最终有效配置应写回对应 YAML；
- 视觉障碍只进入局部代价地图，Navfn 全局路径不会因视觉障碍自动重规划；
- 实车测试应逐级提速，并始终保留物理急停。

---

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

