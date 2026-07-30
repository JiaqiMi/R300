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

`r300_1x_navigation` 是当前 R300 无人车的导航主功能包，负责：

- 独立启动和解析 1X 惯导；
- 在每次导航开始时建立本次 ENU 导航原点；
- 将经纬度航点转换为 `map` 坐标目标；
- 使用 `move_base + NavfnROS + DWAPlannerROS` 完成全局规划、局部规划和速度控制；
- 通过 `scout_base_node` 将 `/subject1/cmd_vel_raw` 发送到底盘；
- 接入视觉障碍、雷达高程障碍、左右指示牌临时目标和赛程末端自主泊车；
- 提供设备门检、话题门检、TF 门检、CAN 配置、参数一致性检查和运行日志。

当前架构强调“感知、定位、导航分开启动”：

```text
1X 串口解析                         MID-360 / FAST-LIO / 高程图
start_1x.sh                         start_single_lidar_elevation.sh
    │                                         │
    ├── /one_x/ins_fix                        ├── /cloud_registered_body
    ├── /one_x/origin                         ├── /Odometry
    ├── /one_x/odom                           └── /elevation_mapping/elevation_map_raw
    │                                         │
    └──────────────────┬──────────────────────┘
                       ↓
            对应导航一键脚本
     start_real_nav.sh / start_r300_vision_nav.sh /
                 start_r300_lidar_nav.sh
                       ↓
           waypoint_executor + move_base
                       ↓
               NavfnROS + DWAPlannerROS
                       ↓
             /subject1/cmd_vel_raw
                       ↓
                scout_base_node
                       ↓
                   R300 底盘
```

> 1X、雷达感知栈和导航是三个独立进程。停止导航不会自动停止已经独立运行的 1X，也不会自动停止 MID-360、FAST-LIO 和高程图。

---

### 1. 当前导航模式

当前保留三种实车导航模式：

| 模式 | 启动脚本 | 主 Launch | 障碍来源 | DWA 配置 | Local Costmap 配置 |
|---|---|---|---|---|---|
| 纯实车导航 | `start_real_nav.sh` | `subject1_waypoint_nav.launch` | 不加载外部障碍层 | `subject1_dwa.yaml` | `subject1_local_costmap.yaml` |
| 视觉避障导航 | `start_r300_vision_nav.sh` | `subject1_waypoint_vision_nav.launch` | D435i + YOLO | `subject1_dwa_vision.yaml` | `subject1_local_costmap_vision.yaml` |
| 雷达避障导航 | `start_r300_lidar_nav.sh` | `subject1_waypoint_lidar_nav.launch` | MID-360 + FAST-LIO + 高程图 | `subject1_dwa_lidar.yaml` | `subject1_local_costmap_lidar.yaml` |

三种实车模式共同使用：

```text
/one_x/odom
    ↓
dwa_odom_adapter.py
    ↓
/subject1/dwa_odom
    ↓
waypoint_executor.py
    ↓
move_base
    ├── NavfnROS
    └── DWAPlannerROS
    ↓
/subject1/cmd_vel_raw
    ↓
scout_base_node
```

三种模式都会启动独立的 `/move_base`、`/waypoint_executor` 和底盘控制链，因此：

> **纯实车导航、视觉避障导航和雷达避障导航不能同时运行。**

---

### 2. Shell 与 Launch 的调用关系

日常运行优先使用 `scripts/one_key` 中的一键脚本，不建议直接执行底层 `roslaunch`。Shell 除了启动 Launch，还会执行设备检查、等待、原点设置、TF 检查、话题类型检查和失败清理。

| Shell | 直接启动的 Launch | Launch 内部继续调用 |
|---|---|---|
| `start_1x.sh` | `one_x_localization_only.launch` | 启动 `/one_x_serial_driver` |
| `start_real_nav.sh` | `subject1_waypoint_nav.launch` | `subject1_move_base.launch`、`scout_base.launch` |
| `start_r300_vision_nav.sh` | `subject1_waypoint_vision_nav.launch` | `subject1_waypoint_nav.launch`、`subject1_vision_avoidance.launch` |
| `start_r300_lidar_nav.sh` | `subject1_waypoint_lidar_nav.launch` | `subject1_waypoint_nav.launch`、`subject1_lidar_avoidance.launch`；可选启动指示牌节点和 `r300_autonomous_parking/autonomous_parking.launch` |
| `start_single_lidar_elevation.sh` | `single_lidar_elevation/single_lidar_elevation.launch` | MID-360、FAST-LIO、GPU 高程图 |
| `start_r300.sh` | `r300_yolo_detector/r300_system_dual.launch` | D435i、普通 YOLO、Web 视频等视觉链路 |

核心 Launch 分层如下：

```text
subject1_waypoint_nav.launch
├── scout_base/scout_base.launch
├── dwa_odom_adapter.py
├── subject1_move_base.launch
│   └── move_base
├── waypoint_executor.py
├── map -> odom 静态 TF
└── base_link -> livox_frame 静态 TF
```

视觉模式：

```text
subject1_waypoint_vision_nav.launch
├── subject1_waypoint_nav.launch
│   ├── 视觉专用 DWA YAML
│   └── 视觉专用 Local Costmap YAML
└── subject1_vision_avoidance.launch
    └── vision_obstacle_layer_node.py
```

雷达模式：

```text
subject1_waypoint_lidar_nav.launch
├── subject1_waypoint_nav.launch
│   ├── 雷达专用 DWA YAML
│   └── 雷达专用 Local Costmap YAML
├── subject1_lidar_avoidance.launch
│   └── lidar_obstacle_scan_node.py
├── one_x_alignment_gate.py
├── direction_sign_local_goal.py             # 开关控制
├── r300_autonomous_parking/autonomous_parking.launch  # 开关控制
└── subject1_lidar_nav.rviz                  # 仅 --rviz 时启动
```

---

### 3. 编译与执行权限

完整编译：

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash
catkin_make
source ~/r300_ws/devel/setup.bash
```

只编译导航功能包：

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash

catkin_make --pkg r300_1x_navigation \
  -DCMAKE_BUILD_TYPE=Release \
  -j4

source ~/r300_ws/devel/setup.bash
```

自主泊车功能发生修改后，建议同时编译：

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash

catkin_make --pkg \
  r300_vision_msgs \
  r300_autonomous_parking \
  r300_1x_navigation

source ~/r300_ws/devel/setup.bash
```

首次运行或出现 `Permission denied` 时：

```bash
chmod +x \
  ~/r300_ws/scripts/start_single_lidar_elevation.sh \
  ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key/*.sh \
  ~/r300_ws/src/R300/r300_1x_navigation/scripts/*.py \
  ~/r300_ws/src/R300/r300_autonomous_parking/scripts/*.py
```

也可以使用：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./fix_permissions.sh
```

---

### 4. 推荐的完整启动顺序

室外雷达避障比赛模式推荐按以下顺序启动：

```text
步骤 1：启动雷达、FAST-LIO 和高程图
步骤 2：启动 1X 惯导解析
步骤 3：按需启动普通视觉系统
步骤 4：启动雷达避障导航
步骤 5：等待全部门检通过
步骤 6：手动调用 start_waypoints，或启动脚本加 --run
步骤 7：全部普通航点完成后，按开关决定是否执行自主泊车
```

推荐使用不同终端分别运行，以便独立观察日志。

---

## 5. 启动 1X 惯导

### 5.1 基本启动

终端 1：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_1x.sh
```

当前默认值：

```text
串口：/dev/ttyACM0
波特率：230400
原点模式：deferred
节点：/one_x_serial_driver
```

需要更换串口或波特率时使用环境变量：

```bash
INS_PORT=/dev/ttyUSB0 \
INS_BAUD=230400 \
./start_1x.sh
```

其他可用环境变量：

```text
FULL_ATT=false
ORIGIN_MODE=deferred
ORIGIN_MAX_AGE=0.50
READY_TIMEOUT=90
```

### 5.2 deferred 原点模式

`start_1x.sh` 默认使用 `deferred`，启动过程是：

```text
启动 1X 串口解析
    ↓
立即发布原始 INS / GPS / IMU / 航向数据
    ↓
暂不固定 ENU 原点
    ↓
暂不发布有效的 /one_x/odom 和 odom -> base_link
    ↓
对应导航脚本调用 /one_x/set_current_origin
    ↓
使用调用时最新的 1X 位置建立本次 ENU 原点
    ↓
开始发布 /one_x/origin、/one_x/odom 和 odom -> base_link
```

因此看到下面日志是正常现象：

```text
1X parser is running; waiting for /one_x/set_current_origin
before publishing odometry/TF
```

直到导航脚本调用原点服务后，才会出现：

```text
Current 1X position set as ENU origin
```

### 5.3 1X 启动脚本的鲁棒检查

`start_1x.sh` 会依次执行：

1. 检查 ROS Noetic 和工作空间环境；
2. 检查串口文件是否存在；
3. 检查当前用户是否拥有串口读写权限；
4. 检查 `/one_x_serial_driver` 是否已经运行，防止重复打开同一个串口；
5. 清理七天前的旧 ROS 日志目录，避免 `roslaunch` 启动前扫描日志过慢；
6. 启动 `one_x_localization_only.launch`；
7. 最多等待 90 秒，直到 `/one_x_serial_driver` 注册；
8. 等待 `/one_x/ins_fix` 出现数据；
9. 等待 `/one_x/set_current_origin` 服务注册；
10. 打印当前 INS/GPS 状态。

> 1X 正常启动一次后，可以多次停止和重新启动导航；不要因为导航退出就重复启动第二个 1X。只有 `/one_x_serial_driver` 已经退出或不在当前 ROS Master 中时，才需要重启 1X。

### 5.4 1X 检查命令

```bash
rosnode list | grep one_x
rosservice list | grep /one_x/set_current_origin

rostopic hz /one_x/ins_fix
rostopic echo -n 1 /one_x/ins_fix
rostopic echo -n 1 /one_x/attitude
rostopic echo -n 1 /one_x/ins_status
rostopic echo -n 1 /one_x/heading_deg
```

导航建立原点后：

```bash
rostopic echo -n 1 /one_x/origin
rostopic hz /one_x/odom
rosrun tf tf_echo odom base_link
```

---

## 6. 启动雷达、FAST-LIO 与高程图

终端 2：

```bash
cd ~/r300_ws
RVIZ=0 ./scripts/start_single_lidar_elevation.sh
```

建议显式写 `RVIZ=0`，避免雷达感知栈自己的 RViz 占用 GPU 和内存。这里的 `RVIZ` 与导航脚本的 `--rviz` 是两个独立开关。

### 6.1 自动搜索雷达网口

未指定 `LIDAR_INTERFACE` 时，脚本会自动查找当前持有：

```text
192.168.1.50
```

的本机网口，而不是固定写死 `eth0` 或 `eth1`。

需要手动指定时：

```bash
LIDAR_INTERFACE=eth1 \
LIDAR_HOST_IP=192.168.1.50 \
LIDAR_IP=192.168.1.192 \
LIDAR_PROFILE=mid360-lidar \
RVIZ=0 \
./scripts/start_single_lidar_elevation.sh
```

常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `LIDAR_INTERFACE` | 自动搜索 | 雷达实际连接网口 |
| `LIDAR_HOST_IP` | `192.168.1.50` | 工控机雷达口地址 |
| `LIDAR_IP` | `192.168.1.192` | MID-360 地址 |
| `LIDAR_PROFILE` | `mid360-lidar` | NetworkManager 连接名称 |
| `RVIZ` | `1` | 雷达感知 RViz；建议实车运行设为 `0` |
| `RESTART_EXISTING` | `1` | 是否自动停止已有的同类雷达 Launch |
| `TILT_PITCH_DEG` | `-42.3` | 当前雷达俯仰安装标定值 |
| `LIO_INPUT_CROP` | `false` | FAST-LIO 输入裁剪开关 |

### 6.2 雷达启动脚本的鲁棒检查

脚本会依次：

1. 自动定位雷达网口；
2. 检查网口是否存在；
3. 使用 `ethtool` 检查物理链路；
4. 检查本机雷达 IP；
5. 必要时尝试启用 `mid360-lidar` 网络连接；
6. 使用指定网口 `ping` MID-360；
7. 检查并停止已有的同类雷达 Launch；
8. 启动 `single_lidar_elevation.launch`；
9. 等待感知栈成熟；
10. 检查 `/elevation_mapping/elevation_map_raw` 是否存在正常发布频率。

### 6.3 雷达感知检查

```bash
rostopic hz /cloud_registered_body
rostopic hz /Odometry
rostopic hz /elevation_mapping/elevation_map_raw

rosrun tf tf_echo odom body
```

高程图类型应为：

```bash
rostopic type /elevation_mapping/elevation_map_raw
```

正常输出：

```text
grid_map_msgs/GridMap
```

---

## 7. 启动纯实车导航

纯实车导航不加载视觉或雷达障碍层，适合：

- 1X 定位检查；
- 航点跟踪测试；
- DWA 基础参数调试；
- 无障碍场地低风险测试。

### 7.1 启动但不立即运动

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_real_nav.sh --no-rviz
```

门检全部通过后，手动开始：

```bash
rosservice call /subject1/start_waypoints "{}"
```

### 7.2 门检通过后自动开始

```bash
./start_real_nav.sh --no-rviz --run
```

### 7.3 常用开关

```bash
# 仅台架检查，不启动底盘
./start_real_nav.sh --no-base --no-rviz

# 不自动配置 CAN，只检查接口是否已经 UP
./start_real_nav.sh --no-setup-can --no-rviz

# 指定航点文件
./start_real_nav.sh \
  --no-rviz \
  --waypoints ~/r300_ws/src/R300/r300_1x_navigation/config/test_waypoints.yaml

# 指定纯实车 DWA 参数
./start_real_nav.sh \
  --no-rviz \
  --dwa-config ~/r300_ws/src/R300/r300_1x_navigation/config/subject1_dwa_test.yaml
```

主要 Launch：

```text
start_real_nav.sh
    ↓
subject1_waypoint_nav.launch
    ↓
subject1_move_base.launch
```

---

## 8. 启动视觉避障导航

视觉避障模式要求外部视觉系统已经发布：

```text
/r300_vision/detections
/camera/color/camera_info
```

先启动视觉系统：

```bash
~/r300_ws/scripts/start_r300.sh web
```

检查：

```bash
rostopic hz /r300_vision/detections
rostopic hz /camera/color/camera_info
```

再启动视觉避障导航：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_r300_vision_nav.sh --no-rviz
```

自动开始航点：

```bash
./start_r300_vision_nav.sh --no-rviz --run
```

台架检查：

```bash
./start_r300_vision_nav.sh --no-base --no-rviz
```

指定视觉话题：

```bash
./start_r300_vision_nav.sh \
  --no-rviz \
  --detections-topic /r300_vision/detections \
  --camera-info-topic /camera/color/camera_info
```

视觉障碍链路：

```text
/r300_vision/detections
    ↓
vision_obstacle_layer_node.py
    ↓
/r300_vision/obstacle_scan
    ↓
VisionSnapshotLayer
    ↓
/r300_vision/active_obstacle_scan
    ↓
local_costmap
    ↓
DWA
```

主要 Launch：

```text
start_r300_vision_nav.sh
    ↓
subject1_waypoint_vision_nav.launch
    ├── subject1_waypoint_nav.launch
    └── subject1_vision_avoidance.launch
```

> 视觉障碍只进入代价地图和 DWA 链路，不会直接向底盘发布速度。

---

## 9. 启动雷达避障导航

雷达避障是当前比赛的主导航模式。启动前必须确保：

```text
1X 已启动
MID-360 已启动
FAST-LIO 已启动
高程图已启动
```

若保持默认的左右指示牌功能，还必须提前启动普通视觉系统并确保：

```text
/r300_vision/detections
```

有数据。

### 9.1 默认完整模式

默认行为：

```text
雷达避障：开启
视觉左右指示牌：开启
航点完成后自主泊车：开启
导航 RViz：关闭
CAN 自动配置：开启
底盘：开启
航点：门检后等待手动开始
地面安全限速：执行
```

启动命令：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_r300_lidar_nav.sh
```

因为默认开启左右指示牌，所以需要提前运行：

```bash
~/r300_ws/scripts/start_r300.sh web
```

门检通过后手动开始：

```bash
rosservice call /subject1/start_waypoints "{}"
```

自动开始：

```bash
./start_r300_lidar_nav.sh --run
```

### 9.2 不使用左右指示牌，但保留末端自主泊车

这是“正常雷达航点避障 + 航点完成后找停车框”的推荐组合：

```bash
./start_r300_lidar_nav.sh \
  --no-sign-guidance \
  --no-rviz \
  --run
```

该模式下：

- 正常航点阶段不等待 `/r300_vision/detections`；
- 不启动 `direction_sign_local_goal.py`；
- 不会根据左右指示牌生成临时目标；
- 雷达避障仍正常工作；
- 自主泊车仍默认开启；
- 普通航点全部完成后，泊车节点会复用已有 D435i，或自动启动相机和 `parking_yolo_depth.launch`。

### 9.3 关闭自主泊车，保留左右指示牌

```bash
./start_r300_lidar_nav.sh \
  --no-auto-parking \
  --no-rviz \
  --run
```

航点全部完成后直接保持原来的 `COMPLETED` 状态，不执行停车巡视。

### 9.4 完全恢复旧版纯雷达行为

关闭左右指示牌和自主泊车：

```bash
./start_r300_lidar_nav.sh \
  --no-sign-guidance \
  --no-auto-parking \
  --no-rviz \
  --run
```

此时只保留：

```text
1X + GPS 航点 + move_base + 雷达高程避障 + DWA + 底盘
```

### 9.5 按需打开导航 RViz

雷达导航 RViz 默认关闭。需要时显式使用：

```bash
./start_r300_lidar_nav.sh --rviz
```

加载的配置文件为：

```text
config/subject1_lidar_nav.rviz
```

默认关闭：

```bash
./start_r300_lidar_nav.sh --no-rviz
```

> 即使启用 RViz，Launch 中的 RViz 也不是 `required` 节点；RViz 异常不应带停主导航。

### 9.6 台架模式

```bash
./start_r300_lidar_nav.sh \
  --no-base \
  --no-sign-guidance \
  --no-rviz
```

台架模式会执行定位、雷达、高程图、TF、代价地图和规划链路检查，但不启动 `scout_base_node`。

`--run` 不应与 `--no-base` 同时使用。

### 9.7 其他常用参数

```bash
# 不重新配置 CAN
./start_r300_lidar_nav.sh --no-setup-can --no-rviz

# 指定航点文件
./start_r300_lidar_nav.sh \
  --waypoints ~/r300_ws/src/R300/r300_1x_navigation/config/test_waypoints.yaml

# 指定高程图话题
./start_r300_lidar_nav.sh \
  --elevation-topic /elevation_mapping/elevation_map_raw

# 指定雷达障碍扫描话题
./start_r300_lidar_nav.sh \
  --scan-topic /r300_lidar/obstacle_scan

# 指定 FAST-LIO 坐标系
./start_r300_lidar_nav.sh \
  --lio-map-frame odom \
  --lio-body-frame body

# 调整门检超时
./start_r300_lidar_nav.sh --timeout 90
```

显示完整帮助：

```bash
./start_r300_lidar_nav.sh --help
```

---

## 10. 雷达导航开关汇总

| 功能 | 默认状态 | 开启方式 | 关闭方式 |
|---|---|---|---|
| 航点自动开始 | 关闭 | `--run` | 不加 `--run` |
| 导航 RViz | 关闭 | `--rviz` | `--no-rviz` |
| 底盘 | 开启 | 默认 | `--no-base` |
| `/one_x/path` 辅助轨迹 | 开启 | 默认 | `--no-path` |
| CAN 自动配置 | 开启 | `--setup-can` 或默认 | `--no-setup-can` |
| 左右指示牌临时目标 | 开启 | `--sign-guidance` 或默认 | `--no-sign-guidance` |
| 航点完成后自主泊车 | 开启 | `--auto-parking` 或默认 | `--no-auto-parking` |
| 地面安全限速真正压入 DWA | 开启 | `SPEED_LIMIT_APPLY=true` | `SPEED_LIMIT_APPLY=false` |
| 雷达感知 RViz | 开启 | `RVIZ=1 start_single_lidar_elevation.sh` | `RVIZ=0 start_single_lidar_elevation.sh` |

等价环境变量示例：

```bash
SIGN_GUIDANCE_ENABLED=false \
AUTO_PARKING_ENABLED=true \
SPEED_LIMIT_APPLY=true \
LAUNCH_RVIZ=false \
./start_r300_lidar_nav.sh --run
```

`SPEED_LIMIT_APPLY=false` 为影子观测模式：

- `/r300_lidar/ground_speed_limit` 仍会计算和发布；
- 计算结果不再动态限制 DWA 最大速度；
- 感知断流的兜底停车逻辑不因此关闭；
- 负障碍和未验证地面不会再通过该限速器主动降速，实车使用需谨慎。

---

## 11. 雷达避障数据链

雷达感知到 DWA 的完整链路：

```text
Livox MID-360
    ↓
FAST-LIO
    ├── /cloud_registered_body
    ├── /Odometry
    └── odom -> body
    ↓
elevation_mapping
    ↓
/elevation_mapping/elevation_map_raw
    ↓
lidar_obstacle_scan_node.py
    ├── /r300_lidar/obstacle_scan
    ├── /r300_lidar/obstacle_points
    └── /r300_lidar/ground_speed_limit
    ↓
VisionSnapshotLayer
    ↓
/r300_lidar/active_obstacle_scan
    ↓
global_costmap / local_costmap
    ↓
NavfnROS + DWAPlannerROS
```

虽然雷达层插件类名仍为：

```text
r300_1x_navigation/VisionSnapshotLayer
```

但雷达模式实例名称是：

```text
lidar_snapshot_layer
```

它订阅：

```text
/r300_lidar/obstacle_scan
```

用于障碍聚类、关联和短时间保持。

当前室外雷达模式还会：

- 将全局代价地图设为 `40 m × 40 m` 滚动窗口；
- 在全局代价地图中加载雷达快照层；
- 使用全局膨胀半径 `1.5 m`；
- 设置 `/move_base/planner_frequency=1.0`，使全局路径能够持续重规划；
- 使用航点执行器的远目标钳制逻辑，将远距离 GPS 航点分段送入滚动全局窗口。

---

## 12. 左右指示牌临时目标

左右指示牌功能只在雷达导航中可选启用，默认开启。

输入：

```text
/r300_vision/detections
```

节点：

```text
direction_sign_local_goal.py
```

参数：

```text
config/subject1_direction_sign.yaml
```

主要流程：

```text
正常执行 GPS 航点
    ↓
识别到左转或右转指示牌
    ↓
调用 /subject1/pause_waypoints
    ↓
根据当前位姿生成左前或右前临时 move_base 目标
    ↓
到达临时目标
    ↓
调用 /subject1/resume_waypoints
    ↓
继续当前尚未完成的 GPS 航点
```

它不会修改：

- 雷达障碍提取逻辑；
- DWA 参数；
- costmap 配置；
- 底盘控制话题。

重要接口：

```bash
rostopic echo /subject1/direction_sign/state
rosservice call /subject1/direction_sign/reset "{}"
```

关闭：

```bash
./start_r300_lidar_nav.sh --no-sign-guidance
```

关闭后，不需要普通 YOLO 在导航启动前持续发布 `/r300_vision/detections`。

---

## 13. 航点完成后的自主泊车

自主泊车默认开启，只集成在雷达导航流程中。

### 13.1 触发条件

导航启动时，节点：

```text
/r300_autonomous_parking
```

会先注册，但保持轻量等待状态，不会立即加载停车 YOLO。

它监听：

```text
/subject1/waypoint_status
```

只有检测到：

```text
state=COMPLETED
```

才会启动赛程末端泊车任务。

`IDLE`、`RUNNING`、`PAUSED`、`FAILED` 不会触发停车巡视。

### 13.2 泊车完整流程

```text
全部普通航点完成
    ↓
自主泊车状态进入 PREPARING
    ↓
复用已经运行的 D435i
或自动启动 D435i
    ↓
启动 r300_yolo_detector/parking_yolo_depth.launch
    ↓
订阅 /r300_parking_vision/detections
    ↓
车辆正方向原地旋转
    ↓
每 30° 停留 5 秒，共巡视一圈
    ↓
发现 parking_empty
置信度 >= 0.20
连续 5 帧有效
    ↓
锁定唯一停车目标
    ↓
停止原地巡视
    ↓
发布停车点地图坐标和经纬度
    ↓
将该停车点作为新的单次 move_base 目标
    ↓
继续使用雷达 costmap + Navfn + DWA 前往停车点
    ↓
到达后进入 FINISHED
```

默认巡视参数：

```text
每次正转角度：30°
每个方向保持：5 s
巡视圈数：1
正转角速度上限：0.25 rad/s
正转角速度下限：0.08 rad/s
角度容差：2°
```

对应配置文件：

```text
src/R300/r300_autonomous_parking/config/autonomous_parking.yaml
```

停车视觉配置：

```text
src/R300/r300_autonomous_parking/config/parking_perception.yaml
```

停车视觉 Launch：

```text
r300_yolo_detector/parking_yolo_depth.launch
```

### 13.3 一次性约束

停车任务具有一次性锁存：

- 本次导航中只允许触发一次；
- `/subject1/waypoint_status` 重复发布 `COMPLETED` 不会重复巡视；
- 找到停车目标后只发送一个新任务；
- 到达停车点后保持 `FINISHED`；
- 不自动重新布防；
- 不会因为停车目标本身完成，再次触发“航点完成后停车”。

### 13.4 泊车状态

状态话题：

```bash
rostopic echo /subject1/autonomous_parking/state
```

可能状态：

```text
WAITING_WAYPOINTS
PREPARING
ARMED
SEARCHING
COUNTING
TARGET_PUBLISHED
NAVIGATING
FINISHED
ERROR
CANCELLED
```

重要话题：

| 话题 | 类型 | 说明 |
|---|---|---|
| `/subject1/autonomous_parking/state` | `std_msgs/String` | 泊车状态和说明 |
| `/subject1/autonomous_parking/active` | `std_msgs/Bool` | 泊车任务是否活跃 |
| `/subject1/autonomous_parking/stable_count` | `std_msgs/Int32` | 连续有效停车框帧数 |
| `/subject1/autonomous_parking/search_active` | `std_msgs/Bool` | 是否正在原地巡视 |
| `/subject1/autonomous_parking/target_fix` | `sensor_msgs/NavSatFix` | 最终停车点经纬度 |
| `/subject1/autonomous_parking/target_pose` | `geometry_msgs/PoseStamped` | 最终停车点 `map` 坐标 |
| `/subject1/autonomous_parking/parking_target` | `r300_autonomous_parking/ParkingTarget` | 完整停车目标信息 |

停车视觉话题：

| 话题 | 说明 |
|---|---|
| `/r300_parking_vision/detections` | 停车框双模型检测结果 |
| `/r300_parking_vision/annotated_image` | 停车框标注图像 |
| `/r300_parking_vision/target_point` | 选中的停车框三维位置 |

调试服务：

```bash
rosservice call /subject1/autonomous_parking/reset "{}"
rosservice call /subject1/autonomous_parking/rearm "{}"
```

正常比赛流程不需要手动调用重置服务。

### 13.5 巡视期间的控制互斥

原地巡视会暂时向：

```text
/subject1/cmd_vel_raw
```

发布正向角速度。

同时发布：

```text
/subject1/autonomous_parking/search_active = true
```

`lidar_obstacle_scan_node.py` 在此期间会暂时禁止雷达自动脱困倒车，避免：

```text
停车巡视正转命令
与
雷达恢复倒车命令
```

互相争抢。巡视结束后恢复正常雷达脱困逻辑。

### 13.6 关闭自主泊车

```bash
./start_r300_lidar_nav.sh --no-auto-parking
```

关闭后，普通航点完成即结束，不启动：

- 停车协调节点；
- 停车 YOLO；
- 30° 分步巡视；
- 停车点新任务。

---

## 14. 雷达导航启动时的鲁棒门检

`start_r300_lidar_nav.sh` 不只是调用 Launch，还会依次执行以下检查。

### 14.1 启动前检查

1. ROS Noetic 和工作空间是否可用；
2. `r300_1x_navigation`、`move_base`、`navfn`、`dwa_local_planner`、`grid_map_msgs`、`scout_base` 等依赖是否存在；
3. 若启用指示牌，检查 `r300_vision_msgs`；
4. 若启用自主泊车，检查 `r300_autonomous_parking`、`r300_yolo_detector` 和 `r300_vision_msgs`；
5. 检查航点文件是否存在；
6. 等待 `/one_x_serial_driver` 注册；
7. 等待 `/one_x/ins_fix`；
8. 检查已有 `/move_base`，防止三种导航模式重复启动；
9. 等待高程图并确认类型为 `grid_map_msgs/GridMap`；
10. 若启用指示牌，等待 `/r300_vision/detections` 并确认消息类型；
11. 检查 FAST-LIO TF：`odom -> body`。

### 14.2 原点与定位门检

导航脚本每次启动都会：

```text
等待 /one_x/set_current_origin
    ↓
调用 /one_x/set_current_origin
    ↓
等待 /one_x/origin
    ↓
等待 /one_x/odom
    ↓
采集 3 秒 /one_x/odom
    ↓
输出帧数和位置极差
    ↓
检查 odom -> base_link
```

正式 Launch 内还会启动：

```text
one_x_alignment_gate.py
```

若室外正常模式下 1X 位姿完全冻结，门检节点会报错并终止导航，避免车辆在假定位状态下运动。

### 14.3 CAN 门检

底盘开启时：

1. 检查 `can0` 是否存在；
2. 默认执行：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

3. 检查 CAN 接口是否处于 `UP`。

不希望脚本重新配置 CAN：

```bash
./start_r300_lidar_nav.sh --no-setup-can
```

### 14.4 启动后验收

Launch 启动后继续检查：

- `/one_x/odom`；
- `/subject1/dwa_odom` 及消息类型；
- 高程图及消息类型；
- FAST-LIO TF；
- `/r300_lidar/obstacle_scan` 及消息类型；
- `/move_base/local_costmap/costmap`；
- `/subject1/start_waypoints`；
- 指示牌状态节点和话题；
- 自主泊车等待节点和话题；
- 雷达适配器实际加载的话题、坐标系；
- `lidar_snapshot_layer` 是否加载；
- 障碍保持时间是否有效；
- `/move_base` 是否订阅雷达扫描；
- `/r300_lidar/active_obstacle_scan`；
- DWA 的 odom 是否为 `/subject1/dwa_odom`；
- 室外全局地图宽度是否为 `40 m`；
- 全局膨胀是否为 `1.5 m`；
- 安全限速开关是否按启动参数加载；
- 关键节点是否达到 `5/5`；
- 航点 Marker；
- `/r300_lidar/ground_speed_limit`。

任何关键门检失败，脚本会退出并向 `/subject1/cmd_vel_raw` 发布零速度。

---

## 15. 航点任务控制

默认航点文件：

```text
config/subject1_waypoints.yaml
```

手动开始：

```bash
rosservice call /subject1/start_waypoints "{}"
```

暂停：

```bash
rosservice call /subject1/pause_waypoints "{}"
```

恢复：

```bash
rosservice call /subject1/resume_waypoints "{}"
```

跳过当前航点：

```bash
rosservice call /subject1/skip_waypoint "{}"
```

取消：

```bash
rosservice call /subject1/cancel_waypoints "{}"
```

实时状态：

```bash
watch -n 1 'rostopic echo -n 1 /subject1/waypoint_status'
```

状态字符串中重点关注：

```text
state=IDLE
state=RUNNING
state=PAUSED
state=COMPLETED
state=FAILED
```

以及：

```text
progress
current
last_command
transition
error
```

航点 Marker：

```bash
rostopic echo -n 1 /subject1/waypoint_markers
```

### 15.1 航点通过与弯道自适应

`subject1_waypoint_nav.launch` 默认启用中间航点提前切换和弯道自适应：

```text
pass_through_enabled=true
angle_adaptive_switch_enabled=true
```

航点执行器根据前后航段夹角，将航点分为：

```text
直线
缓弯
中弯
急弯
近似掉头
```

不同类型使用不同的提前切换距离和速度门限。最终航点仍等待 `move_base` 成功，不会像中间航点一样直接穿越切换。

---

## 16. 重要 ROS 话题

### 16.1 1X 定位

| 话题 | 说明 |
|---|---|
| `/one_x/ins_fix` | 1X INS 经纬高和状态 |
| `/one_x/fix` | 提供给其他任务使用的定位 `NavSatFix` |
| `/one_x/gps_fix` | 1X 协议中的 GPS 结果 |
| `/one_x/ins_imu` | INS 原始惯性数据 |
| `/one_x/imu` | IMU 消息 |
| `/one_x/attitude` | 俯仰、横滚、航向 |
| `/one_x/heading_deg` | 航向角，北 0°、东 90° |
| `/one_x/vel` | `Ve / Vn / Vu` |
| `/one_x/ins_status` | INS 模式、状态和故障信息 |
| `/one_x/diagnostics` | 串口、帧解析和校验诊断 |
| `/one_x/origin` | 本次导航 ENU 原点 |
| `/one_x/odom` | 导航使用的 1X 位置、姿态和速度 |
| `/one_x/path` | 可选的历史轨迹显示 |

### 16.2 规划与控制

| 话题 | 说明 |
|---|---|
| `/subject1/dwa_odom` | 经死区和滤波后的 DWA 速度反馈 |
| `/subject1/cmd_vel_raw` | DWA 或停车巡视发往底盘的速度指令 |
| `/subject1/waypoint_status` | 航点状态、当前航点和进度 |
| `/subject1/waypoint_markers` | 航点 RViz Marker |
| `/move_base/NavfnROS/plan` | Navfn 全局路径 |
| `/move_base/DWAPlannerROS/local_plan` | DWA 局部路径 |
| `/move_base/local_costmap/costmap` | 局部代价地图 |
| `/move_base/global_costmap/costmap` | 全局代价地图 |

### 16.3 视觉障碍与指示牌

| 话题 | 说明 |
|---|---|
| `/r300_vision/detections` | 普通视觉目标检测 |
| `/r300_vision/obstacle_scan` | 视觉障碍虚拟扫描 |
| `/r300_vision/active_obstacle_scan` | 视觉障碍保持层输出 |
| `/subject1/direction_sign/state` | 左右指示牌临时任务状态 |

### 16.4 雷达与高程图

| 话题 | 说明 |
|---|---|
| `/cloud_registered_body` | FAST-LIO 配准点云 |
| `/Odometry` | FAST-LIO 雷达里程计 |
| `/elevation_mapping/elevation_map_raw` | 原始高程图 |
| `/r300_lidar/obstacle_scan` | 高程图转换后的障碍扫描 |
| `/r300_lidar/obstacle_points` | 障碍调试点云 |
| `/r300_lidar/active_obstacle_scan` | 快照层当前有效障碍 |
| `/r300_lidar/ground_speed_limit` | 地面确认安全限速结果 |

### 16.5 自主泊车

| 话题 | 说明 |
|---|---|
| `/subject1/autonomous_parking/state` | 泊车状态 |
| `/subject1/autonomous_parking/search_active` | 是否正在巡视 |
| `/subject1/autonomous_parking/stable_count` | 连续有效帧数 |
| `/subject1/autonomous_parking/target_fix` | 最终停车点经纬度 |
| `/subject1/autonomous_parking/target_pose` | 最终停车点 `map` 坐标 |
| `/subject1/autonomous_parking/parking_target` | 完整停车目标 |
| `/r300_parking_vision/detections` | 停车框识别结果 |
| `/r300_parking_vision/annotated_image` | 停车识别画面 |

---

## 17. 重要服务

| 服务 | 功能 |
|---|---|
| `/one_x/set_current_origin` | 以当前最新 1X 位置建立 ENU 原点 |
| `/subject1/start_waypoints` | 开始全部航点 |
| `/subject1/pause_waypoints` | 暂停航点 |
| `/subject1/resume_waypoints` | 恢复航点 |
| `/subject1/skip_waypoint` | 跳过当前航点 |
| `/subject1/cancel_waypoints` | 取消航点任务 |
| `/subject1/direction_sign/reset` | 重置指示牌单次识别锁存 |
| `/subject1/autonomous_parking/reset` | 重置泊车节点 |
| `/subject1/autonomous_parking/rearm` | 重新布防泊车节点，正常比赛不使用 |

---

## 18. 重要配置文件

### 18.1 公共导航

```text
config/subject1_waypoints.yaml
config/subject1_costmap_common.yaml
config/subject1_global_costmap.yaml
config/subject1_move_base.yaml
config/subject1_nav_params.yaml
```

### 18.2 纯实车

```text
config/subject1_dwa.yaml
config/subject1_local_costmap.yaml
```

### 18.3 视觉避障

```text
config/subject1_dwa_vision.yaml
config/subject1_local_costmap_vision.yaml
config/subject1_vision_obstacles.yaml
```

### 18.4 雷达避障

```text
config/subject1_dwa_lidar.yaml
config/subject1_local_costmap_lidar.yaml
config/subject1_lidar_obstacles.yaml
config/subject1_lidar_nav.rviz
```

### 18.5 指示牌

```text
config/subject1_direction_sign.yaml
```

### 18.6 自主泊车

```text
src/R300/r300_autonomous_parking/config/autonomous_parking.yaml
src/R300/r300_autonomous_parking/config/parking_perception.yaml
```

不同模式的 DWA 和局部代价地图配置相互独立。修改雷达 DWA 不会自动影响纯实车或视觉模式。

---

## 19. 常用运行组合

### 19.1 比赛完整模式

普通视觉用于左右指示牌，雷达用于正负障碍，航点完成后自动找停车框：

```bash
# 终端1：雷达感知
cd ~/r300_ws
RVIZ=0 ./scripts/start_single_lidar_elevation.sh

# 终端2：1X
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_1x.sh

# 终端3：普通视觉
~/r300_ws/scripts/start_r300.sh web

# 终端4：雷达导航
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./start_r300_lidar_nav.sh --no-rviz --run
```

### 19.2 不识别左右指示牌，但保留自主泊车

```bash
./start_r300_lidar_nav.sh \
  --no-sign-guidance \
  --no-rviz \
  --run
```

### 19.3 只进行雷达航点避障

```bash
./start_r300_lidar_nav.sh \
  --no-sign-guidance \
  --no-auto-parking \
  --no-rviz \
  --run
```

### 19.4 只启动链路，不让车立即运动

```bash
./start_r300_lidar_nav.sh \
  --no-sign-guidance \
  --no-rviz
```

确认状态后：

```bash
rosservice call /subject1/start_waypoints "{}"
```

### 19.5 台架静态检查

```bash
./start_r300_lidar_nav.sh \
  --no-base \
  --no-sign-guidance \
  --no-auto-parking \
  --no-rviz
```

---

## 20. 导航链路检查

统一检查：

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./check_r300_nav.sh
```

关键参数：

```bash
rosparam get /move_base/DWAPlannerROS/odom_topic
rosparam get /dwa_odom_adapter/input_odom_topic
rosparam get /dwa_odom_adapter/output_odom_topic
```

正常应为：

```text
/move_base/DWAPlannerROS/odom_topic = /subject1/dwa_odom
/dwa_odom_adapter/input_odom_topic  = /one_x/odom
/dwa_odom_adapter/output_odom_topic = /subject1/dwa_odom
```

检查速度控制：

```bash
rostopic info /subject1/cmd_vel_raw
rostopic hz /subject1/cmd_vel_raw
```

检查 TF：

```bash
rosrun tf tf_echo map odom
rosrun tf tf_echo odom base_link
rosrun tf tf_echo base_link livox_frame
rosrun tf tf_echo odom body
```

检查 CAN：

```bash
ip -details -statistics link show can0
timeout 5 candump can0
```

---

## 21. 雷达链路专项检查

```bash
# 高程图
rostopic hz /elevation_mapping/elevation_map_raw

# 障碍转换
rostopic hz /r300_lidar/obstacle_scan
rostopic hz /r300_lidar/obstacle_points

# 快照层
rostopic hz /r300_lidar/active_obstacle_scan

# 代价地图
rostopic hz /move_base/local_costmap/costmap
rostopic hz /move_base/global_costmap/costmap

# 规划
rostopic hz /move_base/NavfnROS/plan
rostopic hz /move_base/DWAPlannerROS/local_plan

# 限速
rostopic echo /r300_lidar/ground_speed_limit
```

检查插件：

```bash
rosparam get /move_base/local_costmap/plugins
rosparam get /move_base/global_costmap/plugins
```

雷达模式中应包含：

```text
lidar_snapshot_layer
inflation_layer
```

---

## 22. ROS Bag 记录

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./record_r300_bag.sh
```

自定义输出：

```bash
OUT=~/bags/r300_test_01 ./record_r300_bag.sh
```

雷达比赛模式建议重点记录：

```text
/one_x/ins_fix
/one_x/odom
/one_x/origin
/subject1/dwa_odom
/subject1/cmd_vel_raw
/subject1/waypoint_status
/cloud_registered_body
/Odometry
/elevation_mapping/elevation_map_raw
/r300_lidar/obstacle_scan
/r300_lidar/obstacle_points
/r300_lidar/active_obstacle_scan
/r300_lidar/ground_speed_limit
/move_base/global_costmap/costmap
/move_base/local_costmap/costmap
/move_base/NavfnROS/plan
/move_base/DWAPlannerROS/local_plan
/subject1/direction_sign/state
/subject1/autonomous_parking/state
/subject1/autonomous_parking/target_fix
/tf
/tf_static
```

点云和高程图数据量较大，长时间记录前应检查磁盘空间。

---

## 23. 停止与重新启动

### 23.1 推荐停止方式

优先在对应的一键启动终端按：

```text
Ctrl+C
```

一键脚本会：

- 先向 `/subject1/cmd_vel_raw` 发布零速度；
- 停止主 `roslaunch`；
- 停止自身启动的辅助轨迹节点；
- 保留独立运行的 1X；
- 保留独立运行的雷达感知栈。

### 23.2 只停止导航，保留 1X

```bash
cd ~/r300_ws/src/R300/r300_1x_navigation/scripts/one_key
./stop_r300_nav.sh
```

### 23.3 停止 1X

```bash
./stop_1x.sh
```

### 23.4 同时停止导航和 1X

```bash
./stop_r300_nav.sh --with-1x
```

### 23.5 导航异常退出后的重启原则

若导航退出但 1X 仍正常：

```bash
rosnode list | grep -x /one_x_serial_driver
rostopic echo -n 1 /one_x/ins_fix
rosservice list | grep -x /one_x/set_current_origin
```

三项正常时，只需重新运行导航脚本，不要重新启动第二个 1X。

若系统进程里能看到 1X，但 `rosnode list` 看不到 `/one_x_serial_driver`，说明它可能连接在旧的 ROS Master 上，需要先停止残留 1X，再重新运行 `start_1x.sh`。

每次重新启动导航，脚本都会重新调用：

```text
/one_x/set_current_origin
```

以当时车辆当前位置建立新的 ENU 原点。

---

## 24. 使用注意

- 1X 默认波特率以当前代码中的 `230400` 为准；
- 必须先启动 1X，再启动任意实车导航；
- 雷达模式必须先启动 MID-360、FAST-LIO 和高程图；
- 雷达导航默认不启动导航 RViz；
- 雷达感知脚本自己的 RViz 默认开启，实车建议显式使用 `RVIZ=0`；
- 雷达导航默认开启左右指示牌；不需要时必须加 `--no-sign-guidance`；
- 雷达导航默认开启航点完成后的自主泊车；不需要时必须加 `--no-auto-parking`；
- `--no-sign-guidance` 不会关闭自主泊车；
- `--no-auto-parking` 不会关闭左右指示牌；
- 三种实车导航模式不能同时运行；
- `scout_base_node` 的 `odom_pub` 保持关闭，避免与 1X 的 `odom -> base_link` 冲突；
- DWA 使用 `/subject1/dwa_odom`，不是直接使用底盘轮速里程计；
- 雷达障碍同时进入局部和室外滚动全局代价地图；
- 指示牌只生成临时目标，不改变雷达避障和 DWA 控制链；
- 自主泊车只在全部普通航点 `COMPLETED` 后执行一次；
- 停车目标到达后状态保持 `FINISHED`，不会自动重复；
- 自主泊车节点在主雷达 Launch 中不是 `required`，泊车异常不应带停主导航；
- 1X 对准门检、雷达障碍节点和指示牌节点属于关键节点，异常可能终止对应导航 Launch；
- `SPEED_LIMIT_APPLY=false` 只适用于有明确测试目的的影子模式；
- 首次实车测试先用 `--no-base` 完成静态链路检查；
- 负障碍、高程突变和自动泊车必须在低速、空旷、有物理急停和安全员的环境中逐级验证。


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

