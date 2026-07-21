# single_lidar_elevation（bringup）

单 MID-360 独立感知 bringup：**一条 launch 发布点云 + 里程计 + 高程图**。
不含任何运动控制，暂不与导航主业务对接，只把数据话题发布出来，
供 R300 导航等下游模块按需订阅。替代原视觉（D435i+YOLO）障碍感知的数据源。不依赖 URDF。

数据流：

```
MID-360 ──► livox_ros_driver2 ──► FAST-LIO ──► /cloud_registered(_body) + /Odometry + TF
                │                                        │
                └► madgwick ► 重力对齐TF(odom→camera_init)└► elevation_mapping_cupy ► /elevation_mapping/elevation_map_raw
```

## 依赖

**全部 ROS 包依赖已 vendor 在上一级目录**（见 [../README.md](../README.md)），
本仓库单独 clone 即可编译。系统级依赖仅剩：

- apt：`ros-<distro>-imu-filter-madgwick ros-<distro>-pcl-ros ros-<distro>-pybind11-catkin ros-<distro>-cv-bridge`
- Python：cupy（与本机 CUDA 匹配的 cupy-cuda11x/12x，**版本须 <14**，14 起要求 numpy≥2 与 ROS1 冲突）、
  PyTorch（CUDA 版，traversability 滤波用）、`numpy<2`、scipy、shapely
- GPU（Jetson Orin 系列可用）

注意：`rosdep install` 不认识 vendor 包（不在 rosdistro 索引），报 unknown key 属正常，
用 `--skip-keys "livox_ros_driver2 fast_lio elevation_mapping_cupy"` 跳过即可。

## 编译（Jetson，ROS1）

在 R300 仓库根目录（仓库本身即 catkin 工作空间，src/ 在仓库根下）：

```bash
catkin_make
source devel/setup.bash
```

## 运行

```bash
roslaunch single_lidar_elevation single_lidar_elevation.launch
# 常用参数：
#   lidar_ip:=192_168_1_192       雷达 IP（下划线格式，须与驱动 json 一致，默认即前雷达固定 IP）
#   user_config_path:=/path/x.json 驱动网络配置，换 IP 时可指向自己的 json
#   tilt_pitch_deg:=-45           雷达安装俯仰修正；水平安装填 0
#   publish_base_link_tf:=false   是否发布 body->base_link 恒等 TF（默认关，见下方 TF 警告）
#   lio_input_crop:=false         FAST-LIO 输入前裁剪（车体进视场时先标定 crop_front.yaml 再开）
#   rviz:=true                    打开预配置可视化
```

前置条件：主机网口已配 `192.168.1.50/24`（与驱动 json 的 host ip 一致），能 ping 通雷达。

**换雷达 IP 三步**：① 改（或另写一份）`config/MID360_single.json` 里 `lidar_configs[].ip`
和 `host_net_info` 各 ip；② launch 传 `lidar_ip:=<下划线格式新IP>`；③ 若另写了 json，
传 `user_config_path:=<路径>`。

## 输出话题

| 话题 | 类型 | 坐标系 | 频率 | 说明 |
|---|---|---|---|---|
| `/livox/lidar_<IP>` | livox CustomMsg | livox_frame | 10 Hz | 驱动原始点云（非 PointCloud2，rviz 不能直接显示） |
| `/cloud_registered` | sensor_msgs/PointCloud2 | camera_init | 10 Hz | 配准到里程计系的点云 |
| `/cloud_registered_body` | sensor_msgs/PointCloud2 | body | 10 Hz | 配准点云（机体系）。**接 costmap obstacle_layer 等下游用这个** |
| `/Odometry` | nav_msgs/Odometry | camera_init→body | ~10 Hz | FAST-LIO 里程计 |
| `/Odometry_precede` | nav_msgs/Odometry | 同上 | ~200 Hz | fork 内置高频里程计 |
| `/elevation_mapping/elevation_map_raw` | grid_map_msgs/GridMap | odom | ~5 Hz | 高程图（elevation / traversability 层） |
| `/elevation_mapping/elevation_map_filter` | grid_map_msgs/GridMap | odom | ~3 Hz | 高程图 min_filter 层（负障碍检测用） |

TF 链：`odom →(重力对齐,静态)→ camera_init →(FAST-LIO)→ body →(恒等,可选,默认关)→ base_link`

## 与已有导航栈共跑（必读）

- **base_link 双父帧**：R300 上 1X 惯导（one_x_serial_driver）已发布
  `odom->base_link`，故本包 `publish_base_link_tf` 默认 false，请勿在共跑时打开，
  否则 base_link 同时有两个父帧（对方的 odom 与本栈的 body），TF 树抖动。
- **odom 同名合并陷阱（比双父帧更隐蔽）**：本栈的 TF 链
  `odom->camera_init->body` 与 R300 自己的 `odom->base_link` 共用了
  `odom` 这个帧名——TF 会把两棵语义不同的树静默连成一棵，
  `lookupTransform(base_link, body)` 能查成功但结果是错的。
  两套 odom 是**独立里程计、不同原点**，绝不能经同名 odom 互查坐标。
- **后续正确的对接方式**（本包暂不做）：写一个适配节点，订阅本栈输出
  （`/cloud_registered_body`，frame=body，或高程图），在节点内部用
  "雷达安装位→车体"的固定外参把数据换到 R300 自己树里的帧
  （如 base_link）再发布，**不要**试图用跨树 TF 完成这一步。
  然后作为 costmap obstacle_layer 的 PointCloud2 源，或阈值化成
  OccupancyGrid / 虚拟 LaserScan（对齐原视觉链路的
  /r300_vision/obstacle_scan 接口）喂给 VisionSnapshotLayer。

## 验证

```bash
rostopic hz /livox/lidar_192_168_1_192            # ① ~10Hz
rostopic echo /Odometry -n1                        # ② 静止漂移厘米级
rostopic hz /cloud_registered_body                 # ③ ~10Hz
rostopic hz /elevation_mapping/elevation_map_raw   # ④ ~5Hz（需等重力对齐 TF，启动后约 3~5 秒）
```

## 已知边界与运维提示

- 仅支持单雷达；本包是上游感知栈的单雷达冻结拷贝（去掉了 URDF/自体过滤/后雷达），
  上游调参不会自动同步到本包。
- 单雷达为楔形视场，侧后方靠"地图记忆"补盲：起步建议原地慢转一圈把周围种满；
  倒车/原地掉头吃的是旧数据，需谨慎。
- **FAST-LIO 启动后 10 秒内勿碰车、勿站雷达正前**——启动瞬态被扰动可能发散，
  且发散不可自愈（实测曾漂到数十公里），发散就重启 launch。
- 高程图需 GPU（CuPy + PyTorch）。注意 cupy 须 <14（14 起要求 numpy≥2，与 ROS1 Python 栈冲突）。
- 雷达自身/安装件/车体结构若进入视场，既会在高程图留下自体残留，更会污染
  FAST-LIO 里程计（上游真机教训：自体点致导航"起步即碰撞"）。处理方式：
  在 rviz 里按**传感器系**实测自体点范围，填进 `config/crop_front.yaml` 的
  裁剪盒后以 `lio_input_crop:=true` 启动（包内自带 custom_msg_crop.py，
  CustomMsg 进出，裁剪后再喂 FAST-LIO）；`preprocess/blind` 只按距离，不够用。
