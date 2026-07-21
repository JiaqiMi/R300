# single_lidar_elevation —— 单 MID-360 雷达 + 高程图 感知模块（自包含）

本目录是一个**自包含的感知功能组**：单颗 Livox MID-360 经 FAST-LIO 里程计输出配准点云，
再由 GPU 高程图节点发布可通行性地图。只发布话题，不含任何运动控制，与 R300
现有导航/视觉主业务零耦合——不启动本模块时 R300 一切照旧。

所有第三方依赖源码已 vendor 在本目录内，R300 仓库单独 clone 即可编译运行，
无需再拉任何外部仓库：

| 子目录 | 内容 | 来源 / 版本 | 本地改动 |
|---|---|---|---|
| `bringup/` | 本模块的 launch/config/脚本（catkin 包名 `single_lidar_elevation`） | 自研 | — |
| `livox_ros_driver2/` | Livox 雷达 ROS1 驱动 | github.com/tongtybj/livox_ros_driver2 分支 `PR/ros1`（commit 7b47135） | CMake 的 SDK 源指向同级 `../livox_sdk2`（离线编译） |
| `livox_sdk2/` | Livox-SDK2 源码（驱动编译时经 ExternalProject 使用） | github.com/Livox-SDK/Livox-SDK2 tag `v1.2.2` | 已预打 `vendor_patches/spdlog-quiet.patch`（日志级别 debug→warn） |
| `FAST_LIO/` | LIO 里程计（额外发布 /Odometry_precede、/cloud_registered_body 的 fork） | github.com/tongtybj/FAST_LIO 分支 `PR/odometry` | 删除论文/图片素材以瘦身；含 ikd-Tree、IKFoM 源码 |
| `elevation_mapping_cupy/` | GPU 高程图（elevation_map_msgs + elevation_mapping_cupy 两包） | github.com/leggedrobotics/elevation_mapping_cupy | 删除 docs/docker/plane_segmentation/sensor_processing；`elevation_mapping_ros.cpp` 补 boost join 头文件 |
| `grid_map/` | 高程图消息与 rviz 插件等 6 个子包 | github.com/ANYbotics/grid_map（commit cdd0ea2） | 各子包 CMake `-std=c++11` → `-std=c++17` |

## 系统要求

- ROS1（Noetic 或 ROS-O）、CUDA GPU（Jetson Orin 可用）
- apt：`ros-<distro>-imu-filter-madgwick ros-<distro>-pcl-ros ros-<distro>-pybind11-catkin ros-<distro>-cv-bridge`
- Python（pip，用户级）：`cupy-cuda11x`/`cupy-cuda12x`（与本机 CUDA 匹配，**版本须 <14**）、
  PyTorch（CUDA 版）、`numpy<2`、`scipy`、`shapely<2`
- 主机网口需配置雷达网段 IP `192.168.1.50/24`（与 `bringup/config/MID360_single.json` 的 host ip 一致）

## 编译

在 R300 仓库根目录（本仓库即 catkin 工作空间）：

```bash
catkin_make    # 首次会顺带编译本目录全部 vendor 包，Jetson 上约 10~20 分钟
source devel/setup.bash
```

注意：`rosdep install` 不认识 vendor 包（不在 rosdistro 索引），报 unknown key 属正常，
用 `--skip-keys "livox_ros_driver2 fast_lio elevation_mapping_cupy"` 跳过。

## 运行与话题

详见 [bringup/README.md](bringup/README.md)。一句话版：

```bash
roslaunch single_lidar_elevation single_lidar_elevation.launch
rostopic hz /elevation_mapping/elevation_map_raw   # ~5Hz 即通
```
