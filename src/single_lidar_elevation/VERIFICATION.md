# single_lidar_elevation 模块验证记录

日期：2026-07-21。验证环境：Jetson Orin NX 16GB（JetPack 6.2 / Ubuntu 22.04 / CUDA 12.6），
ROS-O（/opt/ros/one）、cupy-cuda12x 13.6、PyTorch 2.5（CUDA 版）。
**当时雷达未开机**，因此运行验证 = 全栈干跑 + 合成点云端到端测试；真机雷达联通后按文末 checklist 复验。

## 结论（TL;DR）

| 项目 | 结果 |
|---|---|
| 静态审查（6 维度多智能体对抗核查） | 通过，无 blocker |
| 编译（catkin_make，11 包白名单） | 通过，exit 0，全部产物就位 |
| 全栈干跑（无雷达） | 6 节点全部存活，话题全部就位 |
| 合成点云端到端 | **PASS**：高程图 5.000Hz，地面中位 -0.500m（期望 -0.5），台阶 -0.276m / 67 格（期望 ≈-0.3、≥30 格） |

## 1. 交付内容

`src/single_lidar_elevation/` 下 11 个 catkin 包（bringup + 全部 vendored 依赖），
来源与本地改动清单见 [README.md](README.md)。配置文件为上游感知栈**真机验证配置的冻结拷贝**
——与已双雷达实测通过的 elevation/fastlio 配置逐项 diff，参数键值完全一致，仅注释措辞不同。

## 2. 静态审查结论

六个维度（bringup launch 链路、bringup 包、livox 驱动+SDK、FAST_LIO、emcupy+grid_map、工作空间集成）
并行审查，重要发现逐条对抗核实。要点：

- **Livox SDK 离线化**：驱动 CMake 的 ExternalProject 已从 git 源改为 `URL ../livox_sdk2`，
  全部 CMake 无任何网络引用残留；vendored SDK 与上游 v1.2.2 逐文件哈希比对，
  唯一差异是预打的 spdlog 降噪补丁（`vendor_patches/spdlog-quiet.patch`，logging.cpp 日志级别 → warn）。
- **FAST_LIO**：除约定排除项（论文/素材/示例）外与已知良好版本逐字节一致；
  fork 专属话题 `/cloud_registered_body`（laserMapping.cpp:894，受 `scan_bodyframe_pub_en:true` 门控）、
  `/Odometry_precede`（:903）发布代码确认在位，与高程图订阅闭环。
- **emcupy / grid_map**：与已验证工作空间逐字节一致；boost join 头文件补丁、
  六个 grid_map 子包的 c++17 补丁全部在位；依赖闭包在【vendored 包 + ROS 标准 apt 包】内封闭。
- **工作空间集成**：新增 11 包与 R300 既有 20 包无重名冲突；vendored 包不引用
  move_base/costmap，与 R300 的 move_base fork 无关联。

### 本次审查中发现并修复的问题

1. **`.gitignore` 的 `core.*` 模式误杀 SDK 必需头文件**
   `livox_sdk2/3rdparty/spdlog/spdlog/fmt/bundled/core.h` —— 不修则 fresh clone 编译必失败。
   已在根 `.gitignore` 追加负模式 `!**/spdlog/fmt/bundled/core.h`，并经
   `git check-ignore` / 未跟踪文件清点双重验证：模块全部文件中仅
   `grid_map_core/doc/` 两个纯文档 PDF 被有意排除（瘦身），其余全部入库。
2. **bringup 两脚本 shebang `python` → `python3`**（Ubuntu 22.04 无 `python` 命令时节点起不来）。
3. **bringup 脚本补可执行位**（rosrun/roslaunch 从源树执行需要）。
4. **补齐 `lio_input_crop` 裁剪链路**（custom_msg_crop.py + crop_front.yaml + launch 开关，默认关）
   ——参照系单雷达形态强烈建议的选项，车体结构进入雷达视场时的唯一有效过滤手段。
   注意 `crop_front.yaml` 是空模板，启用前必须在真机 rviz 里按传感器系实测标定裁剪盒。
5. **bringup package.xml 补 `rviz` / `grid_map_rviz_plugin` / `python3-numpy` exec_depend**；
   **`grid_map/` 补上游 BSD-3-Clause LICENSE**（推 GitLab 分发的合规要求）。

已知无害项（记录备查）：elevation_mapping_cupy 的 setup.py 未声明 `.kernels/.fusion` 子包
（上游缺陷，仅影响 install-space 部署，devel-space 正常）；其 launch/ 下两个 turtlesim 示例
引用了未 vendor 的 semantic_sensor（不被本模块 launch 引用）；`src/CMakeLists.txt` 是指向
`/opt/ros/noetic/.../toplevel.cmake` 的绝对路径软链（仓库既有状态，Noetic 机器直接可用；
ROS-O 等其他前缀的机器 `rm src/CMakeLists.txt` 后再 catkin_make 会自动重建）。

## 3. 编译验证

```bash
cd ~/R300
rm -f src/CMakeLists.txt   # 仅 ROS-O 机器需要（软链指向 /opt/ros/noetic，悬空）
source /opt/ros/one/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="livox_ros_driver2;grid_map_core;grid_map_msgs;grid_map_cv;grid_map_sdf;grid_map_ros;grid_map_rviz_plugin;elevation_map_msgs;elevation_mapping_cupy;fast_lio;single_lidar_elevation" -j6
```

结果：exit 0；`devel/lib/fast_lio/fastlio_mapping`（74MB）、`devel/lib/livox_ros_driver2/livox_ros_driver2_node`、
grid_map 全套 .so、elevation_mapping python 包全部生成。
（白名单只为不动仓库里其他包；整仓编译按根 README 的黑名单说明取舍。
用过白名单后想恢复整仓编译：`catkin_make -DCATKIN_WHITELIST_PACKAGES=""`。）

## 4. 运行验证

### 4.1 全栈干跑（雷达未开机）

`roslaunch single_lidar_elevation single_lidar_elevation.launch` 后：

- 6 节点全部存活：`livox_lidar_publisher2`、`imu_filter`、`livox_imu_transform`、
  `laserMapping`、`livox_odom_transform`、`elevation_mapping`；
- 全链路话题就位：`/livox/lidar_192_168_1_192`、`/livox/imu_192_168_1_192`、`/livox/imu_filtered`、
  `/cloud_registered(_body)`、`/Odometry(_precede)`、`/elevation_mapping/elevation_map_raw`、`.../elevation_map_filter`；
- 日志中 `Still waiting for data on topic /livox/imu_...` 与 `"odom" ... does not exist`
  为无雷达时的**预期输出**（无 IMU → 重力对齐 TF 未发布），雷达上电后自行消失；
- elevation 节点能走到 TF 查询循环，说明 CuPy/PyTorch/权重加载全部成功；
- `lio_input_crop:=true` 形态另测：8 节点存活（多出 `/front_crop`），
  `/common/lid_topic` 正确切换为 `livox/lidar_front_cropped`。

### 4.2 合成点云端到端（无雷达验证高程图链路）

方法（复现步骤）：launch 起全栈后，补 3 条恒等静态 TF 并向 `/cloud_registered_body`
灌合成点云（-0.5m 地面 + (1.5,0) 处 0.2m 台阶，10Hz，frame=test_lidar）：

```bash
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 odom camera_init &
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 camera_init body &
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 body test_lidar &
# 合成云/判定脚本在上游 mevius2 仓库 mevius2_perception/scripts/（Jetson: ~/perception_ws）
python3 synthetic_cloud_pub.py /synthetic_cloud:=/cloud_registered_body &
python3 map_check.py
```

结果：**PASS** —— `/elevation_mapping/elevation_map_raw` 稳定 **5.000Hz**（std dev 0.2ms）；
有效格 22724；地面中位 **-0.500m**（期望 ≈-0.5）；台阶最高 **-0.276m**、67 格（期望 ≈-0.3、≥30 格）。
证明"点云 → GPU 高程图 → elevation/traversability 层 → 5Hz 发布"链路与包内配置完全可用。

## 5. 真机复验 checklist（雷达开机后）

1. 前提：雷达 IP 192.168.1.192，主机网口配 192.168.1.50/24，能 ping 通雷达；
2. `roslaunch single_lidar_elevation single_lidar_elevation.launch`；
3. **启动后 10 秒内勿碰车、勿站雷达正前**（FAST-LIO 启动瞬态发散不可自愈，发散就重启 launch）；
4. 依次 `rostopic hz`：`/livox/lidar_192_168_1_192` ≈10Hz → `/cloud_registered_body` ≈10Hz →
   `/elevation_mapping/elevation_map_raw` ≈5Hz（重力对齐 TF 在 IMU 起流后约 3 秒发布）；
5. `rostopic echo /Odometry -n1` 静止漂移应为厘米级；
6. 与既有导航栈共跑前，务必读 [bringup/README.md](bringup/README.md) 的
   base_link 双父帧与 odom 同名合并陷阱两节；
7. 车体结构若进入雷达视场：先在 rviz 按传感器系标定 `bringup/config/crop_front.yaml`
   裁剪盒，再以 `lio_input_crop:=true` 启动（空盒状态下开裁剪等于没裁）；
8. 若测试机上还有其他含同名包的工作空间（如 `~/perception_ws`），只 source
   本仓库的 `devel/setup.bash`，不要叠加 source——同名包会造成 `$(find)`
   与消息定义解析歧义。
