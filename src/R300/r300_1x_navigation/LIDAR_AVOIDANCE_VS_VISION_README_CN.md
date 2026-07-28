# 雷达高程图避障 vs 原视觉避障 —— 对照说明

> 2026-07-27 方案A实施。一句话：把 local costmap 快照层的**观测源**从
> "相机检测→虚拟激光"换成"高程图→虚拟激光"，下游 VisionSnapshotLayer /
> InflationLayer / move_base / DWA / waypoint_executor **一行未改**。

## 1. 链路对照

```
原（视觉）:
  RealSense → 双YOLO(4Hz,12类) → /r300_vision/detections
    → vision_obstacle_layer_node.py（语义白名单+bbox深度反投影）
    → /r300_vision/obstacle_scan（LaserScan, ±27°, base_link, 10Hz）
    → VisionSnapshotLayer(TTL 3s) + Inflation → local costmap → DWA(≤2.5m/s)

新（雷达）:
  MID360 → FAST-LIO → elevation_mapping_cupy
    → /elevation_mapping/elevation_map_raw（GridMap, odom(FAST-LIO树), 5Hz）
    → lidar_obstacle_scan_node.py（v3 地形算法 + 固定外参换帧）
    → /r300_lidar/obstacle_scan（LaserScan, ±60°, base_link, 5Hz）
    → VisionSnapshotLayer(同一插件,实例名 lidar_snapshot_layer) + Inflation
    → local costmap → DWA(≤1.5m/s, 限速原因见 §6)
```

## 2. 文件清单

| 新增 | 作用 | 镜像自 |
|---|---|---|
| `scripts/lidar_obstacle_scan_node.py` | 高程图→障碍格→虚拟 LaserScan | `vision_obstacle_layer_node.py`（接口）+ mevius2 v3（算法） |
| `config/subject1_lidar_obstacles.yaml` | 节点参数（外参/阈值/扫描布局） | `subject1_vision_obstacles.yaml` |
| `config/subject1_local_costmap_lidar.yaml` | 快照层订阅雷达 scan | `subject1_local_costmap_vision.yaml`（仅 topic/实例名不同） |
| `config/subject1_dwa_lidar.yaml` | 最高速度 2.5→1.5 | `subject1_dwa_vision.yaml`（仅限速不同） |
| `launch/subject1_lidar_avoidance.launch` | 起适配节点 | `subject1_vision_avoidance.launch` |
| `launch/subject1_waypoint_lidar_nav.launch` | **一键替换入口** | `subject1_waypoint_vision_nav.launch` |

不动的：VisionSnapshotLayer(C++插件)、move_base/全局代价地图/waypoint_executor、
视觉检测链（`/r300_vision/detections` 继续服务目标识别/快照记录，与避障解耦）。

## 3. v3 算法是什么（本次移植的核心）

v3 = mevius2 项目 `elevation_obstacle_node.py` 第三代地形障碍提取算法，
在另一台机器人上真机迭代出来的（v2 曾在新场地爆 6000+ 假负障碍把机器人
"活埋"，v3 是对抗审查后的修正版）。三个核心思想：

1. **局部地面自参考**：不信任何绝对 z（TF 高度/固定地面值都会被里程计 z
   漂移和标定误差坑），用"机器人周围 0.6~1.6m 环带的高程中位数"当地面
   基准——车紧邻的区域必然是真地面，天然自校准。环带数据不足时退回
   "雷达高度 − 离地 0.48m"的几何估计并告警。
2. **三通道判定**（都相对局部地面 h）：
   - 正障碍：h > +0.15m 且 < 1.5m（墙/人/桶；>1.5m 视为横梁不拦路）
   - 负障碍：h < −0.15m（坑/沟——雷达看得见坑沿，标虚拟墙挡住入口）
   - 陡坡：24cm 基线中心差分坡度 > 30°（近地面才算，墙面归正障碍通道）
3. **世界格 N 帧去抖**：障碍格按世界坐标索引计数，负障碍连续 3 帧
   (0.6s)/正障碍 2 帧才确认，单帧噪声进不了 costmap。

本次移植的改动：删掉了 v3 的 `/move_base/clear_costmaps` 大规模撤销清理
（VisionSnapshotLayer 每障碍 TTL 3s 自动过期，不需要也不该整图清）；输出
从两路 PointCloud2 改为单路虚拟 LaserScan（对齐本车现有接口）。

## 4. 坐标系设计（两棵 odom 树的红线）

高程图活在 FAST-LIO 的 `odom→camera_init→body`，导航活在 1X 的
`odom→base_link`，**同名 odom、不同原点，禁止跨树查 TF**（bringup README
"共跑必读"）。本节点的合法走法：

1. 只在 FAST-LIO 树内查 `odom→body` 得雷达位姿；
2. 用**固定安装外参**（不是 TF）把车辆水平位姿和障碍格换算成车体系相对坐标；
3. 输出 LaserScan 标 `frame_id=base_link`，VisionSnapshotLayer 收到后用
   1X 自己的 TF 落到它的 odom 保持——两树全程无交叉。

外参只需 3 个平面量（俯仰 −42.3°/横滚 −2.1°/离地 0.48m 已被"重力对齐 +
局部地面自参考"消化，无需配置）：

| 参数 | 当前值 | 依据 |
|---|---|---|
| ext_x | **0.40m** | 用户口述"雷达装在车头中间"，车长 0.9m、base_link 在中心的估计值 |
| ext_y | 0.0 | 横向居中 |
| ext_yaw_deg | 0.0 | 朝前安装 |

精化方法：车正前方 3m 放一个箱子，对比 costmap 中障碍距离与卷尺实测，
差值直接加到 ext_x；左右对称位置各放一个验证 ext_y/ext_yaw。±0.1m 误差
被膨胀半径 0.65m 覆盖，避障够用。

## 5. 接口契约逐项对照

| 契约项 | 视觉链 | 雷达链 | 说明 |
|---|---|---|---|
| 话题 | /r300_vision/obstacle_scan | /r300_lidar/obstacle_scan | costmap yaml 一行切换 |
| frame_id | base_link | base_link | 同 |
| 空束编码 | inf | inf | 快照层只收有限值 |
| 角度范围 | ±27°（D435i 视场） | ±60°（楔形视场内） | 侧向覆盖 ×2.2 |
| 束宽 | 0.5° | 0.5° | 同 |
| 量程 | 0.5~10m | 0.6~5.5m 标记（scan 上限 10.5） | 5.5=高程图有效半径 |
| 频率 | 10Hz | ~5Hz（随高程图） | 满足快照层 0.5s 新鲜度 |
| 安全余量 | 距离减 0.2m | 距离减 0.15m | |
| 断流行为 | 0.6s 停发→TTL 清空 | 2s 告警+停发→TTL 清空 | fail-safe 同构 |
| 快照层参数 | hold 3s/聚类0.45/关联0.5 | 完全相同 | 同一插件不同实例名 |

## 6. 与原逻辑的行为差异（利弊）

**多出来的能力**：
- 任何几何障碍都能避（视觉只有 6 个白名单类别，没训练过的=隐形）；
- **负障碍（坑/沟）**——视觉链明确把 trench 放进忽略名单，雷达链是唯一防线；
- 夜间/逆光/雨天不退化；4cm 格网几何精度；±60° 视场。

**失去/退化的能力**：
- **语义**：烟雾（smoke）视觉可识别并故意穿过，雷达会当成实体障碍拦停——
  场上有烟雾墙时需保留 YOLO 并做扇区豁免（待办）；同理 trench/puddle
  的"故意不避"语义不再存在，坑会被拦（这通常是想要的，但要知道）。
- **延迟**：高程图链端到端 1.0~1.8s（含卡尔曼平滑），负障碍再加 0.6s
  去抖；视觉链约 0.5s。因此 DWA 最高速度先压到 **1.5m/s**
  （subject1_dwa_lidar.yaml），后续加 /cloud_registered_body 快反层再回调。
- 低于 15cm 的矮小障碍两条链都看不见（阈值使然）。

## 7. 启动 / 回切

```bash
# 启动（顺序）：
# ① 1X 定位:  roslaunch r300_1x_navigation one_x_localization_only.launch
# ② 雷达感知: bash scripts/start_single_lidar_elevation.sh   (或 web"启动雷达"按钮)
# ③ 雷达避障导航（替代原视觉版 subject1_waypoint_vision_nav.launch）:
roslaunch r300_1x_navigation subject1_waypoint_lidar_nav.launch

# 回切视觉版（一条命令，互不干扰）:
roslaunch r300_1x_navigation subject1_waypoint_vision_nav.launch
```

验证要点：`rostopic hz /r300_lidar/obstacle_scan` ≈5Hz；rviz 看
`/r300_lidar/obstacle_points`（base_link 系调试点云）与实物方位/距离一致；
local costmap 中障碍出现且 3s TTL 过期正常；正前 3m 纸箱→DWA 绕行。

## 8. 待办

1. 外参精化（§4 方法，当前 ext_x=0.40 为估计值）；
2. 烟雾扇区豁免（保留 YOLO 低频检测 smoke，对应扇区束置 inf）；
3. /cloud_registered_body 10Hz 快反层（压延迟后回调限速）；
4. GPU 叠加实测（高程图 58-60% GR3D + 若同时跑双 YOLO）；
5. 一键脚本（现 start_r300_vision_nav.sh 的门检等视觉话题，雷达版需另写）。
