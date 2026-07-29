# 科目一简化自主泊车说明

## 已实现流程

```text
全部 GPS 航点完成
  -> 原地间歇旋转搜索 park 类别
  -> move_base 靠近 P 牌
  -> 原地间歇旋转搜索 parking_slot 类别
  -> 使用 vehicle 类别排除占用车位
  -> 选择当前视野中最近的空车位
  -> move_base 前进驶入车位
  -> 发布停车完成状态
```

本版本不做倒车。搜索阶段才直接发布低速角速度，前进过程仍由
`move_base + DWA + 雷达局部代价地图` 完成。

## YOLO 类别

当前使用：

- 停车指示牌：`park`
- 障碍车辆：`vehicle`
- 白色停车框：`parking_slot`

白框模型训练完成后，必须确认模型类别名称与
`config/subject1_parking.yaml` 中的 `slot_classes` 一致。
双 YOLO 节点会锁存发布 `/r300_vision/available_classes`。只有该列表同时
包含 `park`、`parking_slot`、`vehicle`，泊车节点才允许启动；当前模型若
没有 `parking_slot`，一键脚本会明确报错并停止，不会让车辆旋转等待。

## 自动触发

`parking_manager.py` 直接监听已有的：

```text
/subject1/waypoint_status
```

当其中出现 `state=COMPLETED` 时自动启动，因此没有修改
`waypoint_executor.py`，也没有新增 `mission_finished` 话题。

## 主要话题和服务

```text
订阅：
/r300_vision/detections
/r300_vision/available_classes
/subject1/waypoint_status
/subject1/dwa_odom

发布：
/subject1/cmd_vel_raw              # 只在搜索时低速旋转
/subject1/parking/state
/subject1/parking/done
/subject1/parking/selected_goal

服务：
/subject1/parking/start            # 台架测试时手动启动
/subject1/parking/reset            # 复位后允许再次测试
```

## 启动

雷达导航脚本默认开启停车，并按需启动双 YOLO。可通过
`--vision-model1`、`--vision-model2` 指定现场模型：

```bash
rosrun r300_1x_navigation start_r300_lidar_nav.sh --run
```

临时关闭停车：

```bash
rosrun r300_1x_navigation start_r300_lidar_nav.sh --no-parking --run
```

## 建议测试顺序

### 1. 编译

```bash
cd ~/R300
catkin_make
source devel/setup.bash
```

### 2. 检查节点

```bash
rosnode list | grep parking_manager
rostopic echo /subject1/parking/state
```

初始状态应为：

```text
state=WAIT_MISSION
```

### 3. 不跑完整航点，手动测试搜索

保证车辆周边安全并架空或留足旋转空间，然后：

```bash
rosservice call /subject1/parking/start
```

车辆开始间歇旋转搜索。停止并复位：

```bash
rosservice call /subject1/parking/reset
```

### 4. 检查模型类别

```bash
rostopic echo /r300_vision/detections
```

应能看到：

```text
class_name: "park"
class_name: "parking_slot"
class_name: "vehicle"
```

## 现场优先调整参数

文件：

```text
config/subject1_parking.yaml
```

优先调整：

- `park_min_confidence`：P 牌误检多时提高。
- `slot_min_confidence`：白框漏检多时降低。
- `search_angular_speed_radps`：搜索转速。
- `park_sign_standoff_m`：靠近 P 牌的距离。
- `slot_goal_extension_m`：车辆进入白框的深度。
- `occupied_distance_m`：车辆与白框中心的占用距离阈值。

## 注意事项

1. 必须先设置 1X 原点并保证 `map -> odom -> base_link` 可用。
2. RealSense 没有 camera TF，本节点使用 YAML 外参把相机坐标转换为
   `base_link`，然后再由 TF 转换到 `map`。
3. 默认相机外参是前方 0.30 m、上方 0.10 m；实际安装位置不同必须修改。
4. 当前 DWA 的 `xy_goal_tolerance` 较大，因此使用
   dynamic_reconfigure 在泊车期间临时切换为低速和较小容差，流程结束后恢复。
5. 车位是否占用采用三种简单判定：检测框中心、检测框重叠、三维距离；
   任意一项满足即判定占用，并要求连续多帧为空。
6. 驶入前必须通过 `/move_base/make_plan`，`SUCCEEDED` 后还要连续验证最终
   位置、朝向和速度，验证通过才发布停车完成。
