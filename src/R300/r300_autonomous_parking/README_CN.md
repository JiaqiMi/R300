# R300 独立自主泊车

本包只新增 `src/R300/r300_autonomous_parking`，不修改以下现有代码：

- `r300_1x_navigation` 控制与航点代码；
- 雷达避障与代价地图代码；
- `R300_vision/r300_yolo_detector` 原视觉代码；
- `scout_base` 和底盘驱动。

泊车包内部包含一份独立的停车视觉节点和一份独立的泊车状态机。它只复用现有 ROS 接口、模型文件、`r300_vision_msgs` 消息、`move_base` 和 DWA。

## 三类目标

| 类别 | 全局 ID | 用途 |
|---|---:|---|
| `park` | 8 | 确认进入停车区域，默认连续 3 帧 |
| `parking_occupied` | 13 | 占用冲突，不能作为目标 |
| `parking_empty` | 14 | 唯一允许生成停车目标的类别 |

## 主流程

```text
IDLE
  -> PREPARING：检查相机；未开启则启动 D435i；启动独立停车视觉
  -> SEARCH_PARK：通过 move_base 分步旋转，连续确认 park
  -> SEARCH_EMPTY / STABILIZING：旋转搜索 parking_empty
  -> parking_empty 高置信、连续 10 帧、无跳变且不与 occupied 冲突
  -> 冻结 10 帧 map 坐标中位数
  -> 发布 map 目标和 WGS-84 经纬度
  -> NAVIGATING：使用 move_base + DWA 低速驶向目标
  -> VERIFYING：距离、线速度、角速度持续满足阈值
  -> FINISHED
```

泊车节点从不直接发布 `cmd_vel`。旋转搜索和最终行驶均通过 `/move_base`，启动时若正常航点正在运行，则先调用 `/subject1/pause_waypoints`，随后取消旧目标，防止两个目标发送者同时工作。

## 相机检查

检查以下三个话题是否在最近 1.5 秒内持续更新：

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
```

- 三个都存在：复用当前相机；
- 三个都不存在：自动启动 `realsense2_camera/rs_camera.launch`；
- 只有部分存在：报错，不重复打开同一台 D435i，避免 USB 设备占用冲突。

自动启动只开启 RGB、Depth、对齐、同步和 TF；关闭 D435i 自带 IMU。车辆的 1X 惯导仍然必须运行。

## 连续 10 帧稳定规则

`parking_empty` 每帧必须同时满足：

1. 类别名为 `parking_empty`；
2. 置信度不低于 `min_empty_confidence`，默认 0.60；
3. 深度有效且在 0.4–15 m；
4. 车辆处于停止状态；
5. 帧间间隔不超过 0.60 s；
6. 当前 map 坐标到已有样本中位数的跳变不超过 0.25 m；
7. 10 帧整体离散范围不超过 0.25 m；
8. 候选点 0.60 m 内没有 `parking_occupied`。

任何一帧缺失、低置信、深度无效、车辆运动或 occupied 冲突都会清零计数；目标跳变时从当前新目标重新计为第 1 帧。

## 主要话题

```text
/r300_parking_vision/detections
/r300_parking_vision/annotated_image
/subject1/autonomous_parking/state
/subject1/autonomous_parking/park_count
/subject1/autonomous_parking/stable_count
/subject1/autonomous_parking/target_pose
/subject1/autonomous_parking/target_fix
```

## 服务

```bash
# 只准备相机和视觉，不让车辆运动
rosservice call /subject1/autonomous_parking/prepare "{}"

# 完整流程：先找 park，再找 parking_empty
rosservice call /subject1/autonomous_parking/start "{}"

# 调试快捷方式：跳过 park，直接旋转搜索 parking_empty
rosservice call /subject1/autonomous_parking/start_empty_search "{}"

# 取消目标、恢复 DWA 参数
rosservice call /subject1/autonomous_parking/reset "{}"
```

## 模型

默认模型路径：

```text
model1: $(find r300_yolo_detector)/models/model_c6_260723.pt
model2: $(find r300_yolo_detector)/models/parking_status_c3.pt
```

其中 `parking_status_c3.pt` 必须实际存在。也可以通过环境变量覆盖：

```bash
export PARKING_MODEL1_PATH=/绝对路径/park模型.pt
export PARKING_MODEL2_PATH=/绝对路径/三分类停车模型.pt
```

## 启动

```bash
cd ~/r300_ws
catkin_make --pkg r300_autonomous_parking
source devel/setup.bash
rosrun r300_autonomous_parking start_autonomous_parking.sh
```
