# R300 独立自主泊车 V3

本包只修改 `src/R300/r300_autonomous_parking`，不修改现有控制、雷达、视觉、Web 或底盘代码。

## 1X输入

```text
/one_x/fix         sensor_msgs/NavSatFix   当前车辆经纬度
/one_x/heading_deg std_msgs/Float64        北0°、东90°、顺时针为正
```

默认拒绝 `/one_x/fix` 的 `(0,0)` 全零样帧。位置和航向必须在最近 0.75 秒内更新。

## 三类停车目标

```text
park             = 8
parking_occupied = 13
parking_empty    = 14
```

`parking_empty` 是唯一停车目标。连续 10 帧高置信、深度有效、位置无跳变且不与 `parking_occupied` 冲突后，冻结白框中心。

## 相机中心点到经纬度

停车视觉输出相机光学坐标：

```text
x：向右
y：向下
z：向前
```

先固定转换为车体 FRD：

```text
forward = z
right   = x
down    = y
```

安装角误差和杆臂当前全部为 0：

```yaml
camera_mount_roll_error_deg: 0.0
camera_mount_pitch_error_deg: 0.0
camera_mount_yaw_error_deg: 0.0
camera_lever_forward_m: 0.0
camera_lever_right_m: 0.0
camera_lever_down_m: 0.0
```

1X 航向为 `H` 时：

```text
East  = forward*sin(H) + right*cos(H)
North = forward*cos(H) - right*sin(H)
```

再以当前 `/one_x/fix` 为局部原点，使用 WGS-84 ENU/ECEF 转换得到白框中心的绝对经纬度。

## 输出话题

### 统一停车目标

```text
/subject1/autonomous_parking/parking_target
消息：r300_autonomous_parking/ParkingTarget
```

示例：

```yaml
park: 1
latitude: 39.xxxxxxxxxx
longitude: 117.xxxxxxxxxx
altitude: 12.3
confidence: 0.91
stable_frames: 10
camera_x_right_m: 0.12
camera_y_down_m: 0.85
camera_z_forward_m: 4.20
east_offset_m: 0.12
north_offset_m: 4.20
```

同时保留兼容话题：

```text
/subject1/autonomous_parking/target_fix   sensor_msgs/NavSatFix
/subject1/autonomous_parking/target_pose  geometry_msgs/PoseStamped
```

## 主流程

```text
IDLE
 -> PREPARING：检查相机；未开启则启动D435i；启动独立停车视觉
 -> SEARCH_PARK：确认park
 -> SEARCH_EMPTY/STABILIZING：寻找parking_empty并连续稳定10帧
 -> 发布parking_target（park=1 + 纬度 + 经度）
 -> NAVIGATING：move_base + DWA驶向同一停车中心
 -> VERIFYING
 -> FINISHED
```

节点不直接发布 `cmd_vel`。

## 编译

由于新增了 `ParkingTarget.msg`，必须重新编译并重新 source：

```bash
cd ~/r300_ws
catkin_make --pkg r300_autonomous_parking
source devel/setup.bash
```

## 查看输入和输出

```bash
rostopic echo /one_x/fix
rostopic echo /one_x/heading_deg
rostopic echo /subject1/autonomous_parking/stable_count
rostopic echo /subject1/autonomous_parking/parking_target
```

## 服务

```bash
rosservice call /subject1/autonomous_parking/prepare "{}"
rosservice call /subject1/autonomous_parking/start "{}"
rosservice call /subject1/autonomous_parking/start_empty_search "{}"
rosservice call /subject1/autonomous_parking/reset "{}"
```
