# R300 即时自主泊车

本包与现有导航、雷达、视觉和底盘代码隔离。它不发布 `cmd_vel`，不修改 DWA 参数，只把确认后的停车点交给现有 `/move_base`。

## 唯一触发逻辑

节点启动后自动进入 `ARMED`：

1. 订阅 `/r300_parking_vision/detections`；
2. 只接受 `class_name=parking_empty`；
3. 每帧置信度必须 `>=0.20` 且深度有效；
4. 连续 5 个不同检测帧均满足要求；
5. 第 5 帧立即发布目标并发送 `move_base`。

不再依赖：

- `/subject1/waypoint_status`；
- `park` 类确认；
- `parking_occupied` 冲突门控；
- 手动 `/start` 服务；
- 车辆静止判断；
- 旋转搜索；
- dynamic_reconfigure 或泊车专用规划器参数。

## 输出

达到 5 帧后，以下三个话题均为锁存话题：

```text
/subject1/autonomous_parking/parking_target
/subject1/autonomous_parking/target_fix
/subject1/autonomous_parking/target_pose
```

`parking_target` 中：

```text
park=1
stable_frames=5
confidence>=0.20
latitude/longitude=停车位中心绝对经纬度
```

## 编译

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg r300_autonomous_parking
source devel/setup.bash
```

新增/修改过 `ParkingTarget.msg` 时，必须重新编译。

## 启动

```bash
conda activate yolo26
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rosrun r300_autonomous_parking start_autonomous_parking.sh
```

成功准备后状态应为：

```text
state=ARMED stable=0/5
```

识别过程：

```text
state=COUNTING stable=1/5
...
state=COUNTING stable=5/5
state=TARGET_PUBLISHED
state=NAVIGATING
```

观察：

```bash
rostopic echo /subject1/autonomous_parking/state
rostopic echo /subject1/autonomous_parking/stable_count
rostopic echo /subject1/autonomous_parking/parking_target
rostopic echo /move_base/goal
```

取消并重新布防：

```bash
rosservice call /subject1/autonomous_parking/reset "{}"
```

## 必要运行条件

- `/one_x/fix` 为非零有效经纬度；
- `/one_x/heading_deg` 持续更新；
- `map -> base_link` TF 可用；
- `/move_base` action server 可用；
- `parking_empty` 检测必须带有效深度与相机三维坐标。

相机与惯导安装误差、杆臂均按 0 处理；只保留光学坐标系到车体坐标定义所必需的轴转换。
