# V3覆盖与编译

这个版本只覆盖：

```text
src/R300/r300_autonomous_parking/
```

## Windows覆盖

```powershell
$repo = "D:\boshixiangmu\R300小车\支域\ZJY_Web_Obstacle_avoidance"
$zip  = "D:\boshixiangmu\R300小车\支域\R300_independent_autonomous_parking_v3_overlay.zip"

Expand-Archive -LiteralPath $zip -DestinationPath $repo -Force
git -C $repo status --short
```

加入Git：

```powershell
cd "D:\boshixiangmu\R300小车\支域\ZJY_Web_Obstacle_avoidance"
git add src/R300/r300_autonomous_parking

git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/autonomous_parking_node.py
git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/parking_perception_node.py
git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/start_autonomous_parking.sh
git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/stop_autonomous_parking.sh
```

## 工控机编译

本版本新增自定义消息，不能只运行旧的 `devel` 文件，必须重新编译：

```bash
cd ~/r300_ws
source /opt/ros/noetic/setup.bash
chmod +x src/R300/r300_autonomous_parking/scripts/*.py
chmod +x src/R300/r300_autonomous_parking/scripts/*.sh
catkin_make --pkg r300_autonomous_parking
source devel/setup.bash
```

检查消息：

```bash
rosmsg show r300_autonomous_parking/ParkingTarget
```

## 启动和验证

```bash
rosrun r300_autonomous_parking start_autonomous_parking.sh

rostopic hz /one_x/fix
rostopic hz /one_x/heading_deg

rosservice call /subject1/autonomous_parking/prepare "{}"
rosservice call /subject1/autonomous_parking/start_empty_search "{}"

rostopic echo /subject1/autonomous_parking/stable_count
rostopic echo /subject1/autonomous_parking/parking_target
```

注意：如果 `/one_x/fix` 仍为纬度0、经度0，节点会拒绝生成绝对停车坐标。
