# 独立自主泊车覆盖说明

这个覆盖包**只新增**：

```text
src/R300/r300_autonomous_parking/
```

不会覆盖或修改控制、雷达、视觉、Web、底盘等现有文件夹。

## Windows 本地覆盖

在仓库上一级目录执行：

```powershell
$repo = "D:\boshixiangmu\R300小车\支域\ZJY_Web_Obstacle_avoidance"
$zip  = "D:\boshixiangmu\R300小车\支域\R300_independent_autonomous_parking_v2_overlay.zip"

Expand-Archive -LiteralPath $zip -DestinationPath $repo -Force

git -C $repo status --short
```

正常只应看到：

```text
?? src/R300/r300_autonomous_parking/
```

加入 Git 并设置 Linux 可执行权限：

```powershell
cd "D:\boshixiangmu\R300小车\支域\ZJY_Web_Obstacle_avoidance"

git add src/R300/r300_autonomous_parking

git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/autonomous_parking_node.py
git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/parking_perception_node.py
git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/start_autonomous_parking.sh
git update-index --chmod=+x src/R300/r300_autonomous_parking/scripts/stop_autonomous_parking.sh
```

## 工控机编译

```bash
cd ~/r300_ws
chmod +x src/R300/r300_autonomous_parking/scripts/*.py
chmod +x src/R300/r300_autonomous_parking/scripts/*.sh
catkin_make --pkg r300_autonomous_parking
source devel/setup.bash
```

## 启动前模型检查

```bash
ls -lh ~/r300_ws/src/R300_vision/r300_yolo_detector/models/model_c6_260723.pt
ls -lh ~/r300_ws/src/R300_vision/r300_yolo_detector/models/parking_status_c3.pt
```

第二个三分类模型必须补齐，或通过环境变量指定：

```bash
export PARKING_MODEL2_PATH=/绝对路径/parking_status_c3.pt
```

## 安全测试顺序

```bash
# 1. 启动独立泊车节点
rosrun r300_autonomous_parking start_autonomous_parking.sh

# 2. 只检查/启动相机和视觉，不动车
rosservice call /subject1/autonomous_parking/prepare "{}"

# 3. 查看三类检测结果
rostopic echo /r300_parking_vision/detections

# 4. 查看 empty 连续帧计数
rostopic echo /subject1/autonomous_parking/stable_count

# 5. 车辆放在开阔区域并准备急停后，直接测试 empty 搜索
rosservice call /subject1/autonomous_parking/start_empty_search "{}"

# 6. 随时取消
rosservice call /subject1/autonomous_parking/reset "{}"
```
