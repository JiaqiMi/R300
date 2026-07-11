# Self-Driving Car
---

## 项目介绍


无人车挑战赛

---

## 系统介绍

pass

---

## 惯性导航系统

pass

---

## 控制系统

pass

--

## Vision System

基于 **ROS Noetic、Intel RealSense D435i、Ultralytics YOLO 和 NVIDIA Jetson GPU** 实现的目标检测与深度定位系统。

系统支持以下功能：

- RealSense D435i 彩色图像与对齐深度图采集
- YOLO 模型 GPU 推理
- 目标二维检测框发布
- 目标三维位置估计
- 检测结果可视化
- ROS bag 数据录制与回放
- Web 浏览器实时查看检测画面
- 相机、模型、Web 和 rosbag 一键启动

---

### 1. 项目结构

```text
r300_ws/
├── src/
│   └── R300_vision/
│       ├── r300_vision_msgs/
│       │   └── msg/
│       │       ├── DetectedObject.msg
│       │       └── DetectedObjectArray.msg
│       │
│       └── r300_yolo_detector/
│           ├── config/
│           ├── launch/
│           ├── models/
│           ├── scripts/
│           ├── CMakeLists.txt
│           └── package.xml
│
├── scripts/
│   └── start_r300.sh
│
├── build/
├── devel/
└── README.md
```

---

### 2. 运行环境

- Ubuntu 20.04
- ROS Noetic
- Python 3.8
- NVIDIA Jetson Orin
- CUDA 11.4
- Intel RealSense D435i
- PyTorch CUDA
- Ultralytics YOLO

---

### 3. 项目编译

进入工作空间：

```bash
cd ~/r300_ws
```

清理旧的编译结果：

```bash
rm -rf build devel
```

加载 ROS Noetic 环境：

```bash
source /opt/ros/noetic/setup.bash
```

完整编译工作空间，并保存编译日志：

```bash
catkin_make 2>&1 | tee build_log.txt
```

加载当前工作空间：

```bash
source ~/r300_ws/devel/setup.bash
```

> 建议将下面两行添加到 `~/.bashrc`，避免每次打开终端后重复执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

---

### 4. 独立编译视觉功能包

#### 4.1 编译消息包

```bash
cd ~/r300_ws
catkin_make --pkg r300_vision_msgs
```

#### 4.2 编译目标检测包

```bash
cd ~/r300_ws
catkin_make --pkg r300_yolo_detector
```

编译完成后重新加载工作空间：

```bash
source ~/r300_ws/devel/setup.bash
```

---

### 5. 分步启动系统

分步启动适合调试和排查问题。

#### 5.1 启动 RealSense D435i

打开第一个终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

启动相机，并开启深度对齐与时间同步：

```bash
roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  enable_sync:=true
```

检查相机话题：

```bash
rostopic list | grep camera
```

检查彩色图像频率：

```bash
rostopic hz /camera/color/image_raw
```

检查对齐深度图频率：

```bash
rostopic hz /camera/aligned_depth_to_color/image_raw
```

---

#### 5.2 启动目标检测与深度定位

打开第二个终端：

```bash
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

启动 YOLO 检测节点：

```bash
roslaunch r300_yolo_detector yolo_depth.launch
```

正常启动后应看到类似日志：

```text
CUDA=True
GPU: Orin
Model classes: ...
R300 YOLO depth node started
```

---

#### 5.3 查看检测结果

打开第三个终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rqt_image_view
```

在图像话题列表中选择：

```text
/r300_vision/annotated_image
```

---

### 6. 主要 ROS 话题

#### 6.1 订阅话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | D435i 彩色图像 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 对齐到彩色图的深度图 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 彩色相机内参 |

#### 6.2 发布话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/r300_vision/annotated_image` | `sensor_msgs/Image` | 带检测框和距离信息的图像 |
| `/r300_vision/detections` | `r300_vision_msgs/DetectedObjectArray` | 所有目标检测和位置结果 |
| `/r300_vision/target_point` | `geometry_msgs/PointStamped` | 选中目标的三维位置 |

查看检测结果：

```bash
rostopic echo /r300_vision/detections
```

查看选中目标位置：

```bash
rostopic echo /r300_vision/target_point
```

检查标注图像发布频率：

```bash
rostopic hz /r300_vision/annotated_image
```

---

### 7. 录制检测图像和检测结果

#### 7.1 创建记录目录

```bash
mkdir -p ~/r300_records
cd ~/r300_records
```

#### 7.2 开始录制 rosbag

```bash
rosbag record \
  -O yolo_test_$(date +%Y%m%d_%H%M%S).bag \
  /r300_vision/annotated_image \
  /r300_vision/detections \
  /r300_vision/target_point
```

录制结束时按：

```text
Ctrl + C
```

生成的文件示例：

```text
yolo_test_20260711_183000.bag
```

---

#### 7.3 查看 rosbag 信息

```bash
rosbag info ~/r300_records/yolo_test_20260711_183000.bag
```

请将文件名替换为实际生成的文件名。

查看已有记录：

```bash
ls -lh ~/r300_records
```

---

#### 7.4 回放 rosbag

先启动 ROS Master：

```bash
roscore
```

打开一个新终端，以 `0.5` 倍速循环播放：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

rosbag play \
  ~/r300_records/yolo_test_20260711_183000.bag \
  --loop \
  -r 0.5
```

---

#### 7.5 查看回放画面

再打开一个新终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rqt_image_view
```

选择话题：

```text
/r300_vision/annotated_image
```

---

### 8. Web 实时画面发布

#### 8.1 启动 Web 视频服务器

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
```

启动服务：

```bash
rosrun web_video_server web_video_server \
  _port:=8080 \
  _address:=0.0.0.0 \
  _server_threads:=2 \
  _ros_threads:=2
```

其中：

- `0.0.0.0` 表示监听所有网卡；
- `8080` 为 Web 服务端口。

不建议固定绑定某个具体 IP，否则工控机 IP 变化后可能导致服务启动失败。

---

#### 8.2 查询工控机 IP

```bash
hostname -I
```

假设工控机 IP 为：

```text
192.168.1.107
```

浏览器访问 Web 首页：

```text
http://192.168.1.107:8080/
```

直接查看检测视频流：

```text
http://192.168.1.107:8080/stream?topic=/r300_vision/annotated_image&type=mjpeg
```

低带宽模式：

```text
http://192.168.1.107:8080/stream?topic=/r300_vision/annotated_image&type=mjpeg&quality=70&width=640&height=480
```

> `127.0.0.1` 只表示当前电脑本机。  
> 在其他电脑浏览器中，应使用工控机的实际局域网 IP。

---

### 9. 一键启动

一键启动脚本位于：

```text
~/r300_ws/scripts/start_r300.sh
```

首次使用前增加执行权限：

```bash
chmod +x ~/r300_ws/scripts/start_r300.sh
```

#### 9.1 启动相机、模型和 Web

```bash
~/r300_ws/scripts/start_r300.sh web
```

启动内容：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ Web视频服务
```

---

#### 9.2 启动相机、模型和 rosbag

```bash
~/r300_ws/scripts/start_r300.sh bag
```

启动内容：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ rosbag录制
```

---

#### 9.3 同时启动 Web 和 rosbag

```bash
~/r300_ws/scripts/start_r300.sh both
```

启动内容：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ Web视频服务
+ rosbag录制
```

停止所有节点并安全结束 rosbag：

```text
Ctrl + C
```

---

### 10. 直接使用总 Launch 文件

总 launch 文件可以启动：

```text
RealSense D435i
+ YOLO目标检测
+ 深度定位
+ Web视频服务
```

启动命令：

```bash
source ~/venvs/yolo26/bin/activate
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

roslaunch r300_yolo_detector r300_system.launch
```

关闭 Web：

```bash
roslaunch r300_yolo_detector r300_system.launch \
  enable_web:=false
```

修改 Web 端口：

```bash
roslaunch r300_yolo_detector r300_system.launch \
  web_port:=8081
```

---

### 11. 坐标系说明

目标三维位置默认发布在相机光学坐标系中：

```text
camera_color_optical_frame
```

坐标方向定义如下：

| 坐标轴 | 方向 |
|---|---|
| X | 图像右侧 |
| Y | 图像下方 |
| Z | 相机正前方 |

因此：

```text
position.z
```

表示目标距离相机的前向距离。

在正式接入无人车控制系统前，建议通过 TF 将目标位置从：

```text
camera_color_optical_frame
```

转换到：

```text
base_link
```

---

### 12. 常见问题

#### 12.1 找不到 ROS 包

```bash
source /opt/ros/noetic/setup.bash
source ~/r300_ws/devel/setup.bash
rospack profile
```

检查：

```bash
rospack find r300_yolo_detector
```

---

#### 12.2 CUDA 不可用

```bash
source ~/venvs/yolo26/bin/activate

python3 - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

#### 12.3 Web 页面无法访问

检查 Web 服务是否运行：

```bash
ss -lntp | grep 8080
```

检查防火墙：

```bash
sudo ufw status
```

如有需要，开放端口：

```bash
sudo ufw allow 8080/tcp
```

---

#### 12.4 Web 页面可以打开但没有图像

检查标注图像是否正常发布：

```bash
rostopic hz /r300_vision/annotated_image
```

检查话题发布者：

```bash
rostopic info /r300_vision/annotated_image
```

---

#### 12.5 rosbag 异常中断

如果出现：

```text
xxx.bag.active
```

重新建立索引：

```bash
rosbag reindex xxx.bag.active
```

修复并生成新文件：

```bash
rosbag fix \
  xxx.bag.active \
  xxx_fixed.bag
```

---

### 13. 推荐运行方式

日常查看检测效果：

```bash
~/r300_ws/scripts/start_r300.sh web
```

正式采集实验数据：

```bash
~/r300_ws/scripts/start_r300.sh both
```

仅启动 ROS 节点进行调试：

```bash
roslaunch r300_yolo_detector r300_system.launch
```

