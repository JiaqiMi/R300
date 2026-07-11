# 项目说明



# 魔法指令

#### 项目编译

cd ~/r300_ws

rm -rf build devel

source /opt/ros/noetic/setup.bash

catkin_make 2>&1 | tee build_log.txt


## 视觉侧指令

#### 独立编译

catkin_make --pkg r300_vision_msgs


#### 启动相机

roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  enable_sync:=true

#### 启动深度模型识别

roslaunch \
r300_yolo_detector \
yolo_depth.launch


#### 查看标注图像

rqt_image_view

选择话题：/r300_vision/annotated_image


#### 录制标注图像和检测结果

// Step 1: 录制bag信息
rosbag record \
  -O yolo_test_$(date +%Y%m%d_%H%M%S).bag \
  /r300_vision/annotated_image \
  /r300_vision/detections \
  /r300_vision/target_point

// Step 2: 查看bag信息
rosbag info ~/r300_records/yolo_test_20260711_183000.bag

// Step 3: 回放bag信息, 需要开启新核
rosbag play yolo_test_20260711_183000.bag --loop -r 0.5

// Step 4: 需要开启画面，选择合适的话题
rqt_image_view


#### Web发布

rosrun web_video_server web_video_server \
  _port:=8080 \
  _address:=192.168.1.107 \
  _server_threads:=2 \
  _ros_threads:=2

// 查询工控机IP
hostname -I

// 浏览器中直接访问： http://127.0.0.1:8080/