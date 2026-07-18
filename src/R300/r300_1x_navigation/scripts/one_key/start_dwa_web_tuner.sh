#!/usr/bin/env bash
set -Eeuo pipefail

WS="${R300_WS:-$HOME/r300_ws}"
HOST="${WEB_HOST:-127.0.0.1}"
PORT="${WEB_PORT:-8090}"
RVIZ="${RVIZ:-false}"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

cat <<EOF
[INFO] 启动 R300 DWA Web 仿真实验台
[INFO] 本模式不启动惯导、GPS、相机、YOLO或底盘，禁止同时连接实车控制链路。
[INFO] Web 监听：$HOST:$PORT

VSCode Remote-SSH 使用方法：
  1. 打开 VSCode 的“端口/Ports”面板；
  2. 转发远端端口 $PORT；
  3. 浏览器打开 http://127.0.0.1:$PORT

普通 SSH 也可使用：
  ssh -L $PORT:127.0.0.1:$PORT explorer@工控机地址
EOF

roslaunch r300_1x_navigation subject1_dwa_web_sim.launch \
  web_host:="$HOST" \
  web_port:="$PORT" \
  use_rviz:="$RVIZ"
