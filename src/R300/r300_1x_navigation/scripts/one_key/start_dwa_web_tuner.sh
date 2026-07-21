#!/usr/bin/env bash
set -Eeuo pipefail

WS="${R300_WS:-$HOME/r300_ws}"

# Web服务器监听所有网卡，允许Windows通过工控机IP访问。
HOST="${WEB_HOST:-0.0.0.0}"

# 默认使用8070端口。
PORT="${WEB_PORT:-8070}"

RVIZ="${RVIZ:-false}"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }

# 自动识别适合从Windows访问的工控机IP。
detect_access_ip() {
  local detected_ip=""

  # Remote-SSH环境下，SSH_CONNECTION第三项就是工控机
  # 接受当前SSH连接时使用的地址，优先级最高。
  if [[ -n "${SSH_CONNECTION:-}" ]]; then
    detected_ip="$(awk '{print $3}' <<< "$SSH_CONNECTION")"
  fi

  # 没有SSH_CONNECTION时，根据默认路由识别主网卡IP。
  if [[ -z "$detected_ip" ]]; then
    detected_ip="$(
      ip route get 1.1.1.1 2>/dev/null |
      awk '{
        for (i = 1; i <= NF; ++i) {
          if ($i == "src" && (i + 1) <= NF) {
            print $(i + 1)
            exit
          }
        }
      }'
    )"
  fi

  # 再退回到hostname返回的第一个非回环IPv4地址。
  if [[ -z "$detected_ip" ]]; then
    detected_ip="$(
      hostname -I 2>/dev/null |
      tr ' ' '\n' |
      awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ && $0 !~ /^127\./ {
        print
        exit
      }'
    )"
  fi

  # 最后的兜底地址。
  if [[ -z "$detected_ip" ]]; then
    detected_ip="127.0.0.1"
  fi

  printf '%s' "$detected_ip"
}

ACCESS_IP="${WEB_ACCESS_IP:-$(detect_access_ip)}"

[[ -f /opt/ros/noetic/setup.bash ]] || {
  error "未找到 /opt/ros/noetic/setup.bash"
  exit 1
}

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

[[ -f "$WS/devel/setup.bash" ]] || {
  error "未找到 $WS/devel/setup.bash，请先编译工作空间"
  exit 1
}

# shellcheck disable=SC1090
source "$WS/devel/setup.bash"

# 检查端口是否已经被占用。
if ss -lnt 2>/dev/null |
   awk '{print $4}' |
   grep -Eq "(^|:)$PORT$"; then
  error "端口 $PORT 已被占用"
  error "检查命令：sudo ss -lntp | grep ':${PORT}'"
  exit 1
fi

cat <<EOF
[INFO] 启动 R300 DWA Web 仿真实验台
[INFO] 本模式不启动惯导、GPS、相机、YOLO或底盘。
[INFO] move_base直接加载实车视觉导航使用的DWA与local costmap参数文件。
[INFO] Web支持常用参数调节、多点目标、人工视觉障碍和YAML/JSON导出。
[INFO] Web监听地址：$HOST:$PORT
[INFO] 自动识别工控机IP：$ACCESS_IP

浏览器访问：
  http://$ACCESS_IP:$PORT

VSCode Remote-SSH端口转发访问：
  1. 打开VSCode底部“端口 / Ports”面板；
  2. 转发远端端口 $PORT；
  3. 浏览器打开 http://127.0.0.1:$PORT

普通SSH端口转发：
  ssh -L $PORT:127.0.0.1:$PORT explorer@$ACCESS_IP
EOF

roslaunch r300_1x_navigation subject1_dwa_web_sim.launch \
  web_host:="$HOST" \
  web_port:="$PORT" \
  use_rviz:="$RVIZ"