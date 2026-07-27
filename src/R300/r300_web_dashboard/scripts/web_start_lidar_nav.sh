#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/.ros/r300_web_dashboard"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_start_lidar_nav.log"

WS="${R300_WS:-$HOME/r300_ws}"
NAV_DIR="$WS/src/R300/r300_1x_navigation/scripts/one_key"
PKG="r300_1x_navigation"
LAUNCH_FILE="${R300_LIDAR_NAV_LAUNCH:-subject1_waypoint_lidar_nav.launch}"
CAN_PORT="${CAN_PORT:-can0}"
CAN_BITRATE="${CAN_BITRATE:-500000}"
SUDO_PASS="${R300_SUDO_PASS:-1234}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"

{
  echo "============================================================"
  echo "[$(date '+%F %T')] web_start_lidar_nav.sh"
  echo "USER=$USER HOME=$HOME SHELL=$SHELL"

  source /opt/ros/noetic/setup.bash
  source "$WS/devel/setup.bash"
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  export PYTHONUNBUFFERED=1

  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  echo "LAUNCH=$PKG/$LAUNCH_FILE"
  echo "CAN=$CAN_PORT bitrate=$CAN_BITRATE"

  rospack find "$PKG" >/dev/null 2>&1 || {
    echo "ROS 找不到功能包：$PKG"
    exit 1
  }
  PKG_PATH="$(rospack find "$PKG")"
  [[ -f "$PKG_PATH/launch/$LAUNCH_FILE" ]] || {
    echo "未找到雷达避障 launch：$PKG_PATH/launch/$LAUNCH_FILE"
    exit 1
  }

  ADAPTER="$PKG_PATH/scripts/lidar_obstacle_scan_node.py"
  [[ -f "$ADAPTER" ]] || {
    echo "未找到雷达障碍适配节点：$ADAPTER"
    exit 1
  }
  [[ -x "$ADAPTER" ]] || chmod +x "$ADAPTER"

  # 1X 必须先独立运行；本按钮不重复启动 1X。
  rosnode list >/dev/null 2>&1 || {
    echo "ROS master 未运行，请先点击‘① 启动 1X 惯导’"
    exit 1
  }
  rosnode list 2>/dev/null | grep -qx '/one_x_serial_driver' || {
    echo "未发现 /one_x_serial_driver，请先点击‘① 启动 1X 惯导’"
    exit 1
  }
  timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/ins_fix >/dev/null 2>&1 || {
    echo "独立 1X 没有发布 /one_x/ins_fix"
    exit 1
  }

  # 雷达导航 launch 不负责启动 MID-360/FAST-LIO/高程图，必须先有高程图数据。
  timeout "$READY_TIMEOUT" rostopic echo -n 1 \
    /elevation_mapping/elevation_map_raw >/dev/null 2>&1 || {
    echo "没有收到 /elevation_mapping/elevation_map_raw"
    echo "请先点击‘启动雷达 / 点云 / 高程’，等待高程图出现"
    exit 1
  }

  if rosnode list 2>/dev/null | grep -qx '/move_base'; then
    echo "检测到已有 /move_base，请先停止纯实车或视觉避障导航"
    exit 1
  fi

  # 以当前最新 1X 位置建立本次导航 ENU 原点。
  for _ in $(seq 1 $((READY_TIMEOUT * 5))); do
    rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' && break
    sleep 0.2
  done
  rosservice list 2>/dev/null | grep -qx '/one_x/set_current_origin' || {
    echo "未发现 /one_x/set_current_origin"
    exit 1
  }
  echo "以当前 1X 位置建立导航原点..."
  ORIGIN_RESPONSE="$(rosservice call /one_x/set_current_origin '{}' 2>&1 || true)"
  printf '%s\n' "$ORIGIN_RESPONSE"
  grep -Eq 'success:[[:space:]]*(True|true)' <<<"$ORIGIN_RESPONSE" || {
    echo "1X 导航原点设置失败"
    exit 1
  }
  timeout "$READY_TIMEOUT" rostopic echo -n 1 /one_x/odom >/dev/null 2>&1 || {
    echo "设置原点后没有收到 /one_x/odom"
    exit 1
  }

  # 与纯实车、视觉导航相同，在后台完成 CAN 的 sudo 操作。
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' -v
  ip link show "$CAN_PORT" >/dev/null 2>&1 || {
    echo "未找到 CAN 接口：$CAN_PORT"
    exit 1
  }
  sudo -n ip link set "$CAN_PORT" down >/dev/null 2>&1 || true
  sudo -n ip link set "$CAN_PORT" type can bitrate "$CAN_BITRATE"
  sudo -n ip link set "$CAN_PORT" up

  cd "$NAV_DIR"
  echo "启动雷达高程图避障导航："
  echo "roslaunch $PKG $LAUNCH_FILE launch_rviz:=false"
  echo "说明：不自动开始航点；就绪后请在网页点击‘开始航点’。"
  exec roslaunch "$PKG" "$LAUNCH_FILE" launch_rviz:=false
} 2>&1 | tee -a "$LOG_FILE"
