#!/usr/bin/env bash
# R300 开机自启引导（用户级 crontab @reboot 调用, 2026-07-30）
# 目标: 上电后无需任何 ssh/终端操作, 浏览器打开面板即可从 0 到 1 全 web 操作。
#
# 做四件事(全部幂等, 失败互不牵连):
#   1. can0 配置 —— 依赖一次性 sudoers 白名单(见 README_CN.md "开机自启"节):
#        echo 'explorer ALL=(ALL) NOPASSWD: /usr/sbin/ip, /usr/bin/jetson_clocks' \
#          | sudo tee /etc/sudoers.d/r300-boot
#      未配置白名单时本步跳过, 导航按钮启动时仍会用缓存密码机制兜底配置。
#   2. jetson_clocks —— 锁频+风扇策略, 防下午 80°C FAST-LIO 被饿死复发。
#   3. roscore 独立启动 —— master 与面板解耦: 重启面板不再导致全栈(感知/导航)全灭。
#   4. 面板(dashboard) —— 加入已有 master。
set -u
LOG="$HOME/web_boot.log"
exec >>"$LOG" 2>&1
echo "============================================================"
echo "[$(date '+%F %T')] boot_r300_web.sh 开机引导"

# 等系统网络/设备就绪
sleep 20

echo "-- 1. can0 --"
if sudo -n ip link set can0 down 2>/dev/null; then
  sudo -n ip link set can0 type can bitrate 500000
  sudo -n ip link set can0 up
  ip -br link show can0
else
  echo "无 NOPASSWD 白名单, 跳过(导航按钮会用缓存密码兜底配置 can0)"
fi

echo "-- 2. jetson_clocks --"
sudo -n jetson_clocks 2>/dev/null && echo "已锁频" || echo "无权限, 跳过"

source /opt/ros/noetic/setup.bash
source "$HOME/r300_ws/devel/setup.bash"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

echo "-- 3. roscore(独立 master, 与面板解耦) --"
if ! rostopic list >/dev/null 2>&1; then
  nohup roscore >"$HOME/roscore.log" 2>&1 &
  for _ in $(seq 1 30); do
    rostopic list >/dev/null 2>&1 && break
    sleep 1
  done
fi
rostopic list >/dev/null 2>&1 && echo "master 就绪" || echo "!! master 未就绪"

echo "-- 4. 面板 --"
if ! pgrep -f dashboard_server.py >/dev/null; then
  nohup roslaunch r300_web_dashboard r300_web_dashboard.launch \
    >"$HOME/web.log" 2>&1 &
  sleep 15
fi
curl -s -o /dev/null -w "面板 HTTP %{http_code}\n" http://localhost:8090/ || true
echo "[$(date '+%F %T')] 引导完成"
