#!/usr/bin/env bash

set -u

# ============================================================
# R300视觉系统一键启动脚本
#
# 用法：
#   ./start_r300.sh web
#   ./start_r300.sh bag
#   ./start_r300.sh both
#
# web  ：相机 + YOLO + Web
# bag  ：相机 + YOLO + rosbag
# both ：相机 + YOLO + Web + rosbag
# ============================================================


# ---------------------------
# 1. 基础路径
# ---------------------------

R300_WS="/home/explorer/r300_ws"
RECORD_DIR="/home/explorer/r300_records"

# 请根据实际虚拟环境路径修改
# 在当前(yolo26)终端执行：
# echo $VIRTUAL_ENV
YOLO_ENV="/home/explorer/venvs/yolo26"

WEB_PORT="8080"
WEB_ADDRESS="0.0.0.0"


# ---------------------------
# 2. 启动模式
# ---------------------------

MODE="${1:-web}"

case "${MODE}" in
    web)
        ENABLE_WEB="true"
        ENABLE_BAG="false"
        ;;

    bag)
        ENABLE_WEB="false"
        ENABLE_BAG="true"
        ;;

    both)
        ENABLE_WEB="true"
        ENABLE_BAG="true"
        ;;

    *)
        echo "错误：未知启动模式 ${MODE}"
        echo
        echo "正确用法："
        echo "  $0 web"
        echo "  $0 bag"
        echo "  $0 both"
        exit 1
        ;;
esac


# ---------------------------
# 3. 加载环境
# ---------------------------

if [ ! -f "/opt/ros/noetic/setup.bash" ]; then
    echo "[ERROR] 找不到ROS Noetic环境"
    exit 1
fi

source /opt/ros/noetic/setup.bash

if [ ! -f "${R300_WS}/devel/setup.bash" ]; then
    echo "[ERROR] 找不到工作空间环境："
    echo "${R300_WS}/devel/setup.bash"
    echo "请先执行："
    echo "cd ${R300_WS} && catkin_make"
    exit 1
fi

source "${R300_WS}/devel/setup.bash"


# 激活Python虚拟环境
if [ -f "${YOLO_ENV}/bin/activate" ]; then
    source "${YOLO_ENV}/bin/activate"
else
    echo "[ERROR] 找不到YOLO虚拟环境："
    echo "${YOLO_ENV}/bin/activate"
    echo
    echo "请在(yolo26)终端执行："
    echo "echo \$VIRTUAL_ENV"
    echo "然后修改脚本中的YOLO_ENV"
    exit 1
fi


# Jetson上的libgomp兼容设置
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONUNBUFFERED=1


# ---------------------------
# 4. 创建记录目录
# ---------------------------

mkdir -p "${RECORD_DIR}"


# ---------------------------
# 5. 检查环境
# ---------------------------

echo "=========================================="
echo "R300视觉系统启动"
echo "=========================================="
echo "模式           : ${MODE}"
echo "工作空间       : ${R300_WS}"
echo "Python         : $(which python3)"
echo "记录目录       : ${RECORD_DIR}"
echo "Web启用        : ${ENABLE_WEB}"
echo "Rosbag启用     : ${ENABLE_BAG}"
echo "=========================================="


python3 - <<'PY'
import torch

print("Torch :", torch.__version__)
print("CUDA  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA不可用，停止启动")
PY

if [ $? -ne 0 ]; then
    echo "[ERROR] GPU环境检查失败"
    exit 1
fi


# ---------------------------
# 6. 清理函数
# ---------------------------

ROSLAUNCH_PID=""
ROSBAG_PID=""

cleanup()
{
    echo
    echo "正在停止R300视觉系统……"

    if [ -n "${ROSBAG_PID}" ]; then
        kill -SIGINT "${ROSBAG_PID}" 2>/dev/null || true
        wait "${ROSBAG_PID}" 2>/dev/null || true
    fi

    if [ -n "${ROSLAUNCH_PID}" ]; then
        kill -SIGINT "${ROSLAUNCH_PID}" 2>/dev/null || true
        wait "${ROSLAUNCH_PID}" 2>/dev/null || true
    fi

    echo "R300视觉系统已停止"
}

trap cleanup SIGINT SIGTERM EXIT


# ---------------------------
# 7. 启动相机、YOLO和Web
# ---------------------------

echo
echo "[1/3] 启动D435i和YOLO节点……"

roslaunch \
    r300_yolo_detector \
    r300_system.launch \
    enable_web:="${ENABLE_WEB}" \
    web_port:="${WEB_PORT}" \
    web_address:="${WEB_ADDRESS}" &

ROSLAUNCH_PID=$!

echo "roslaunch PID = ${ROSLAUNCH_PID}"


# ---------------------------
# 8. 等待检测图像话题
# ---------------------------

echo
echo "[2/3] 等待检测图像话题……"

MAX_WAIT_SECONDS=60
WAIT_COUNT=0

while true
do
    if rostopic list 2>/dev/null | \
       grep -qx "/r300_vision/annotated_image"; then
        echo "检测图像话题已出现"
        break
    fi

    if ! kill -0 "${ROSLAUNCH_PID}" 2>/dev/null; then
        echo "[ERROR] roslaunch进程已经退出"
        exit 1
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))

    if [ "${WAIT_COUNT}" -ge "${MAX_WAIT_SECONDS}" ]; then
        echo "[ERROR] 等待检测图像话题超时"
        exit 1
    fi

    echo "等待中…… ${WAIT_COUNT}/${MAX_WAIT_SECONDS}"
    sleep 1
done


# 再等待几秒，让YOLO模型完成预热
sleep 3


# ---------------------------
# 9. 可选启动rosbag
# ---------------------------

if [ "${ENABLE_BAG}" = "true" ]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    BAG_PREFIX="${RECORD_DIR}/yolo_test_${TIMESTAMP}"

    echo
    echo "[3/3] 启动rosbag录制……"
    echo "保存位置：${BAG_PREFIX}.bag"

    rosbag record \
        -O "${BAG_PREFIX}" \
        /r300_vision/annotated_image \
        /r300_vision/detections \
        /r300_vision/target_point &

    ROSBAG_PID=$!

    echo "rosbag PID = ${ROSBAG_PID}"
else
    echo
    echo "[3/3] 当前模式不录制rosbag"
fi


# ---------------------------
# 10. 打印访问信息
# ---------------------------

echo
echo "=========================================="
echo "R300视觉系统已启动"
echo "=========================================="

if [ "${ENABLE_WEB}" = "true" ]; then
    CURRENT_IP="$(hostname -I | awk '{print $1}')"

    echo "Web主页："
    echo "http://${CURRENT_IP}:${WEB_PORT}/"
    echo
    echo "检测视频流："
    echo "http://${CURRENT_IP}:${WEB_PORT}/stream?topic=/r300_vision/annotated_image&type=mjpeg"
fi

if [ "${ENABLE_BAG}" = "true" ]; then
    echo
    echo "Rosbag正在录制："
    echo "${BAG_PREFIX}.bag"
fi

echo
echo "按 Ctrl+C 停止所有节点并安全结束录制"
echo "=========================================="


# 等待roslaunch进程结束
wait "${ROSLAUNCH_PID}"
