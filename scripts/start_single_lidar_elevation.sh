#!/usr/bin/env bash

set -Eeuo pipefail

workspace_dir="${R300_WS:-/home/explorer/r300_ws}"
interface_name="${LIDAR_INTERFACE:-eth0}"
lidar_ip="${LIDAR_IP:-192.168.1.192}"
host_ip="${LIDAR_HOST_IP:-192.168.1.50}"
network_profile="${LIDAR_PROFILE:-mid360-lidar}"
rviz_enabled="${RVIZ:-1}"
tilt_pitch_deg="${TILT_PITCH_DEG:--55.5}"
lio_input_crop="${LIO_INPUT_CROP:-false}"
restart_existing="${RESTART_EXISTING:-1}"

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
    echo "[ERROR] ROS Noetic is not installed."
    exit 1
fi
if [[ ! -f "${workspace_dir}/devel/setup.bash" ]]; then
    echo "[ERROR] Workspace is not built: ${workspace_dir}"
    exit 1
fi

source /opt/ros/noetic/setup.bash
source "${workspace_dir}/devel/setup.bash"

if [[ -f /home/explorer/venvs/yolo26/bin/activate ]]; then
    source /home/explorer/venvs/yolo26/bin/activate
fi
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONUNBUFFERED=1
export DISABLE_ROS1_EOL_WARNINGS=1

if ! ip link show "${interface_name}" >/dev/null 2>&1; then
    echo "[ERROR] Interface does not exist: ${interface_name}"
    exit 1
fi

if ! ethtool "${interface_name}" 2>/dev/null | grep -q 'Link detected: yes'; then
    echo "[ERROR] No Ethernet carrier on ${interface_name}. Check MID-360 power and cable."
    exit 2
fi

if ! ip -4 addr show dev "${interface_name}" | grep -q "${host_ip}/"; then
    nmcli connection up "${network_profile}" >/dev/null 2>&1 || true
fi

if ! ip -4 addr show dev "${interface_name}" | grep -q "${host_ip}/"; then
    echo "[ERROR] ${interface_name} has no ${host_ip}/24 address."
    echo "        Configure ${network_profile} or run: sudo ip addr add ${host_ip}/24 dev ${interface_name}"
    exit 3
fi

if ! ping -I "${interface_name}" -c 2 -W 1 "${lidar_ip}" >/dev/null 2>&1; then
    echo "[ERROR] Lidar ${lidar_ip} is not reachable through ${interface_name}."
    exit 4
fi

mapfile -t existing_pids < <(pgrep -f 'roslaunch single_lidar_elevation single_lidar_elevation.launch' || true)
if (( ${#existing_pids[@]} > 0 )); then
    if [[ "${restart_existing}" != 1 && "${restart_existing,,}" != true ]]; then
        echo "[ERROR] The single-lidar launch is already running: ${existing_pids[*]}"
        exit 5
    fi

    echo "Stopping previous single-lidar launch: ${existing_pids[*]}"
    kill -INT "${existing_pids[@]}"
    for _ in {1..60}; do
        mapfile -t existing_pids < <(pgrep -f 'roslaunch single_lidar_elevation single_lidar_elevation.launch' || true)
        (( ${#existing_pids[@]} == 0 )) && break
        sleep 0.5
    done
    if (( ${#existing_pids[@]} > 0 )); then
        echo "[ERROR] Previous single-lidar launch did not stop: ${existing_pids[*]}"
        exit 5
    fi
fi

if [[ "${rviz_enabled}" == 1 || "${rviz_enabled,,}" == true ]]; then
    export DISPLAY="${DISPLAY:-:0}"
    export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
    rviz_arg=true
else
    rviz_arg=false
fi

echo "Lidar: ${lidar_ip} via ${interface_name} (${host_ip})"
echo "RViz: ${rviz_arg}"
echo "Tilt pitch: ${tilt_pitch_deg} deg"
echo "FAST-LIO input crop: ${lio_input_crop}"
exec roslaunch single_lidar_elevation single_lidar_elevation.launch \
    rviz:="${rviz_arg}" \
    tilt_pitch_deg:="${tilt_pitch_deg}" \
    lio_input_crop:="${lio_input_crop}"
