#!/usr/bin/env bash
# Delay elevation_mapping_cupy until FAST-LIO has both a live body-frame cloud
# and a usable odom->body TF.  This avoids the startup-only TF extrapolation
# caused by elevation_mapping subscribing before FAST-LIO's first TF sample.

set -Eeuo pipefail

timeout_s="${ELEVATION_START_TIMEOUT_S:-40}"
settle_s="${ELEVATION_START_SETTLE_S:-1}"
cloud_topic="${ELEVATION_INPUT_CLOUD_TOPIC:-/cloud_registered_body}"
parent_frame="${ELEVATION_MAP_FRAME:-odom}"
child_frame="${ELEVATION_SENSOR_FRAME:-body}"

deadline=$((SECONDS + timeout_s))

echo "[INFO] elevation_mapping 等待 ${cloud_topic} 和 TF ${parent_frame}->${child_frame}（最多 ${timeout_s}s）……"

while (( SECONDS < deadline )); do
    cloud_ok=false
    tf_ok=false

    if timeout 2 rostopic echo -n 1 "${cloud_topic}" >/dev/null 2>&1; then
        cloud_ok=true
    fi

    tf_output="$(timeout 2 rosrun tf tf_echo "${parent_frame}" "${child_frame}" 2>/dev/null || true)"
    if grep -q -- '- Translation:' <<<"${tf_output}"; then
        tf_ok=true
    fi

    if [[ "${cloud_ok}" == true && "${tf_ok}" == true ]]; then
        echo "[ OK ] FAST-LIO 点云与 TF 已就绪，${settle_s}s 后启动 elevation_mapping。"
        sleep "${settle_s}"
        exec "$@"
    fi

    sleep 0.5
done

echo "[ERROR] 等待 FAST-LIO 点云/TF 超时：cloud=${cloud_topic}, TF=${parent_frame}->${child_frame}" >&2
echo "        请检查：rostopic hz ${cloud_topic}" >&2
echo "        请检查：rosrun tf tf_echo ${parent_frame} ${child_frame}" >&2
exit 1
