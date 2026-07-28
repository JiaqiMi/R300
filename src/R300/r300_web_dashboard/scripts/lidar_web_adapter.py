#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 FAST-LIO 点云与 GridMap 高程图压缩成浏览器友好的 JSON 话题。

本节点只做显示适配，不修改 single_lidar_elevation、导航或代价地图逻辑。
输出：
  /r300_web/lidar_points_json  (std_msgs/String)
  /r300_web/elevation_json     (std_msgs/String)
"""

import json
import math
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from grid_map_msgs.msg import GridMap
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class LidarWebAdapter:
    def __init__(self) -> None:
        self.cloud_topic = rospy.get_param("~cloud_topic", "/cloud_registered_body")
        self.elevation_topic = rospy.get_param(
            "~elevation_topic", "/elevation_mapping/elevation_map_raw"
        )
        self.cloud_output_topic = rospy.get_param(
            "~cloud_output_topic", "/r300_web/lidar_points_json"
        )
        self.elevation_output_topic = rospy.get_param(
            "~elevation_output_topic", "/r300_web/elevation_json"
        )
        self.max_cloud_points = max(300, int(rospy.get_param("~max_cloud_points", 6000)))
        self.cloud_publish_hz = max(0.2, float(rospy.get_param("~cloud_publish_hz", 2.0)))
        self.elevation_publish_hz = max(
            0.2, float(rospy.get_param("~elevation_publish_hz", 1.0))
        )
        self.elevation_max_dim = max(40, int(rospy.get_param("~elevation_max_dim", 150)))
        self.cloud_max_range_m = max(
            1.0, float(rospy.get_param("~cloud_max_range_m", 35.0))
        )
        self.cloud_min_z_m = float(rospy.get_param("~cloud_min_z_m", -5.0))
        self.cloud_max_z_m = float(rospy.get_param("~cloud_max_z_m", 5.0))

        # 高程图是 odom 轴对齐的、GridMap 不携带车辆朝向；前端"车头朝上"视图需要
        # odom->body 的 yaw。必须取 FAST-LIO 这棵 TF 树（odom->camera_init->body），
        # 绝不能用 /one_x/odom 的朝向——那是 1X 惯导的另一棵 odom 树，两树 yaw 差一个任意常量。
        self.map_frame = str(rospy.get_param("~map_frame", "odom"))
        self.base_frame = str(rospy.get_param("~base_frame", "body"))
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self.cloud_pub = rospy.Publisher(
            self.cloud_output_topic, String, queue_size=1, latch=False
        )
        self.elevation_pub = rospy.Publisher(
            self.elevation_output_topic, String, queue_size=1, latch=False
        )

        self._lock = threading.RLock()
        self._last_cloud_pub = 0.0
        self._last_elevation_pub = 0.0

        rospy.Subscriber(
            self.cloud_topic,
            PointCloud2,
            self._cloud_callback,
            queue_size=1,
            buff_size=32 * 1024 * 1024,
        )
        rospy.Subscriber(
            self.elevation_topic,
            GridMap,
            self._elevation_callback,
            queue_size=1,
            buff_size=64 * 1024 * 1024,
        )

        rospy.loginfo(
            "Lidar web adapter ready: cloud=%s -> %s, elevation=%s -> %s",
            self.cloud_topic,
            self.cloud_output_topic,
            self.elevation_topic,
            self.elevation_output_topic,
        )

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        try:
            value = float(stamp.to_sec())
            if value > 0.0:
                return value
        except Exception:
            pass
        return rospy.Time.now().to_sec()

    def _lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """查 odom->body 最新变换，返回 (x, y, yaw)；TF 未就绪时返回 None。

        yaw 按 ZYX 欧拉角提取 = body x 轴在水平面上的方位角，雷达斜装（pitch≠0）
        时依然正确，仅安装接近竖直（|pitch|→90°）时退化。
        """
        try:
            t = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.05)
            )
        except tf2_ros.TransformException as exc:
            rospy.logwarn_throttle(
                10.0, "TF %s->%s 不可用，车辆朝向暂缺：%s", self.map_frame, self.base_frame, exc
            )
            return None
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return (
            float(t.transform.translation.x),
            float(t.transform.translation.y),
            float(yaw),
        )

    def _should_publish(self, kind: str, hz: float) -> bool:
        now = rospy.get_time()
        with self._lock:
            if kind == "cloud":
                if now - self._last_cloud_pub < 1.0 / hz:
                    return False
                self._last_cloud_pub = now
            else:
                if now - self._last_elevation_pub < 1.0 / hz:
                    return False
                self._last_elevation_pub = now
        return True

    def _sample_uvs(self, msg: PointCloud2) -> List[Tuple[int, int]]:
        width = max(1, int(msg.width))
        height = max(1, int(msg.height))
        total = width * height
        count = min(total, self.max_cloud_points * 2)
        if count >= total:
            return [(i % width, i // width) for i in range(total)]
        indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
        return [(int(i % width), int(i // width)) for i in indices]

    def _cloud_callback(self, msg: PointCloud2) -> None:
        if not self._should_publish("cloud", self.cloud_publish_hz):
            return
        try:
            field_names = {field.name for field in msg.fields}
            if not {"x", "y", "z"}.issubset(field_names):
                rospy.logwarn_throttle(5.0, "点云缺少 x/y/z 字段：%s", sorted(field_names))
                return

            uvs = self._sample_uvs(msg)
            points_mm: List[int] = []
            bounds = [math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf]
            max_r2 = self.cloud_max_range_m * self.cloud_max_range_m

            for x, y, z in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs
            ):
                x = float(x)
                y = float(y)
                z = float(z)
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    continue
                if x * x + y * y > max_r2:
                    continue
                if z < self.cloud_min_z_m or z > self.cloud_max_z_m:
                    continue
                points_mm.extend(
                    (int(round(x * 1000.0)), int(round(y * 1000.0)), int(round(z * 1000.0)))
                )
                bounds[0] = min(bounds[0], x)
                bounds[1] = max(bounds[1], x)
                bounds[2] = min(bounds[2], y)
                bounds[3] = max(bounds[3], y)
                bounds[4] = min(bounds[4], z)
                bounds[5] = max(bounds[5], z)
                if len(points_mm) // 3 >= self.max_cloud_points:
                    break

            count = len(points_mm) // 3
            if count == 0:
                bounds = [0.0] * 6
            payload: Dict[str, object] = {
                "version": 1,
                "stamp": self._stamp_to_sec(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "source_points": int(msg.width) * int(msg.height),
                "count": count,
                "scale": 0.001,
                "points": points_mm,
                "bounds": [round(float(v), 3) for v in bounds],
            }
            self.cloud_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "点云 Web 适配失败：%s", exc)

    @staticmethod
    def _matrix_from_grid_layer(msg: GridMap, layer_index: int) -> Optional[np.ndarray]:
        array_msg = msg.data[layer_index]
        dims = list(array_msg.layout.dim)
        if len(dims) < 2:
            return None
        label0 = str(dims[0].label)
        size0 = int(dims[0].size)
        size1 = int(dims[1].size)
        if size0 <= 0 or size1 <= 0:
            return None

        data = np.asarray(array_msg.data, dtype=np.float32)
        needed = size0 * size1
        if data.size < needed:
            return None
        if data.size > needed:
            data = data[:needed]

        if label0 == "row_index":
            rows, cols = size0, size1
            matrix = data.reshape((rows, cols), order="C")
        else:
            # grid_map 默认 Eigen::MatrixXf 为列主序，dim[0] 标记 column_index。
            cols, rows = size0, size1
            matrix = data.reshape((rows, cols), order="F")

        # 将环形缓存还原为逻辑地图顺序。逻辑 row=0/col=0 对应地图左上角。
        outer = int(msg.outer_start_index) % max(1, rows)
        inner = int(msg.inner_start_index) % max(1, cols)
        matrix = np.roll(matrix, shift=(-outer, -inner), axis=(0, 1))
        return matrix

    def _elevation_callback(self, msg: GridMap) -> None:
        if not self._should_publish("elevation", self.elevation_publish_hz):
            return
        try:
            if "elevation" not in msg.layers:
                rospy.logwarn_throttle(5.0, "GridMap 未包含 elevation 层：%s", list(msg.layers))
                return
            layer_index = list(msg.layers).index("elevation")
            matrix = self._matrix_from_grid_layer(msg, layer_index)
            if matrix is None or matrix.size == 0:
                return

            rows, cols = matrix.shape
            out_rows = min(rows, self.elevation_max_dim)
            out_cols = min(cols, self.elevation_max_dim)
            row_indices = np.linspace(0, rows - 1, out_rows, dtype=np.int32)
            col_indices = np.linspace(0, cols - 1, out_cols, dtype=np.int32)
            small = matrix[np.ix_(row_indices, col_indices)]

            valid_mask = np.isfinite(small)
            valid_values = small[valid_mask]
            sentinel = -32768
            encoded = np.full(small.shape, sentinel, dtype=np.int32)
            encoded[valid_mask] = np.clip(
                np.rint(small[valid_mask] * 1000.0), -32767, 32767
            ).astype(np.int32)

            if valid_values.size:
                actual_min = float(np.min(valid_values))
                actual_max = float(np.max(valid_values))
                color_min = float(np.percentile(valid_values, 2.0))
                color_max = float(np.percentile(valid_values, 98.0))
                if color_max - color_min < 0.05:
                    center = 0.5 * (color_min + color_max)
                    color_min = center - 0.025
                    color_max = center + 0.025
            else:
                actual_min = actual_max = color_min = color_max = 0.0

            robot_pose = self._lookup_robot_pose()

            payload: Dict[str, object] = {
                "version": 2,
                "stamp": self._stamp_to_sec(msg.info.header.stamp),
                "frame_id": msg.info.header.frame_id,
                "robot_x": round(robot_pose[0], 3) if robot_pose else None,
                "robot_y": round(robot_pose[1], 3) if robot_pose else None,
                "robot_yaw": round(robot_pose[2], 4) if robot_pose else None,
                "rows": int(out_rows),
                "cols": int(out_cols),
                "source_rows": int(rows),
                "source_cols": int(cols),
                "resolution": float(msg.info.resolution),
                "length_x": float(msg.info.length_x),
                "length_y": float(msg.info.length_y),
                "center_x": float(msg.info.pose.position.x),
                "center_y": float(msg.info.pose.position.y),
                "scale": 0.001,
                "invalid": sentinel,
                "valid_count": int(valid_values.size),
                "min": round(actual_min, 4),
                "max": round(actual_max, 4),
                "color_min": round(color_min, 4),
                "color_max": round(color_max, 4),
                "values": encoded.ravel(order="C").tolist(),
            }
            self.elevation_pub.publish(
                String(data=json.dumps(payload, separators=(",", ":")))
            )
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "高程图 Web 适配失败：%s", exc)


def main() -> None:
    rospy.init_node("r300_lidar_web_adapter", anonymous=False)
    LidarWebAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
