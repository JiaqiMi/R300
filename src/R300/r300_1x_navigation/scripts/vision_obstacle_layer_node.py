#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert semantic RGB-D detections into a virtual 2-D obstacle scan.

The node deliberately does not command the chassis.  It only converts
r300_vision_msgs/DetectedObjectArray into:

* sensor_msgs/LaserScan for costmap_2d::ObstacleLayer;
* sensor_msgs/PointCloud2 for RViz/debugging.

A short per-beam hold time suppresses one-frame detector dropouts.  If the
input detection topic becomes stale, scan publication stops.  Together with
costmap's expected_update_rate this makes move_base stop instead of driving
blindly when the camera/detector fails.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import rospy
import sensor_msgs.point_cloud2 as point_cloud2
import tf2_ros
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, LaserScan, PointCloud2

from r300_vision_msgs.msg import DetectedObject, DetectedObjectArray


class VisionObstacleLayerNode:
    """Filter detections and publish the obstacle observation used by costmap."""

    def __init__(self) -> None:
        # Topics and frames.
        self.detections_topic = str(
            rospy.get_param("~detections_topic", "/r300_vision/detections")
        )
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        )
        self.scan_topic = str(
            rospy.get_param("~scan_topic", "/r300_vision/obstacle_scan")
        )
        self.cloud_topic = str(
            rospy.get_param("~cloud_topic", "/r300_vision/obstacle_points")
        )
        self.target_frame = str(rospy.get_param("~target_frame", "base_link"))

        # Semantic and geometric acceptance conditions.
        self.obstacle_classes = {
            str(item) for item in rospy.get_param("~obstacle_classes", [])
        }
        self.ignored_classes = {
            str(item) for item in rospy.get_param("~ignored_classes", [])
        }
        self.default_min_confidence = float(
            rospy.get_param("~default_min_confidence", 0.50)
        )
        raw_class_thresholds = rospy.get_param("~class_min_confidence", {})
        if not isinstance(raw_class_thresholds, dict):
            raise ValueError("~class_min_confidence must be a dictionary")
        self.class_min_confidence: Dict[str, float] = {
            str(name): float(value)
            for name, value in raw_class_thresholds.items()
        }

        self.require_depth = bool(rospy.get_param("~require_depth", True))
        self.min_distance_m = float(rospy.get_param("~min_distance_m", 0.35))
        self.max_distance_m = float(rospy.get_param("~max_distance_m", 7.50))
        self.min_forward_x_m = float(rospy.get_param("~min_forward_x_m", 0.10))
        self.max_abs_lateral_m = float(
            rospy.get_param("~max_abs_lateral_m", 5.00)
        )
        self.min_center_height_m = float(
            rospy.get_param("~min_center_height_m", -0.50)
        )
        self.max_center_height_m = float(
            rospy.get_param("~max_center_height_m", 2.50)
        )
        self.min_bbox_width_px = int(rospy.get_param("~min_bbox_width_px", 8))
        self.min_bbox_height_px = int(rospy.get_param("~min_bbox_height_px", 8))
        self.max_message_age_s = float(
            rospy.get_param("~max_message_age_s", 0.50)
        )

        # Virtual scan layout and conservative expansion.
        self.scan_angle_min = math.radians(
            float(rospy.get_param("~scan_angle_min_deg", -58.0))
        )
        self.scan_angle_max = math.radians(
            float(rospy.get_param("~scan_angle_max_deg", 58.0))
        )
        self.scan_angle_increment = math.radians(
            float(rospy.get_param("~scan_angle_increment_deg", 0.50))
        )
        self.scan_range_min = float(rospy.get_param("~scan_range_min_m", 0.20))
        self.scan_range_max = float(rospy.get_param("~scan_range_max_m", 8.00))
        self.angular_padding = math.radians(
            float(rospy.get_param("~angular_padding_deg", 1.50))
        )
        self.lateral_padding_m = float(
            rospy.get_param("~lateral_padding_m", 0.18)
        )
        self.range_safety_margin_m = float(
            rospy.get_param("~range_safety_margin_m", 0.15)
        )
        self.hold_time_s = float(rospy.get_param("~hold_time_s", 0.35))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.input_timeout_s = float(rospy.get_param("~input_timeout_s", 0.60))
        self.tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.08))
        self.allow_latest_transform = bool(
            rospy.get_param("~allow_latest_transform", True)
        )
        self.debug_log = bool(rospy.get_param("~debug_log", True))

        self._validate_parameters()

        self.beam_count = int(
            math.floor(
                (self.scan_angle_max - self.scan_angle_min)
                / self.scan_angle_increment
            )
        ) + 1

        # Every beam stores the last conservative obstacle range and expiry.
        self._beam_ranges: List[float] = [math.inf] * self.beam_count
        self._beam_expiry: List[rospy.Time] = [rospy.Time(0)] * self.beam_count
        self._lock = threading.Lock()
        self._last_input_wall_time: Optional[rospy.Time] = None
        self._last_source_stamp = rospy.Time(0)
        self._camera_frame = ""

        # Camera intrinsics are needed to project bbox left/right edges.
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.scan_pub = rospy.Publisher(self.scan_topic, LaserScan, queue_size=5)
        self.cloud_pub = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=5)

        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1,
        )
        self.detections_sub = rospy.Subscriber(
            self.detections_topic,
            DetectedObjectArray,
            self.detections_callback,
            queue_size=5,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate_hz),
            self.publish_timer_callback,
        )

        rospy.loginfo(
            "Vision obstacle adapter ready: %s -> %s (frame=%s, beams=%d)",
            self.detections_topic,
            self.scan_topic,
            self.target_frame,
            self.beam_count,
        )

    def _validate_parameters(self) -> None:
        if self.scan_angle_max <= self.scan_angle_min:
            raise ValueError("scan_angle_max_deg must be greater than scan_angle_min_deg")
        if self.scan_angle_increment <= 0.0:
            raise ValueError("scan_angle_increment_deg must be positive")
        if self.scan_range_max <= self.scan_range_min:
            raise ValueError("scan_range_max_m must be greater than scan_range_min_m")
        if self.max_distance_m <= self.min_distance_m:
            raise ValueError("max_distance_m must be greater than min_distance_m")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self.hold_time_s < 0.0:
            raise ValueError("hold_time_s cannot be negative")
        if self.input_timeout_s <= 0.0:
            raise ValueError("input_timeout_s must be positive")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.K) < 9 or msg.K[0] <= 0.0 or msg.K[4] <= 0.0:
            rospy.logwarn_throttle(5.0, "Received invalid camera intrinsics")
            return
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])
        if msg.header.frame_id:
            self._camera_frame = msg.header.frame_id

    @staticmethod
    def _finite_point(obj: DetectedObject) -> bool:
        return all(
            math.isfinite(value)
            for value in (obj.position.x, obj.position.y, obj.position.z)
        )

    def _confidence_threshold(self, class_name: str) -> float:
        return self.class_min_confidence.get(
            class_name,
            self.default_min_confidence,
        )

    def _message_is_fresh(self, stamp: rospy.Time) -> bool:
        if stamp == rospy.Time(0):
            return True
        age = (rospy.Time.now() - stamp).to_sec()
        # Future timestamps can appear with imperfect clock synchronization.
        return -0.20 <= age <= self.max_message_age_s

    def _object_passes_basic_filters(self, obj: DetectedObject) -> Tuple[bool, str]:
        if self.obstacle_classes and obj.class_name not in self.obstacle_classes:
            return False, "class_not_enabled"
        if obj.class_name in self.ignored_classes:
            return False, "class_ignored"
        if obj.confidence < self._confidence_threshold(obj.class_name):
            return False, "low_confidence"
        if (obj.x_max - obj.x_min) < self.min_bbox_width_px:
            return False, "bbox_too_narrow"
        if (obj.y_max - obj.y_min) < self.min_bbox_height_px:
            return False, "bbox_too_short"
        if self.require_depth and not obj.depth_valid:
            return False, "invalid_depth"
        if not obj.depth_valid or not math.isfinite(obj.depth_m):
            return False, "invalid_depth"
        if not self._finite_point(obj):
            return False, "invalid_position"
        if not (self.min_distance_m <= obj.depth_m <= self.max_distance_m):
            return False, "depth_out_of_range"
        return True, "accepted"

    def _make_camera_point(
        self,
        obj: DetectedObject,
        u: float,
        v: float,
        frame_id: str,
        stamp: rospy.Time,
    ) -> PointStamped:
        if None in (self.fx, self.fy, self.cx, self.cy):
            raise RuntimeError("CameraInfo has not been received")

        depth = float(obj.depth_m)
        point = PointStamped()
        point.header.frame_id = frame_id
        point.header.stamp = stamp
        point.point.x = (float(u) - float(self.cx)) * depth / float(self.fx)
        point.point.y = (float(v) - float(self.cy)) * depth / float(self.fy)
        point.point.z = depth
        return point

    @staticmethod
    def _apply_transform(
        point: PointStamped,
        transform,
        target_frame: str,
    ) -> PointStamped:
        """Apply geometry_msgs/TransformStamped without tf2_geometry_msgs/PyKDL.

        The quaternion-vector rotation is evaluated directly, then translation
        is added.  This keeps the node compatible with the YOLO virtual
        environment while still using tf2_ros for TF lookup.
        """
        q = transform.transform.rotation
        t = transform.transform.translation

        # Normalize the quaternion defensively.
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm <= 1e-12:
            raise RuntimeError("TF contains an invalid zero-length quaternion")

        qx = q.x / norm
        qy = q.y / norm
        qz = q.z / norm
        qw = q.w / norm

        px = float(point.point.x)
        py = float(point.point.y)
        pz = float(point.point.z)

        # Rotate vector p using q * p * q^-1.
        tx = 2.0 * (qy * pz - qz * py)
        ty = 2.0 * (qz * px - qx * pz)
        tz = 2.0 * (qx * py - qy * px)

        rx = px + qw * tx + (qy * tz - qz * ty)
        ry = py + qw * ty + (qz * tx - qx * tz)
        rz = pz + qw * tz + (qx * ty - qy * tx)

        output = PointStamped()
        output.header.stamp = point.header.stamp
        output.header.frame_id = target_frame
        output.point.x = rx + t.x
        output.point.y = ry + t.y
        output.point.z = rz + t.z
        return output

    def _lookup_and_transform(
        self,
        point: PointStamped,
        stamp: rospy.Time,
    ) -> PointStamped:
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            point.header.frame_id,
            stamp,
            rospy.Duration(self.tf_timeout_s),
        )
        return self._apply_transform(point, transform, self.target_frame)

    def _transform_point(self, point: PointStamped) -> PointStamped:
        try:
            return self._lookup_and_transform(point, point.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as first_error:
            if not self.allow_latest_transform or point.header.stamp == rospy.Time(0):
                raise first_error

            # Fall back to the latest available TF while preserving point data.
            latest_point = PointStamped()
            latest_point.header.frame_id = point.header.frame_id
            latest_point.header.stamp = rospy.Time(0)
            latest_point.point.x = point.point.x
            latest_point.point.y = point.point.y
            latest_point.point.z = point.point.z
            return self._lookup_and_transform(latest_point, rospy.Time(0))

    def _object_sector(
        self,
        obj: DetectedObject,
        frame_id: str,
        stamp: rospy.Time,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return (left_angle, right_angle, range, center_z) in target_frame."""
        center_v = float(obj.center_v)
        center = self._transform_point(
            self._make_camera_point(obj, obj.center_u, center_v, frame_id, stamp)
        )
        left = self._transform_point(
            self._make_camera_point(obj, obj.x_min, center_v, frame_id, stamp)
        )
        right = self._transform_point(
            self._make_camera_point(obj, obj.x_max, center_v, frame_id, stamp)
        )

        cx = float(center.point.x)
        cy = float(center.point.y)
        cz = float(center.point.z)
        planar_range = math.hypot(cx, cy)

        if cx < self.min_forward_x_m:
            return None
        if abs(cy) > self.max_abs_lateral_m:
            return None
        if not (self.min_center_height_m <= cz <= self.max_center_height_m):
            return None
        if not (self.min_distance_m <= planar_range <= self.max_distance_m):
            return None

        angles = [
            math.atan2(left.point.y, left.point.x),
            math.atan2(right.point.y, right.point.x),
            math.atan2(cy, cx),
        ]
        angle_low = min(angles)
        angle_high = max(angles)

        # Expand the apparent object width before costmap inflation is applied.
        lateral_padding_angle = math.atan2(
            max(0.0, self.lateral_padding_m),
            max(planar_range, 1e-3),
        )
        total_padding = self.angular_padding + lateral_padding_angle
        angle_low -= total_padding
        angle_high += total_padding

        conservative_range = max(
            self.scan_range_min,
            planar_range - max(0.0, self.range_safety_margin_m),
        )
        return angle_low, angle_high, conservative_range, cz

    def _angle_to_index(self, angle: float) -> int:
        return int(math.floor((angle - self.scan_angle_min) / self.scan_angle_increment))

    def _paint_sector(
        self,
        frame_ranges: List[float],
        sector: Tuple[float, float, float, float],
    ) -> int:
        angle_low, angle_high, obstacle_range, _ = sector
        clipped_low = max(self.scan_angle_min, angle_low)
        clipped_high = min(self.scan_angle_max, angle_high)
        if clipped_high < clipped_low:
            return 0

        first = max(0, self._angle_to_index(clipped_low))
        last = min(self.beam_count - 1, self._angle_to_index(clipped_high) + 1)
        updated = 0
        for index in range(first, last + 1):
            frame_ranges[index] = min(frame_ranges[index], obstacle_range)
            updated += 1
        return updated

    def _commit_frame_ranges(
        self,
        frame_ranges: Sequence[float],
        now: rospy.Time,
    ) -> None:
        expiry = now + rospy.Duration(self.hold_time_s)
        with self._lock:
            for index, value in enumerate(frame_ranges):
                if math.isfinite(value):
                    # Replace with this frame's observation, even when the new
                    # obstacle is farther away.  Beams absent from this frame
                    # retain their previous value only until hold_time expires.
                    self._beam_ranges[index] = value
                    self._beam_expiry[index] = expiry

    def detections_callback(self, msg: DetectedObjectArray) -> None:
        now = rospy.Time.now()
        source_stamp = msg.header.stamp
        if source_stamp == rospy.Time(0):
            source_stamp = now

        if not self._message_is_fresh(msg.header.stamp):
            rospy.logwarn_throttle(2.0, "Ignoring stale/future vision detection message")
            return

        if None in (self.fx, self.fy, self.cx, self.cy):
            rospy.logwarn_throttle(2.0, "Waiting for CameraInfo before building obstacle scan")
            return

        accepted = 0
        rejected = 0
        transformed_failures = 0
        processing_failures = 0
        eligible_objects = 0
        updated_beams = 0
        frame_ranges = [math.inf] * self.beam_count

        for obj in msg.objects:
            valid, _reason = self._object_passes_basic_filters(obj)
            if not valid:
                rejected += 1
                continue

            eligible_objects += 1
            frame_id = obj.header.frame_id or msg.header.frame_id or self._camera_frame
            stamp = obj.header.stamp
            if stamp == rospy.Time(0):
                stamp = msg.header.stamp
            if not frame_id:
                processing_failures += 1
                rospy.logwarn_throttle(2.0, "Detection has no source frame_id")
                continue

            try:
                sector = self._object_sector(obj, frame_id, stamp)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException, RuntimeError) as exc:
                transformed_failures += 1
                processing_failures += 1
                rospy.logwarn_throttle(
                    2.0,
                    "Cannot transform visual obstacle %s -> %s: %s",
                    frame_id,
                    self.target_frame,
                    str(exc),
                )
                continue

            if sector is None:
                rejected += 1
                continue

            updated_beams += self._paint_sector(frame_ranges, sector)
            accepted += 1

        # If there were usable semantic/depth candidates but none could be
        # processed because TF/frame information failed, do not publish an
        # all-free scan.  Let the costmap source become stale instead.
        if eligible_objects > 0 and accepted == 0 and processing_failures > 0:
            rospy.logerr_throttle(
                2.0,
                "Vision candidates exist but cannot be transformed; keeping safety watchdog stale",
            )
            return

        self._commit_frame_ranges(frame_ranges, now)
        with self._lock:
            self._last_input_wall_time = now
            self._last_source_stamp = source_stamp

        if self.debug_log:
            rospy.loginfo_throttle(
                1.0,
                "Vision obstacles: input=%d accepted=%d rejected=%d tf_fail=%d active_updates=%d",
                len(msg.objects),
                accepted,
                rejected,
                transformed_failures,
                updated_beams,
            )

    def _snapshot_ranges(self, now: rospy.Time) -> List[float]:
        output = [math.inf] * self.beam_count
        with self._lock:
            for index in range(self.beam_count):
                if self._beam_expiry[index] > now:
                    output[index] = self._beam_ranges[index]
                else:
                    self._beam_ranges[index] = math.inf
        return output

    def _input_is_current(self, now: rospy.Time) -> bool:
        with self._lock:
            last_input = self._last_input_wall_time
        if last_input is None:
            return False
        return (now - last_input).to_sec() <= self.input_timeout_s

    def _publish_cloud(self, scan: LaserScan, ranges: Sequence[float]) -> None:
        points = []
        angle = scan.angle_min
        for value in ranges:
            if math.isfinite(value) and scan.range_min <= value <= scan.range_max:
                points.append((value * math.cos(angle), value * math.sin(angle), 0.25))
            angle += scan.angle_increment
        cloud = point_cloud2.create_cloud_xyz32(scan.header, points)
        self.cloud_pub.publish(cloud)

    def publish_timer_callback(self, _event) -> None:
        now = rospy.Time.now()

        # Fail safe: no fresh detector input means no observation publication.
        # Costmap sees the source as stale through expected_update_rate and
        # move_base stops instead of clearing the map and driving blind.
        if not self._input_is_current(now):
            rospy.logwarn_throttle(
                2.0,
                "Vision obstacle input timed out; scan publication paused",
            )
            return

        ranges = self._snapshot_ranges(now)

        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = self.target_frame
        scan.angle_min = self.scan_angle_min
        scan.angle_max = self.scan_angle_min + (
            self.beam_count - 1
        ) * self.scan_angle_increment
        scan.angle_increment = self.scan_angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / self.publish_rate_hz
        scan.range_min = self.scan_range_min
        scan.range_max = self.scan_range_max
        scan.ranges = ranges
        scan.intensities = []

        self.scan_pub.publish(scan)
        self._publish_cloud(scan, ranges)


def main() -> None:
    rospy.init_node("vision_obstacle_layer_node")
    try:
        VisionObstacleLayerNode()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("vision_obstacle_layer_node failed: %s", str(exc))
        raise


if __name__ == "__main__":
    main()
