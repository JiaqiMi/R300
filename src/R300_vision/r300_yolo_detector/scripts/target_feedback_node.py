#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish lightweight real-time target geolocation feedback for the Web UI.

This node deliberately does not subscribe to images and does not write files.
It reuses the same camera/body/geodetic convention as target_snapshot_recorder:

  /r300_vision/detections + /one_x/fix + /one_x/heading_deg
      -> /r300_vision/target_feedback_json (std_msgs/String)

The snapshot recorder remains responsible for candidate JPG/JSON/CSV storage.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Dict, List, Tuple

import rospy

from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64, String

from r300_vision_msgs.msg import DetectedObjectArray


class TargetFeedbackNode:
    EARTH_RADIUS_M = 6378137.0

    def __init__(self) -> None:
        self.detections_topic = str(
            rospy.get_param("~detections_topic", "/r300_vision/detections")
        )
        self.gps_topic = str(
            rospy.get_param("~gps_topic", "/one_x/fix")
        )
        self.heading_topic = str(
            rospy.get_param("~heading_topic", "/one_x/heading_deg")
        )
        self.feedback_topic = str(
            rospy.get_param(
                "~feedback_topic",
                "/r300_vision/target_feedback_json",
            )
        )

        self.minimum_confidence = float(
            rospy.get_param("~minimum_confidence", 0.25)
        )
        self.class_conf_thresholds = dict(
            rospy.get_param("~class_conf_thresholds", {})
        )
        self.max_gps_age_s = float(
            rospy.get_param("~max_gps_age_s", 1.0)
        )
        self.max_heading_age_s = float(
            rospy.get_param("~max_heading_age_s", 1.0)
        )
        self.publish_when_geolocation_invalid = bool(
            rospy.get_param(
                "~publish_when_geolocation_invalid",
                True,
            )
        )

        rotation_values = rospy.get_param(
            "~camera_to_body_rotation",
            [
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
            ],
        )
        translation_values = rospy.get_param(
            "~camera_in_body_translation_m",
            [0.30, 0.00, -0.10],
        )
        if len(rotation_values) != 9:
            raise ValueError("camera_to_body_rotation 必须包含9个数")
        if len(translation_values) != 3:
            raise ValueError("camera_in_body_translation_m 必须包含3个数")

        self.rotation_body_from_camera = [
            float(value) for value in rotation_values
        ]
        self.translation_body_from_camera = [
            float(value) for value in translation_values
        ]

        self.nav_lock = threading.Lock()
        self.vehicle_latitude = 0.0
        self.vehicle_longitude = 0.0
        self.vehicle_altitude = 0.0
        self.gps_valid = False
        self.gps_received_monotonic = 0.0
        self.heading_deg = 0.0
        self.heading_valid = False
        self.heading_received_monotonic = 0.0

        self.feedback_pub = rospy.Publisher(
            self.feedback_topic,
            String,
            queue_size=5,
        )
        self.gps_sub = rospy.Subscriber(
            self.gps_topic,
            NavSatFix,
            self.gps_callback,
            queue_size=10,
        )
        self.heading_sub = rospy.Subscriber(
            self.heading_topic,
            Float64,
            self.heading_callback,
            queue_size=10,
        )
        self.detections_sub = rospy.Subscriber(
            self.detections_topic,
            DetectedObjectArray,
            self.detections_callback,
            queue_size=5,
        )

        rospy.loginfo("R300 target feedback node started")
        rospy.loginfo("detections=%s", self.detections_topic)
        rospy.loginfo("gps=%s", self.gps_topic)
        rospy.loginfo("heading=%s", self.heading_topic)
        rospy.loginfo("feedback=%s", self.feedback_topic)
        rospy.loginfo(
            "R_body_camera=%s",
            self.rotation_body_from_camera,
        )
        rospy.loginfo(
            "t_body_camera=%s",
            self.translation_body_from_camera,
        )

    def gps_callback(self, msg: NavSatFix) -> None:
        latitude = float(msg.latitude)
        longitude = float(msg.longitude)
        altitude = float(msg.altitude)
        valid = (
            msg.status.status != NavSatStatus.STATUS_NO_FIX
            and math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
            and not (
                abs(latitude) < 1e-12
                and abs(longitude) < 1e-12
            )
        )
        with self.nav_lock:
            self.gps_received_monotonic = time.monotonic()
            if valid:
                self.vehicle_latitude = latitude
                self.vehicle_longitude = longitude
                self.vehicle_altitude = altitude if math.isfinite(altitude) else 0.0
                self.gps_valid = True
            else:
                self.gps_valid = False

    def heading_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        with self.nav_lock:
            self.heading_received_monotonic = time.monotonic()
            if math.isfinite(value):
                self.heading_deg = value % 360.0
                self.heading_valid = True
            else:
                self.heading_valid = False

    def detections_callback(self, msg: DetectedObjectArray) -> None:
        now = time.monotonic()
        with self.nav_lock:
            vehicle_lat = float(self.vehicle_latitude)
            vehicle_lon = float(self.vehicle_longitude)
            vehicle_alt = float(self.vehicle_altitude)
            gps_age_s = (
                now - self.gps_received_monotonic
                if self.gps_received_monotonic > 0.0
                else float("inf")
            )
            heading_age_s = (
                now - self.heading_received_monotonic
                if self.heading_received_monotonic > 0.0
                else float("inf")
            )
            gps_valid = bool(
                self.gps_valid and gps_age_s <= self.max_gps_age_s
            )
            heading_deg = float(self.heading_deg)
            heading_valid = bool(
                self.heading_valid
                and heading_age_s <= self.max_heading_age_s
            )

        geolocation_valid = bool(gps_valid and heading_valid)
        targets: List[Dict[str, Any]] = []

        for detection in msg.objects:
            class_id = int(detection.class_id)
            class_name = str(detection.class_name)
            confidence = float(detection.confidence)
            if confidence < self.get_class_threshold(class_id, class_name):
                continue

            bbox = [
                int(detection.x_min),
                int(detection.y_min),
                int(detection.x_max),
                int(detection.y_max),
            ]
            target: Dict[str, Any] = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "center_u": int(detection.center_u),
                "center_v": int(detection.center_v),
                "depth_valid": bool(detection.depth_valid),
                "depth_m": self.finite_or_none(float(detection.depth_m)),
                "geolocation_valid": False,
                "reason": "depth_or_camera_position_invalid",
            }

            camera_values = [
                float(detection.position.x),
                float(detection.position.y),
                float(detection.position.z),
            ]
            camera_valid = bool(
                detection.depth_valid
                and all(math.isfinite(value) for value in camera_values)
            )
            if not camera_valid:
                targets.append(target)
                continue

            r = self.rotation_body_from_camera
            t = self.translation_body_from_camera
            cx, cy, cz = camera_values
            body_point = (
                r[0] * cx + r[1] * cy + r[2] * cz + t[0],
                r[3] * cx + r[4] * cy + r[5] * cz + t[1],
                r[6] * cx + r[7] * cy + r[8] * cz + t[2],
            )
            (
                target_lat,
                target_lon,
                north_offset_m,
                east_offset_m,
            ) = self.compute_target_geodetic(
                vehicle_lat=vehicle_lat,
                vehicle_lon=vehicle_lon,
                gps_valid=geolocation_valid,
                heading_deg=heading_deg,
                body_point=body_point,
            )

            target.update(
                {
                    "camera_x_m": float(cx),
                    "camera_y_m": float(cy),
                    "camera_z_m": float(cz),
                    "body_x_m": float(body_point[0]),
                    "body_y_m": float(body_point[1]),
                    "body_z_m": float(body_point[2]),
                    "north_offset_m": float(north_offset_m),
                    "east_offset_m": float(east_offset_m),
                    "target_latitude": (
                        float(target_lat) if geolocation_valid else None
                    ),
                    "target_longitude": (
                        float(target_lon) if geolocation_valid else None
                    ),
                    "geolocation_valid": geolocation_valid,
                    "reason": (
                        "ok"
                        if geolocation_valid
                        else "gps_or_heading_invalid_or_stale"
                    ),
                }
            )
            targets.append(target)

        if not geolocation_valid and not self.publish_when_geolocation_invalid:
            return

        payload = {
            "stamp_sec": int(msg.header.stamp.secs),
            "stamp_nsec": int(msg.header.stamp.nsecs),
            "frame_id": str(msg.header.frame_id),
            "source_detection_topic": self.detections_topic,
            "gps_topic": self.gps_topic,
            "heading_topic": self.heading_topic,
            "gps_valid": gps_valid,
            "gps_age_s": self.finite_or_none(gps_age_s),
            "heading_valid": heading_valid,
            "heading_age_s": self.finite_or_none(heading_age_s),
            "geolocation_valid": geolocation_valid,
            "vehicle_latitude": vehicle_lat if gps_valid else None,
            "vehicle_longitude": vehicle_lon if gps_valid else None,
            "vehicle_altitude": vehicle_alt if gps_valid else None,
            "heading_deg": heading_deg if heading_valid else None,
            "target_count": len(targets),
            "targets": targets,
        }
        self.feedback_pub.publish(
            String(
                data=json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        )

    def compute_target_geodetic(
        self,
        vehicle_lat: float,
        vehicle_lon: float,
        gps_valid: bool,
        heading_deg: float,
        body_point: Tuple[float, float, float],
    ) -> Tuple[float, float, float, float]:
        heading_rad = math.radians(heading_deg)
        body_x = float(body_point[0])
        body_y = float(body_point[1])
        north_offset_m = (
            math.cos(heading_rad) * body_x
            - math.sin(heading_rad) * body_y
        )
        east_offset_m = (
            math.sin(heading_rad) * body_x
            + math.cos(heading_rad) * body_y
        )
        if not gps_valid:
            return 0.0, 0.0, north_offset_m, east_offset_m

        latitude_rad = math.radians(vehicle_lat)
        cos_latitude = math.cos(latitude_rad)
        if abs(cos_latitude) < 1e-9:
            return 0.0, 0.0, north_offset_m, east_offset_m

        target_latitude = vehicle_lat + math.degrees(
            north_offset_m / self.EARTH_RADIUS_M
        )
        target_longitude = vehicle_lon + math.degrees(
            east_offset_m
            / (self.EARTH_RADIUS_M * cos_latitude)
        )
        return (
            target_latitude,
            target_longitude,
            north_offset_m,
            east_offset_m,
        )

    def get_class_threshold(self, class_id: int, class_name: str) -> float:
        for key in (class_name, str(class_id), class_id):
            if key in self.class_conf_thresholds:
                return float(self.class_conf_thresholds[key])
        return self.minimum_confidence

    @staticmethod
    def finite_or_none(value: float):
        return value if math.isfinite(value) else None


def main() -> None:
    rospy.init_node("r300_target_feedback_node", anonymous=False)
    TargetFeedbackNode()
    rospy.spin()


if __name__ == "__main__":
    main()
