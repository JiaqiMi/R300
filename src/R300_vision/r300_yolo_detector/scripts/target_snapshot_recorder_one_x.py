#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rospy

from sensor_msgs.msg import Image, NavSatFix, NavSatStatus
from std_msgs.msg import Float64

from r300_vision_msgs.msg import DetectedObjectArray


class TargetSnapshotRecorder:
    """保存候选障碍物库，并维护全局 Top10 提交结果。"""

    # WGS-84 椭球参数。目标通常距离车辆仅数米到数十米，
    # 使用局部子午圈/卯酉圈曲率半径换算，比固定球半径更准确。
    WGS84_A_M = 6378137.0
    WGS84_F = 1.0 / 298.257223563
    WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
    EARTH_RADIUS_M = WGS84_A_M
    SUMMARY_FIELDS = [
        "rank",
        "class_id",
        "class_name",
        "confidence",
        "bbox_area",
        "gps_valid",
        "heading_valid",
        "geolocation_valid",
        "vehicle_latitude",
        "vehicle_longitude",
        "vehicle_altitude",
        "gps_status",
        "gps_age_s",
        "raw_heading_deg",
        "heading_deg",
        "heading_age_s",
        "target_latitude",
        "target_longitude",
        "target_altitude",
        "north_offset_m",
        "east_offset_m",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "body_x_m",
        "body_y_m",
        "body_z_m",
        "depth_m",
        "bbox",
        "image_file",
        "metadata_file",
        "record_id",
        "stamp_sec",
        "stamp_nsec",
        "timestamp_ns",
    ]

    def __init__(self) -> None:
        self.image_topic = str(
            rospy.get_param("~image_topic", "/r300_vision/annotated_image")
        )
        self.detections_topic = str(
            rospy.get_param("~detections_topic", "/r300_vision/detections")
        )
        self.gps_topic = str(
            rospy.get_param("~gps_topic", "/one_x/fix")
        )
        self.heading_topic = str(
            rospy.get_param("~heading_topic", "/one_x/heading_deg")
        )
        self.max_gps_age_s = float(
            rospy.get_param("~max_gps_age_s", 1.0)
        )
        self.max_heading_age_s = float(
            rospy.get_param("~max_heading_age_s", 1.0)
        )
        # /one_x/heading_deg 的定义在项目驱动中为：北0°、东90°、
        # 顺时针为正。该偏置用于后期现场标定，默认不修正。
        self.heading_offset_deg = float(
            rospy.get_param("~heading_offset_deg", 0.0)
        )

        self.output_dir = Path(
            str(
                rospy.get_param(
                    "~output_dir",
                    "/home/explorer/r300_target_records",
                )
            )
        ).expanduser()

        self.candidate_max_per_target = int(
            rospy.get_param("~candidate_max_per_target", 20)
        )
        self.candidate_max_total = int(
            rospy.get_param("~candidate_max_total", 500)
        )
        self.submit_max = int(rospy.get_param("~submit_max", 10))
        self.min_target_distance_m = float(
            rospy.get_param("~min_target_distance_m", 5.0)
        )
        self.minimum_confidence = float(
            rospy.get_param("~minimum_confidence", 0.25)
        )
        self.class_conf_thresholds = dict(
            rospy.get_param("~class_conf_thresholds", {})
        )
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))
        self.save_when_gps_invalid = bool(
            rospy.get_param("~save_when_gps_invalid", False)
        )
        self.require_depth_valid = bool(
            rospy.get_param("~require_depth_valid", True)
        )

        self.sync_queue_size = int(
            rospy.get_param("~sync_queue_size", 10)
        )
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))

        self.default_latitude = float(
            rospy.get_param("~default_latitude", 0.0)
        )
        self.default_longitude = float(
            rospy.get_param("~default_longitude", 0.0)
        )
        self.default_heading_deg = float(
            rospy.get_param("~default_heading_deg", 0.0)
        )

        # Camera optical: X right, Y down, Z forward
        # Body FRD:      X forward, Y right, Z down
        # Xb=Zc, Yb=Xc, Zb=Yc
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
        if self.candidate_max_per_target <= 0:
            raise ValueError("candidate_max_per_target 必须大于0")
        if self.candidate_max_total <= 0:
            raise ValueError("candidate_max_total 必须大于0")
        if self.submit_max <= 0:
            raise ValueError("submit_max 必须大于0")
        if self.min_target_distance_m <= 0.0:
            raise ValueError("min_target_distance_m 必须大于0")
        if self.max_gps_age_s <= 0.0:
            raise ValueError("max_gps_age_s 必须大于0")
        if self.max_heading_age_s <= 0.0:
            raise ValueError("max_heading_age_s 必须大于0")

        self.rotation_body_from_camera = np.asarray(
            rotation_values,
            dtype=np.float64,
        ).reshape(3, 3)
        self.translation_body_from_camera = np.asarray(
            translation_values,
            dtype=np.float64,
        ).reshape(3)

        self.candidate_dir = self.output_dir / "candidate_records"
        self.submit_dir = self.output_dir / "submit_results"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.submit_dir.mkdir(parents=True, exist_ok=True)

        self.candidate_index_path = self.candidate_dir / "index.json"
        self.candidate_summary_path = self.candidate_dir / "summary.csv"
        self.submit_index_path = self.submit_dir / "index.json"
        self.submit_summary_path = self.submit_dir / "summary.csv"

        self.nav_lock = threading.Lock()
        self.vehicle_latitude = self.default_latitude
        self.vehicle_longitude = self.default_longitude
        self.vehicle_altitude = 0.0
        self.gps_status = int(NavSatStatus.STATUS_NO_FIX)
        self.gps_valid = False
        self.gps_received_monotonic = 0.0
        self.gps_message_stamp_sec = 0
        self.gps_message_stamp_nsec = 0

        self.raw_heading_deg = self.default_heading_deg
        self.heading_deg = (
            self.default_heading_deg + self.heading_offset_deg
        ) % 360.0
        self.heading_valid = False
        self.heading_received_monotonic = 0.0

        self.state_lock = threading.Lock()
        self.candidate_records: List[Dict[str, Any]] = (
            self.load_record_list(
                self.candidate_index_path,
                self.candidate_dir,
            )
        )
        self.submit_records: List[Dict[str, Any]] = (
            self.load_record_list(
                self.submit_index_path,
                self.submit_dir,
            )
        )
        self.enforce_candidate_max_total()
        self.rebuild_submit_results()
        self.save_all_state()

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

        self.image_sub = message_filters.Subscriber(
            self.image_topic,
            Image,
            queue_size=2,
        )
        self.detections_sub = message_filters.Subscriber(
            self.detections_topic,
            DetectedObjectArray,
            queue_size=10,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.detections_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo("Target snapshot recorder started")
        rospy.loginfo("GPS topic: %s [sensor_msgs/NavSatFix]", self.gps_topic)
        rospy.loginfo("Heading topic: %s [std_msgs/Float64]", self.heading_topic)
        rospy.loginfo(
            "Heading convention: North=0 deg, East=90 deg, clockwise; "
            "offset=%.3f deg",
            self.heading_offset_deg,
        )
        rospy.loginfo("Output directory: %s", str(self.output_dir))
        rospy.loginfo(
            "candidate_max_per_target=%d, candidate_max_total=%d, "
            "submit_max=%d, minimum distance=%.2f m",
            self.candidate_max_per_target,
            self.candidate_max_total,
            self.submit_max,
            self.min_target_distance_m,
        )
        rospy.loginfo(
            "R_body_camera=%s",
            np.array2string(self.rotation_body_from_camera, precision=3),
        )
        rospy.loginfo(
            "t_body_camera=%s",
            np.array2string(self.translation_body_from_camera, precision=3),
        )

    def gps_callback(self, msg: NavSatFix) -> None:
        """读取 /one_x/fix 发布的车辆实时WGS-84位置。"""
        latitude = float(msg.latitude)
        longitude = float(msg.longitude)
        altitude = float(msg.altitude)
        status = int(msg.status.status)

        coordinates_finite = (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
        )
        nonzero_fix = not (
            abs(latitude) < 1.0e-12
            and abs(longitude) < 1.0e-12
        )
        valid = bool(
            status != NavSatStatus.STATUS_NO_FIX
            and coordinates_finite
            and nonzero_fix
        )

        received_monotonic = time.monotonic()
        with self.nav_lock:
            self.gps_received_monotonic = received_monotonic
            self.gps_status = status
            self.gps_message_stamp_sec = int(msg.header.stamp.secs)
            self.gps_message_stamp_nsec = int(msg.header.stamp.nsecs)

            if valid:
                self.vehicle_latitude = latitude
                self.vehicle_longitude = longitude
                self.vehicle_altitude = (
                    altitude if math.isfinite(altitude) else 0.0
                )
                self.gps_valid = True
            else:
                self.vehicle_latitude = self.default_latitude
                self.vehicle_longitude = self.default_longitude
                self.vehicle_altitude = 0.0
                self.gps_valid = False

        if valid:
            rospy.loginfo_throttle(
                5.0,
                "one_x fix valid: lat=%.9f lon=%.9f alt=%.3f status=%d",
                latitude,
                longitude,
                altitude if math.isfinite(altitude) else 0.0,
                status,
            )
        else:
            rospy.logwarn_throttle(
                2.0,
                "one_x fix invalid: lat=%.9f lon=%.9f status=%d; "
                "target snapshots requiring geolocation will not be saved",
                latitude,
                longitude,
                status,
            )

    def heading_callback(self, msg: Float64) -> None:
        """读取 /one_x/heading_deg：北0°、东90°、顺时针为正。"""
        raw_value = float(msg.data)
        with self.nav_lock:
            self.heading_received_monotonic = time.monotonic()
            if math.isfinite(raw_value):
                self.raw_heading_deg = raw_value % 360.0
                self.heading_deg = (
                    self.raw_heading_deg + self.heading_offset_deg
                ) % 360.0
                self.heading_valid = True
            else:
                self.raw_heading_deg = self.default_heading_deg
                self.heading_deg = (
                    self.default_heading_deg + self.heading_offset_deg
                ) % 360.0
                self.heading_valid = False

    def synced_callback(
        self,
        image_msg: Image,
        detections_msg: DetectedObjectArray,
    ) -> None:
        try:
            image = self.ros_image_to_bgr(image_msg)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Image conversion failed: %r",
                exc,
            )
            return

        now_monotonic = time.monotonic()
        with self.nav_lock:
            vehicle_lat = float(self.vehicle_latitude)
            vehicle_lon = float(self.vehicle_longitude)
            vehicle_alt = float(self.vehicle_altitude)
            gps_status = int(self.gps_status)
            gps_message_stamp_sec = int(self.gps_message_stamp_sec)
            gps_message_stamp_nsec = int(self.gps_message_stamp_nsec)
            gps_age_s = (
                now_monotonic - self.gps_received_monotonic
                if self.gps_received_monotonic > 0.0
                else float("inf")
            )
            heading_age_s = (
                now_monotonic - self.heading_received_monotonic
                if self.heading_received_monotonic > 0.0
                else float("inf")
            )
            gps_valid = bool(
                self.gps_valid and gps_age_s <= self.max_gps_age_s
            )
            raw_heading_deg = float(self.raw_heading_deg)
            heading_deg = float(self.heading_deg)
            heading_valid = bool(
                self.heading_valid
                and heading_age_s <= self.max_heading_age_s
            )

        geolocation_valid = bool(gps_valid and heading_valid)
        if not geolocation_valid:
            rospy.logwarn_throttle(
                2.0,
                "Geolocation unavailable: gps_valid=%s gps_age=%.3fs "
                "heading_valid=%s heading_age=%.3fs lat=%.9f lon=%.9f",
                gps_valid,
                gps_age_s,
                heading_valid,
                heading_age_s,
                vehicle_lat,
                vehicle_lon,
            )

        for detection in detections_msg.objects:
            confidence = float(detection.confidence)
            class_id = int(detection.class_id)
            class_name = str(detection.class_name)

            threshold = self.get_class_threshold(class_id, class_name)
            if confidence < threshold:
                continue
            if self.require_depth_valid and not detection.depth_valid:
                continue
            if not geolocation_valid and not self.save_when_gps_invalid:
                continue

            camera_point = np.asarray(
                [
                    float(detection.position.x),
                    float(detection.position.y),
                    float(detection.position.z),
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(camera_point)):
                continue

            body_point = (
                self.rotation_body_from_camera @ camera_point
                + self.translation_body_from_camera
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

            # 车体系采用FRD，Z向下，因此目标海拔=车辆海拔-Zb。
            target_altitude = (
                vehicle_alt - float(body_point[2])
                if geolocation_valid
                else 0.0
            )

            bbox = [
                int(detection.x_min),
                int(detection.y_min),
                int(detection.x_max),
                int(detection.y_max),
            ]
            bbox_area = max(0, bbox[2] - bbox[0]) * max(
                0,
                bbox[3] - bbox[1],
            )
            stamp_sec = int(detections_msg.header.stamp.secs)
            stamp_nsec = int(detections_msg.header.stamp.nsecs)

            candidate: Dict[str, Any] = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox_area": int(bbox_area),
                "gps_valid": gps_valid,
                "heading_valid": heading_valid,
                "geolocation_valid": geolocation_valid,
                "vehicle_latitude": vehicle_lat if gps_valid else 0.0,
                "vehicle_longitude": vehicle_lon if gps_valid else 0.0,
                "vehicle_altitude": vehicle_alt if gps_valid else 0.0,
                "gps_status": gps_status,
                "gps_age_s": gps_age_s if math.isfinite(gps_age_s) else None,
                "gps_message_stamp_sec": gps_message_stamp_sec,
                "gps_message_stamp_nsec": gps_message_stamp_nsec,
                "raw_heading_deg": raw_heading_deg,
                "heading_deg": heading_deg,
                "heading_age_s": (
                    heading_age_s if math.isfinite(heading_age_s) else None
                ),
                "target_latitude": target_lat if geolocation_valid else 0.0,
                "target_longitude": target_lon if geolocation_valid else 0.0,
                "target_altitude": target_altitude,
                "north_offset_m": north_offset_m,
                "east_offset_m": east_offset_m,
                "camera_x_m": float(camera_point[0]),
                "camera_y_m": float(camera_point[1]),
                "camera_z_m": float(camera_point[2]),
                "body_x_m": float(body_point[0]),
                "body_y_m": float(body_point[1]),
                "body_z_m": float(body_point[2]),
                "depth_m": float(detection.depth_m),
                "bbox": bbox,
                "center_u": int(detection.center_u),
                "center_v": int(detection.center_v),
                "frame_id": str(detections_msg.header.frame_id),
                "stamp_sec": stamp_sec,
                "stamp_nsec": stamp_nsec,
                "timestamp_ns": stamp_sec * 1_000_000_000 + stamp_nsec,
            }

            self.consider_candidate(image, candidate)

    def compute_target_geodetic(
        self,
        vehicle_lat: float,
        vehicle_lon: float,
        gps_valid: bool,
        heading_deg: float,
        body_point: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        将相机测得的目标位置换算为WGS-84目标经纬度。

        车体系FRD：X前、Y右、Z下。
        /one_x/heading_deg：北0°、东90°、顺时针为正。
        """
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
        sin_lat = math.sin(latitude_rad)
        cos_lat = math.cos(latitude_rad)
        if abs(cos_lat) < 1.0e-12:
            return 0.0, 0.0, north_offset_m, east_offset_m

        denominator = math.sqrt(
            1.0 - self.WGS84_E2 * sin_lat * sin_lat
        )
        prime_vertical_radius_m = self.WGS84_A_M / denominator
        meridian_radius_m = (
            self.WGS84_A_M
            * (1.0 - self.WGS84_E2)
            / (denominator ** 3)
        )

        target_latitude = vehicle_lat + math.degrees(
            north_offset_m / meridian_radius_m
        )
        target_longitude = vehicle_lon + math.degrees(
            east_offset_m / (prime_vertical_radius_m * cos_lat)
        )

        return (
            target_latitude,
            target_longitude,
            north_offset_m,
            east_offset_m,
        )

    @staticmethod
    def haversine_distance_m(
        latitude_1: float,
        longitude_1: float,
        latitude_2: float,
        longitude_2: float,
    ) -> float:
        radius_m = 6378137.0
        lat1 = math.radians(latitude_1)
        lat2 = math.radians(latitude_2)
        d_lat = math.radians(latitude_2 - latitude_1)
        d_lon = math.radians(longitude_2 - longitude_1)
        a = (
            math.sin(d_lat / 2.0) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(d_lon / 2.0) ** 2
        )
        return 2.0 * radius_m * math.asin(
            min(1.0, math.sqrt(a))
        )

    @staticmethod
    def ranking_key(record: Dict[str, Any]) -> Tuple[float, float, int]:
        """越小越优：confidence高、面积大、时间早。"""
        return (
            -float(record["confidence"]),
            -float(record.get("bbox_area", 0)),
            int(record.get("timestamp_ns", 0)),
        )

    @staticmethod
    def eviction_key(record: Dict[str, Any]) -> Tuple[float, float, int]:
        """越小越差、越先删除：confidence低、面积小、时间旧。"""
        return (
            float(record["confidence"]),
            float(record.get("bbox_area", 0)),
            int(record.get("timestamp_ns", 0)),
        )

    def consider_candidate(
        self,
        image: np.ndarray,
        candidate: Dict[str, Any],
    ) -> None:
        with self.state_lock:
            saved = self.save_to_candidate_records(image, candidate)
            if saved is None:
                return
            self.enforce_candidate_max_total()
            self.rebuild_submit_results()
            self.save_all_state()
            rospy.loginfo(
                "Saved candidate: class=%s id=%d conf=%.3f area=%d "
                "lat=%.8f lon=%.8f gps=%s candidates=%d submit=%d",
                candidate["class_name"],
                candidate["class_id"],
                candidate["confidence"],
                candidate["bbox_area"],
                candidate["target_latitude"],
                candidate["target_longitude"],
                candidate["gps_valid"],
                len(self.candidate_records),
                len(self.submit_records),
            )

    def save_to_candidate_records(
        self,
        image: np.ndarray,
        candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        same_indices = self.find_same_target_indices(
            self.candidate_records,
            candidate,
        )

        if len(same_indices) < self.candidate_max_per_target:
            saved_record = self.save_record_files(
                image,
                candidate,
                self.candidate_dir,
                subdir=self.sanitize_name(candidate["class_name"]),
            )
            self.candidate_records.append(saved_record)
            return saved_record

        worst_index = max(
            same_indices,
            key=lambda index: self.ranking_key(
                self.candidate_records[index]
            ),
        )
        worst_record = self.candidate_records[worst_index]
        if self.ranking_key(candidate) >= self.ranking_key(worst_record):
            return None

        saved_record = self.save_record_files(
            image,
            candidate,
            self.candidate_dir,
            subdir=self.sanitize_name(candidate["class_name"]),
        )
        self.delete_record_files(worst_record, self.candidate_dir)
        self.candidate_records[worst_index] = saved_record
        return saved_record

    def enforce_candidate_max_total(self) -> None:
        while len(self.candidate_records) > self.candidate_max_total:
            worst_index = min(
                range(len(self.candidate_records)),
                key=lambda index: self.eviction_key(
                    self.candidate_records[index]
                ),
            )
            worst_record = self.candidate_records.pop(worst_index)
            self.delete_record_files(worst_record, self.candidate_dir)
            rospy.loginfo(
                "Evicted candidate by global limit: id=%s conf=%.3f "
                "area=%s total=%d",
                worst_record.get("record_id"),
                float(worst_record.get("confidence", 0.0)),
                worst_record.get("bbox_area"),
                len(self.candidate_records),
            )

    def clear_submit_result_files(self) -> None:
        """清理 submit_results 下旧结果图/元数据，保留 index.json 与 summary.csv。"""
        if not self.submit_dir.exists():
            self.submit_dir.mkdir(parents=True, exist_ok=True)
            return

        preserve_names = {
            self.submit_index_path.name,  # index.json
            self.submit_summary_path.name,  # summary.csv
        }

        for path in self.submit_dir.iterdir():
            if not path.is_file():
                continue
            if path.name in preserve_names:
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".json"}:
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                rospy.logwarn(
                    "Failed to clear submit file %s: %r",
                    str(path),
                    exc,
                )

    def rebuild_submit_results(self) -> None:
        # 先清空目录，再按当前 TopN 完整重建，保证磁盘文件数 <= submit_max。
        self.clear_submit_result_files()

        desired = sorted(
            self.candidate_records,
            key=self.ranking_key,
        )[: self.submit_max]

        new_submit_records: List[Dict[str, Any]] = []
        for record in desired:
            new_submit_records.append(
                self.copy_record_to_submit(record)
            )

        self.submit_records = new_submit_records

    def copy_record_to_submit(
        self,
        candidate_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        record_id = str(candidate_record["record_id"])
        source_image = self.candidate_dir / str(
            candidate_record["image_file"]
        )

        image_name = f"{record_id}.jpg"
        metadata_name = f"{record_id}.json"
        target_image = self.submit_dir / image_name
        target_metadata = self.submit_dir / metadata_name

        if not source_image.exists():
            raise RuntimeError(
                f"候选图片不存在，无法生成提交结果：{source_image}"
            )

        shutil.copy2(str(source_image), str(target_image))

        submit_record = dict(candidate_record)
        submit_record["image_file"] = image_name
        submit_record["metadata_file"] = metadata_name

        with target_metadata.open("w", encoding="utf-8") as file:
            json.dump(
                submit_record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return submit_record

    def find_same_target_indices(
        self,
        records: List[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> List[int]:
        indices: List[int] = []
        for index, record in enumerate(records):
            if self.is_same_target(record, candidate):
                indices.append(index)
        return indices

    def is_same_target(
        self,
        record: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> bool:
        if int(record["class_id"]) != int(candidate["class_id"]):
            return False
        if str(record["class_name"]) != str(candidate["class_name"]):
            return False

        record_gps_valid = bool(
            record.get(
                "geolocation_valid",
                record.get("gps_valid", False),
            )
        )
        candidate_gps_valid = bool(
            candidate.get(
                "geolocation_valid",
                candidate.get("gps_valid", False),
            )
        )

        if record_gps_valid and candidate_gps_valid:
            distance_m = self.haversine_distance_m(
                float(record["target_latitude"]),
                float(record["target_longitude"]),
                float(candidate["target_latitude"]),
                float(candidate["target_longitude"]),
            )
            return distance_m < self.min_target_distance_m

        if not record_gps_valid and not candidate_gps_valid:
            # GPS无效时无法验证地理间隔；同类无效GPS记录视为同一目标簇。
            return True

        return False

    def save_record_files(
        self,
        image: np.ndarray,
        candidate: Dict[str, Any],
        base_dir: Path,
        subdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if subdir:
            record_dir = base_dir / subdir
        else:
            record_dir = base_dir
        record_dir.mkdir(parents=True, exist_ok=True)

        class_name_safe = self.sanitize_name(candidate["class_name"])
        timestamp = (
            f"{candidate['stamp_sec']}_"
            f"{candidate['stamp_nsec']:09d}"
        )
        record_id = (
            f"{candidate['class_id']:02d}_{class_name_safe}_"
            f"{candidate['confidence']:.4f}_{timestamp}_{time.time_ns()}"
        )

        image_path = record_dir / f"{record_id}.jpg"
        metadata_path = record_dir / f"{record_id}.json"

        output_image = self.draw_candidate(image, candidate)
        write_ok = cv2.imwrite(
            str(image_path),
            output_image,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not write_ok:
            raise RuntimeError(f"无法保存图片：{image_path}")

        saved_record = dict(candidate)
        saved_record["image_file"] = str(
            image_path.relative_to(base_dir)
        )
        saved_record["metadata_file"] = str(
            metadata_path.relative_to(base_dir)
        )
        saved_record["record_id"] = record_id

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                saved_record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return saved_record

    def draw_candidate(
        self,
        image: np.ndarray,
        candidate: Dict[str, Any],
    ) -> np.ndarray:
        output = np.ascontiguousarray(image.copy(), dtype=np.uint8)
        height, width = output.shape[:2]
        x_min, y_min, x_max, y_max = candidate["bbox"]
        x_min = max(0, min(width - 1, x_min))
        x_max = max(0, min(width - 1, x_max))
        y_min = max(0, min(height - 1, y_min))
        y_max = max(0, min(height - 1, y_max))

        cv2.rectangle(
            output,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 255),
            4,
        )

        lines = [
            (
                f"{candidate['class_name']} "
                f"conf={candidate['confidence']:.3f}"
            ),
            (
                f"LAT={candidate['target_latitude']:.8f} "
                f"LON={candidate['target_longitude']:.8f}"
            ),
            (
                f"ALT={candidate.get('target_altitude', 0.0):.2f}m "
                f"HDG={candidate['heading_deg']:.2f}deg"
            ),
            (
                "BODY FRD="
                f"({candidate['body_x_m']:.2f}, "
                f"{candidate['body_y_m']:.2f}, "
                f"{candidate['body_z_m']:.2f}) m"
            ),
        ]

        text_x = max(5, min(width - 10, x_min))
        text_y = max(25, y_min - 60)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        line_height = 24

        max_text_width = 0
        for line in lines:
            text_size, _ = cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness,
            )
            max_text_width = max(max_text_width, text_size[0])

        box_x2 = min(width - 1, text_x + max_text_width + 12)
        box_y1 = max(0, text_y - 20)
        box_y2 = min(
            height - 1,
            text_y + line_height * (len(lines) - 1) + 8,
        )

        cv2.rectangle(
            output,
            (text_x - 4, box_y1),
            (box_x2, box_y2),
            (0, 0, 0),
            -1,
        )

        for index, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (text_x, text_y + index * line_height),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return output

    def delete_record_files(
        self,
        record: Dict[str, Any],
        base_dir: Path,
    ) -> None:
        for key in ("image_file", "metadata_file"):
            relative_path = record.get(key)
            if not relative_path:
                continue
            path = base_dir / str(relative_path)
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                rospy.logwarn(
                    "Failed to remove old file %s: %r",
                    str(path),
                    exc,
                )

    def load_record_list(
        self,
        index_path: Path,
        base_dir: Path,
    ) -> List[Dict[str, Any]]:
        if not index_path.exists():
            return []

        try:
            with index_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            records = payload.get("records", [])
            cleaned: List[Dict[str, Any]] = []
            for record in records:
                image_file = record.get("image_file")
                metadata_file = record.get("metadata_file")
                if not image_file or not metadata_file:
                    continue
                if not (base_dir / image_file).exists():
                    continue
                if not (base_dir / metadata_file).exists():
                    continue

                record = dict(record)
                if "bbox_area" not in record and "bbox" in record:
                    bbox = record["bbox"]
                    record["bbox_area"] = max(
                        0,
                        int(bbox[2]) - int(bbox[0]),
                    ) * max(
                        0,
                        int(bbox[3]) - int(bbox[1]),
                    )
                if "timestamp_ns" not in record:
                    record["timestamp_ns"] = (
                        int(record.get("stamp_sec", 0)) * 1_000_000_000
                        + int(record.get("stamp_nsec", 0))
                    )
                cleaned.append(record)

            cleaned.sort(key=self.ranking_key)
            return cleaned
        except Exception as exc:
            rospy.logwarn(
                "Failed to load %s, starting empty: %r",
                str(index_path),
                exc,
            )
            return []

    def save_all_state(self) -> None:
        self.save_store_state(
            index_path=self.candidate_index_path,
            summary_path=self.candidate_summary_path,
            records=self.candidate_records,
            store_name="candidate_records",
            extra={
                "candidate_max_per_target": self.candidate_max_per_target,
                "candidate_max_total": self.candidate_max_total,
                "min_target_distance_m": self.min_target_distance_m,
            },
        )
        self.save_store_state(
            index_path=self.submit_index_path,
            summary_path=self.submit_summary_path,
            records=self.submit_records,
            store_name="submit_results",
            extra={
                "submit_max": self.submit_max,
            },
        )

    def save_store_state(
        self,
        index_path: Path,
        summary_path: Path,
        records: List[Dict[str, Any]],
        store_name: str,
        extra: Dict[str, Any],
    ) -> None:
        payload = {
            "version": 2,
            "store": store_name,
            "count": len(records),
            "records": records,
        }
        payload.update(extra)

        temporary_index = index_path.with_suffix(".json.tmp")
        with temporary_index.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_index, index_path)
        self.write_summary_csv(summary_path, records)

    def write_summary_csv(
        self,
        summary_path: Path,
        records: List[Dict[str, Any]],
    ) -> None:
        temporary_summary = summary_path.with_suffix(".csv.tmp")
        with temporary_summary.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.SUMMARY_FIELDS,
            )
            writer.writeheader()

            ranked = sorted(records, key=self.ranking_key)
            for rank, record in enumerate(ranked, start=1):
                row = {
                    field: record.get(field, "")
                    for field in self.SUMMARY_FIELDS
                }
                row["rank"] = rank
                if isinstance(row.get("bbox"), list):
                    row["bbox"] = json.dumps(
                        row["bbox"],
                        ensure_ascii=False,
                    )
                writer.writerow(row)

        os.replace(temporary_summary, summary_path)

    def get_class_threshold(
        self,
        class_id: int,
        class_name: str,
    ) -> float:
        possible_keys = (class_name, str(class_id), class_id)
        for key in possible_keys:
            if key in self.class_conf_thresholds:
                return float(self.class_conf_thresholds[key])
        return self.minimum_confidence

    @staticmethod
    def sanitize_name(name: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
        return cleaned or "unknown"

    @staticmethod
    def ros_image_to_bgr(msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        if height <= 0 or width <= 0 or step <= 0:
            raise ValueError("Invalid ROS image dimensions")

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        expected_size = height * step
        if raw.size < expected_size:
            raise ValueError("ROS image data is shorter than height*step")

        rows = raw[:expected_size].reshape(height, step)
        encoding = msg.encoding.lower()

        if encoding == "bgr8":
            image = rows[:, : width * 3].reshape(height, width, 3)
        elif encoding == "rgb8":
            rgb = rows[:, : width * 3].reshape(height, width, 3)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(
                f"Unsupported image encoding: {msg.encoding}"
            )

        return np.ascontiguousarray(image, dtype=np.uint8)


def main() -> None:
    rospy.init_node(
        "r300_target_snapshot_recorder",
        anonymous=False,
    )
    TargetSnapshotRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
