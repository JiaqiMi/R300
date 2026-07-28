#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import os
import re
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
    """每类保留置信度最高且地理位置互异的目标快照。"""

    EARTH_RADIUS_M = 6378137.0

    def __init__(self) -> None:
        self.image_topic = str(
            rospy.get_param("~image_topic", "/r300_vision/annotated_image")
        )
        self.detections_topic = str(
            rospy.get_param("~detections_topic", "/r300_vision/detections")
        )
        self.gps_topic = str(
            rospy.get_param("~gps_topic", "/vehicle/gps/fix")
        )
        self.heading_topic = str(
            rospy.get_param("~heading_topic", "/vehicle/heading_deg")
        )

        self.output_dir = Path(
            str(
                rospy.get_param(
                    "~output_dir",
                    "/home/explorer/r300_target_records",
                )
            )
        ).expanduser()

        self.top_k = int(rospy.get_param("~top_k", 3))
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
            rospy.get_param("~save_when_gps_invalid", True)
        )
        self.prefer_valid_gps = bool(
            rospy.get_param("~prefer_valid_gps", True)
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
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于0")
        if self.min_target_distance_m <= 0.0:
            raise ValueError("min_target_distance_m 必须大于0")

        self.rotation_body_from_camera = np.asarray(
            rotation_values,
            dtype=np.float64,
        ).reshape(3, 3)
        self.translation_body_from_camera = np.asarray(
            translation_values,
            dtype=np.float64,
        ).reshape(3)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "index.json"
        self.summary_path = self.output_dir / "summary.csv"

        self.nav_lock = threading.Lock()
        self.vehicle_latitude = self.default_latitude
        self.vehicle_longitude = self.default_longitude
        self.gps_valid = False
        self.heading_deg = self.default_heading_deg
        self.heading_valid = False

        self.state_lock = threading.Lock()
        self.records_by_class: Dict[str, List[Dict[str, Any]]] = (
            self.load_state()
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
        rospy.loginfo("Output directory: %s", str(self.output_dir))
        rospy.loginfo(
            "TopK=%d, minimum distance=%.2f m",
            self.top_k,
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
        latitude = float(msg.latitude)
        longitude = float(msg.longitude)
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
            if valid:
                self.vehicle_latitude = latitude
                self.vehicle_longitude = longitude
                self.gps_valid = True
            else:
                self.vehicle_latitude = self.default_latitude
                self.vehicle_longitude = self.default_longitude
                self.gps_valid = False

    def heading_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        with self.nav_lock:
            if math.isfinite(value):
                self.heading_deg = value % 360.0
                self.heading_valid = True
            else:
                self.heading_deg = self.default_heading_deg
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

        with self.nav_lock:
            vehicle_lat = float(self.vehicle_latitude)
            vehicle_lon = float(self.vehicle_longitude)
            gps_valid = bool(self.gps_valid)
            heading_deg = float(self.heading_deg)
            heading_valid = bool(self.heading_valid)

        for detection in detections_msg.objects:
            confidence = float(detection.confidence)
            class_id = int(detection.class_id)
            class_name = str(detection.class_name)

            threshold = self.get_class_threshold(class_id, class_name)
            if confidence < threshold:
                continue
            if self.require_depth_valid and not detection.depth_valid:
                continue
            if not gps_valid and not self.save_when_gps_invalid:
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
                gps_valid=gps_valid,
                heading_deg=heading_deg,
                body_point=body_point,
            )

            candidate: Dict[str, Any] = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "gps_valid": gps_valid,
                "heading_valid": heading_valid,
                "vehicle_latitude": vehicle_lat if gps_valid else 0.0,
                "vehicle_longitude": vehicle_lon if gps_valid else 0.0,
                "heading_deg": heading_deg,
                "target_latitude": target_lat if gps_valid else 0.0,
                "target_longitude": target_lon if gps_valid else 0.0,
                "north_offset_m": north_offset_m,
                "east_offset_m": east_offset_m,
                "camera_x_m": float(camera_point[0]),
                "camera_y_m": float(camera_point[1]),
                "camera_z_m": float(camera_point[2]),
                "body_x_m": float(body_point[0]),
                "body_y_m": float(body_point[1]),
                "body_z_m": float(body_point[2]),
                "depth_m": float(detection.depth_m),
                "bbox": [
                    int(detection.x_min),
                    int(detection.y_min),
                    int(detection.x_max),
                    int(detection.y_max),
                ],
                "center_u": int(detection.center_u),
                "center_v": int(detection.center_v),
                "frame_id": str(detections_msg.header.frame_id),
                "stamp_sec": int(detections_msg.header.stamp.secs),
                "stamp_nsec": int(detections_msg.header.stamp.nsecs),
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
        # Heading is clockwise from true north.
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

    def consider_candidate(
        self,
        image: np.ndarray,
        candidate: Dict[str, Any],
    ) -> None:
        class_key = self.make_class_key(
            candidate["class_id"],
            candidate["class_name"],
        )

        with self.state_lock:
            records = list(
                self.records_by_class.get(class_key, [])
            )

            same_index = self.find_same_target_index(
                records,
                candidate,
            )
            replace_index: Optional[int] = None

            if same_index is not None:
                if (
                    candidate["confidence"]
                    <= records[same_index]["confidence"]
                ):
                    return
                replace_index = same_index

            elif len(records) < self.top_k:
                replace_index = None

            else:
                invalid_indices = [
                    index
                    for index, record in enumerate(records)
                    if not record.get("gps_valid", False)
                ]

                if (
                    self.prefer_valid_gps
                    and candidate["gps_valid"]
                    and invalid_indices
                ):
                    replace_index = min(
                        invalid_indices,
                        key=lambda index: records[index]["confidence"],
                    )
                elif (
                    self.prefer_valid_gps
                    and not candidate["gps_valid"]
                    and not invalid_indices
                ):
                    return
                else:
                    lowest_index = min(
                        range(len(records)),
                        key=lambda index: records[index]["confidence"],
                    )
                    if (
                        candidate["confidence"]
                        <= records[lowest_index]["confidence"]
                    ):
                        return
                    replace_index = lowest_index

            saved_record = self.save_candidate_files(
                image,
                candidate,
            )

            if replace_index is None:
                records.append(saved_record)
            else:
                old_record = records[replace_index]
                self.delete_record_files(old_record)
                records[replace_index] = saved_record

            records.sort(
                key=lambda record: float(record["confidence"]),
                reverse=True,
            )
            self.records_by_class[class_key] = records[: self.top_k]
            self.save_state()

            rank = self.records_by_class[class_key].index(saved_record) + 1
            rospy.loginfo(
                "Saved target: class=%s id=%d conf=%.3f rank=%d "
                "lat=%.8f lon=%.8f gps=%s",
                candidate["class_name"],
                candidate["class_id"],
                candidate["confidence"],
                rank,
                candidate["target_latitude"],
                candidate["target_longitude"],
                candidate["gps_valid"],
            )

    def find_same_target_index(
        self,
        records: List[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> Optional[int]:
        for index, record in enumerate(records):
            record_gps_valid = bool(record.get("gps_valid", False))
            candidate_gps_valid = bool(candidate["gps_valid"])

            if record_gps_valid and candidate_gps_valid:
                distance_m = self.haversine_distance_m(
                    float(record["target_latitude"]),
                    float(record["target_longitude"]),
                    float(candidate["target_latitude"]),
                    float(candidate["target_longitude"]),
                )
                if distance_m < self.min_target_distance_m:
                    return index
            elif not record_gps_valid and not candidate_gps_valid:
                # GPS无效时无法验证5 m间隔；所有0,0记录视为同一目标。
                return index

        return None

    def save_candidate_files(
        self,
        image: np.ndarray,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        class_name_safe = self.sanitize_name(candidate["class_name"])
        class_dir = self.output_dir / class_name_safe
        class_dir.mkdir(parents=True, exist_ok=True)

        timestamp = (
            f"{candidate['stamp_sec']}_"
            f"{candidate['stamp_nsec']:09d}"
        )
        record_id = (
            f"{candidate['class_id']:02d}_{class_name_safe}_"
            f"{candidate['confidence']:.4f}_{timestamp}_{time.time_ns()}"
        )

        image_path = class_dir / f"{record_id}.jpg"
        metadata_path = class_dir / f"{record_id}.json"

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
            image_path.relative_to(self.output_dir)
        )
        saved_record["metadata_file"] = str(
            metadata_path.relative_to(self.output_dir)
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

    def delete_record_files(self, record: Dict[str, Any]) -> None:
        for key in ("image_file", "metadata_file"):
            relative_path = record.get(key)
            if not relative_path:
                continue
            path = self.output_dir / str(relative_path)
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                rospy.logwarn(
                    "Failed to remove old file %s: %r",
                    str(path),
                    exc,
                )

    def load_state(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.index_path.exists():
            return {}

        try:
            with self.index_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            classes = payload.get("classes", {})
            cleaned: Dict[str, List[Dict[str, Any]]] = {}

            for class_key, records in classes.items():
                valid_records = []
                for record in records:
                    image_file = record.get("image_file")
                    metadata_file = record.get("metadata_file")
                    if (
                        image_file
                        and metadata_file
                        and (self.output_dir / image_file).exists()
                        and (self.output_dir / metadata_file).exists()
                    ):
                        valid_records.append(record)

                if valid_records:
                    valid_records.sort(
                        key=lambda item: float(item["confidence"]),
                        reverse=True,
                    )
                    cleaned[class_key] = valid_records[: self.top_k]

            return cleaned
        except Exception as exc:
            rospy.logwarn(
                "Failed to load index.json, starting empty: %r",
                exc,
            )
            return {}

    def save_state(self) -> None:
        payload = {
            "version": 1,
            "top_k": self.top_k,
            "min_target_distance_m": self.min_target_distance_m,
            "classes": self.records_by_class,
        }

        temporary_index = self.index_path.with_suffix(".json.tmp")
        with temporary_index.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_index, self.index_path)
        self.write_summary_csv()

    def write_summary_csv(self) -> None:
        fields = [
            "rank",
            "class_id",
            "class_name",
            "confidence",
            "gps_valid",
            "heading_valid",
            "vehicle_latitude",
            "vehicle_longitude",
            "heading_deg",
            "target_latitude",
            "target_longitude",
            "north_offset_m",
            "east_offset_m",
            "camera_x_m",
            "camera_y_m",
            "camera_z_m",
            "body_x_m",
            "body_y_m",
            "body_z_m",
            "depth_m",
            "image_file",
            "metadata_file",
            "record_id",
            "stamp_sec",
            "stamp_nsec",
        ]

        temporary_summary = self.summary_path.with_suffix(".csv.tmp")
        with temporary_summary.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            sorted_classes = sorted(
                self.records_by_class.items(),
                key=lambda item: int(item[0].split(":", 1)[0]),
            )

            for _, records in sorted_classes:
                for rank, record in enumerate(records, start=1):
                    row = {
                        field: record.get(field, "")
                        for field in fields
                    }
                    row["rank"] = rank
                    writer.writerow(row)

        os.replace(temporary_summary, self.summary_path)

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
    def make_class_key(class_id: int, class_name: str) -> str:
        return f"{class_id}:{class_name}"

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
