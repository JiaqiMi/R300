#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Jetson/aarch64 环境中优先导入 torch，避免 libgomp TLS 加载问题。
import torch
from ultralytics import YOLO

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import message_filters
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image

from r300_vision_msgs.msg import (
    DetectedObject,
    DetectedObjectArray,
)


# ============================================================
# 对外统一类别编号
# ============================================================
# 原有13类编号保持不变；泊车状态追加为13、14，避免与 tire/barrel 冲突。
GLOBAL_CLASS_NAME_TO_ID: Dict[str, int] = {
    "tire": 0,
    "barrel": 1,
    "vehicle": 2,
    "smoke": 3,
    "trench": 4,
    "puddle": 5,
    "person": 6,
    "rockfall": 7,
    "park": 8,
    "chevro_left": 9,
    "chevro_right": 10,
    "sandbag": 11,
    "crater": 12,
    "parking_occupied": 13,
    "parking_empty": 14,
}

GLOBAL_CLASS_ID_TO_NAME: Dict[int, str] = {
    class_id: class_name
    for class_name, class_id in GLOBAL_CLASS_NAME_TO_ID.items()
}

# ============================================================
# 模型局部ID -> 统一全局ID
# ============================================================
# 模型1复用现有6类模型，但泊车节点只运行 local_id=4 的 park。
MODEL1_LOCAL_TO_GLOBAL: Dict[int, int] = {
    4: 8,  # park
}

# 模型2是3分类模型，但泊车节点只运行 local_id=0、1；local_id=2忽略。
MODEL2_LOCAL_TO_GLOBAL: Dict[int, int] = {
    0: 13,  # parking_occupied
    1: 14,  # parking_empty
}

MODEL1_PREDICT_CLASSES = sorted(MODEL1_LOCAL_TO_GLOBAL.keys())
MODEL2_PREDICT_CLASSES = sorted(MODEL2_LOCAL_TO_GLOBAL.keys())


@dataclass(frozen=True)
class CandidateDetection:
    """映射到统一类别编号后的检测结果。"""

    source_model: str
    local_class_id: int
    global_class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


class ParkingYoloDepthNode:
    """
    D435i + 泊车双模型 + 深度定位节点。

    模型1：复用现有6类模型，仅检测 local_id=4 的 park。
    模型2：3分类停车框模型，仅使用：
        local_id=0 -> parking_occupied
        local_id=1 -> parking_empty
        local_id=2 -> 忽略

    两个模型互不覆盖、互不抑制，只将结果映射后拼接到同一消息中。

    默认输出为独立泊车话题：
        /r300_parking_vision/detections
        /r300_parking_vision/annotated_image
        /r300_parking_vision/target_point
    """

    def __init__(self) -> None:
        # ============================================================
        # 1. 模型与推理参数
        # ============================================================
        self.model1_path = str(
            rospy.get_param("~model1_path", "")
        )
        self.model2_path = str(
            rospy.get_param("~model2_path", "")
        )

        self.model1_conf_threshold = float(
            rospy.get_param("~model1_conf_threshold", 0.20)
        )
        self.model1_iou_threshold = float(
            rospy.get_param("~model1_iou_threshold", 0.45)
        )
        raw_model1_thresholds = rospy.get_param(
            "~model1_class_conf_thresholds",
            {},
        )
        self.model1_class_conf_thresholds: Dict[str, float] = {
            str(class_name): float(threshold)
            for class_name, threshold in raw_model1_thresholds.items()
        }

        self.model2_conf_threshold = float(
            rospy.get_param("~model2_conf_threshold", 0.20)
        )
        self.model2_iou_threshold = float(
            rospy.get_param("~model2_iou_threshold", 0.45)
        )
        raw_model2_thresholds = rospy.get_param(
            "~model2_class_conf_thresholds",
            {},
        )
        self.model2_class_conf_thresholds: Dict[str, float] = {
            str(class_name): float(threshold)
            for class_name, threshold in raw_model2_thresholds.items()
        }

        self.imgsz = int(
            rospy.get_param("~imgsz", 640)
        )
        self.device = str(
            rospy.get_param("~device", "0")
        )
        self.max_det = int(
            rospy.get_param("~max_det", 100)
        )
        self.infer_hz = float(
            rospy.get_param("~infer_hz", 4.0)
        )
        self.show_model_source = bool(
            rospy.get_param("~show_model_source", True)
        )

        # ============================================================
        # 2. ROS话题与深度参数
        # ============================================================
        self.rgb_topic = str(
            rospy.get_param(
                "~rgb_topic",
                "/camera/color/image_raw",
            )
        )
        self.depth_topic = str(
            rospy.get_param(
                "~depth_topic",
                "/camera/aligned_depth_to_color/image_raw",
            )
        )
        self.camera_info_topic = str(
            rospy.get_param(
                "~camera_info_topic",
                "/camera/color/camera_info",
            )
        )
        self.annotated_topic = str(
            rospy.get_param(
                "~annotated_topic",
                "/r300_parking_vision/annotated_image",
            )
        )
        self.detections_topic = str(
            rospy.get_param(
                "~detections_topic",
                "/r300_parking_vision/detections",
            )
        )
        self.target_point_topic = str(
            rospy.get_param(
                "~target_point_topic",
                "/r300_parking_vision/target_point",
            )
        )

        self.sync_queue_size = int(
            rospy.get_param("~sync_queue_size", 8)
        )
        self.sync_slop = float(
            rospy.get_param("~sync_slop", 0.08)
        )
        self.depth_window_radius = int(
            rospy.get_param("~depth_window_radius", 4)
        )
        self.min_valid_depth_pixels = int(
            rospy.get_param("~min_valid_depth_pixels", 5)
        )
        self.depth_scale_16u = float(
            rospy.get_param("~depth_scale_16u", 0.001)
        )
        self.min_depth_m = float(
            rospy.get_param("~min_depth_m", 0.15)
        )
        self.max_depth_m = float(
            rospy.get_param("~max_depth_m", 20.0)
        )

        target_classes = rospy.get_param(
            "~target_classes",
            ["parking_empty"],
        )
        self.target_classes: Set[str] = {
            str(class_name)
            for class_name in target_classes
        }
        self.target_policy = str(
            rospy.get_param("~target_policy", "nearest")
        )

        self._validate_parameters()

        # ============================================================
        # 3. 加载两个模型
        # ============================================================
        rospy.loginfo(
            "Torch=%s, CUDA=%s, device=%s",
            torch.__version__,
            torch.cuda.is_available(),
            self.device,
        )
        if torch.cuda.is_available():
            rospy.loginfo(
                "GPU: %s",
                torch.cuda.get_device_name(0),
            )

        rospy.loginfo(
            "Loading parking area model: %s",
            self.model1_path,
        )
        self.model1 = YOLO(self.model1_path)

        rospy.loginfo(
            "Loading parking status model: %s",
            self.model2_path,
        )
        self.model2 = YOLO(self.model2_path)

        rospy.loginfo(
            "Model1 classes: %s",
            str(self.model1.names),
        )
        rospy.loginfo(
            "Model2 classes: %s",
            str(self.model2.names),
        )

        self._validate_model_classes()

        # predict()先使用各模型最终阈值中的最小值保留候选框，
        # 解析时再按类别进行最终过滤。
        model1_threshold_values = [self.model1_conf_threshold]
        model1_threshold_values.extend(
            self.model1_class_conf_thresholds.values()
        )
        self.model1_predict_conf = min(model1_threshold_values)

        model2_threshold_values = [self.model2_conf_threshold]
        model2_threshold_values.extend(
            self.model2_class_conf_thresholds.values()
        )
        self.model2_predict_conf = min(model2_threshold_values)

        # ============================================================
        # 4. CameraInfo内参
        # ============================================================
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1,
        )

        # ============================================================
        # 5. ROS发布接口
        # ============================================================
        self.bridge = CvBridge()

        self.annotated_pub = rospy.Publisher(
            self.annotated_topic,
            Image,
            queue_size=1,
        )
        self.detections_pub = rospy.Publisher(
            self.detections_topic,
            DetectedObjectArray,
            queue_size=10,
        )
        self.target_point_pub = rospy.Publisher(
            self.target_point_topic,
            PointStamped,
            queue_size=10,
        )

        # ============================================================
        # 6. RGB与深度同步
        # ============================================================
        self.rgb_sub = message_filters.Subscriber(
            self.rgb_topic,
            Image,
            queue_size=1,
        )
        self.depth_sub = message_filters.Subscriber(
            self.depth_topic,
            Image,
            queue_size=1,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        # ============================================================
        # 7. 只保留最新同步帧，避免双模型推理积压
        # ============================================================
        self.frame_lock = threading.Lock()
        self.latest_rgb_msg: Optional[Image] = None
        self.latest_depth_msg: Optional[Image] = None

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.infer_hz),
            self.inference_timer_callback,
        )

        self.frame_counter = 0

        rospy.loginfo(
            "Parking YOLO node ready: infer_hz=%.2f, imgsz=%d",
            self.infer_hz,
            self.imgsz,
        )
        rospy.loginfo("RGB topic: %s", self.rgb_topic)
        rospy.loginfo("Depth topic: %s", self.depth_topic)
        rospy.loginfo("Detections topic: %s", self.detections_topic)
        rospy.loginfo("Target classes: %s", sorted(self.target_classes))

    # ============================================================
    # 参数与模型检查
    # ============================================================
    def _validate_parameters(self) -> None:
        if not self.model1_path:
            raise RuntimeError("参数 ~model1_path 不能为空")
        if not self.model2_path:
            raise RuntimeError("参数 ~model2_path 不能为空")

        for model_path in (self.model1_path, self.model2_path):
            if not Path(model_path).is_file():
                raise FileNotFoundError(
                    f"模型文件不存在：{model_path}"
                )

        if self.infer_hz <= 0.0:
            raise ValueError("参数 ~infer_hz 必须大于0")
        if self.max_det <= 0:
            raise ValueError("参数 ~max_det 必须大于0")

        for parameter_name, value in (
            ("model1_conf_threshold", self.model1_conf_threshold),
            ("model1_iou_threshold", self.model1_iou_threshold),
            ("model2_conf_threshold", self.model2_conf_threshold),
            ("model2_iou_threshold", self.model2_iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"参数 ~{parameter_name} 必须在[0, 1]范围内"
                )

        self._validate_class_thresholds(
            "model1_class_conf_thresholds",
            self.model1_class_conf_thresholds,
            {"park"},
        )
        self._validate_class_thresholds(
            "model2_class_conf_thresholds",
            self.model2_class_conf_thresholds,
            {"parking_occupied", "parking_empty"},
        )

        allowed_target_classes = {
            "park",
            "parking_occupied",
            "parking_empty",
        }
        unknown_target_classes = (
            self.target_classes - allowed_target_classes
        )
        if unknown_target_classes:
            raise ValueError(
                "参数 ~target_classes 包含未知类别："
                f"{sorted(unknown_target_classes)}"
            )

        if self.target_policy not in {
            "nearest",
            "highest_confidence",
        }:
            raise ValueError(
                "参数 ~target_policy 只能是 nearest 或 "
                "highest_confidence"
            )

        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "配置要求GPU推理，但torch.cuda.is_available()为False"
            )

    @staticmethod
    def _validate_class_thresholds(
        parameter_name: str,
        thresholds: Dict[str, float],
        expected_names: Set[str],
    ) -> None:
        unknown_names = set(thresholds) - expected_names
        if unknown_names:
            raise ValueError(
                f"参数 ~{parameter_name} 包含未知类别："
                f"{sorted(unknown_names)}；允许类别：{sorted(expected_names)}"
            )

        for class_name, threshold in thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"参数 ~{parameter_name}/{class_name} "
                    "必须在[0, 1]范围内"
                )

    @staticmethod
    def _normalized_model_names(model: YOLO) -> Dict[int, str]:
        names = model.names
        if isinstance(names, dict):
            return {
                int(class_id): str(class_name)
                for class_id, class_name in names.items()
            }
        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(names)
        }

    def _validate_model_classes(self) -> None:
        model1_names = self._normalized_model_names(self.model1)
        model2_names = self._normalized_model_names(self.model2)

        # 模型1只使用 local_id=4，但该ID必须确实是 park。
        if 4 not in model1_names:
            raise RuntimeError(
                "模型1缺少 local_id=4，无法检测 park。"
            )
        if model1_names[4] != "park":
            raise RuntimeError(
                "模型1类别顺序错误：local_id=4 应为 park，"
                f"实际为 {model1_names[4]}。"
            )

        # 模型2按用户指定的ID语义工作。必须至少存在0、1；第三类只记录并忽略。
        if 0 not in model2_names or 1 not in model2_names:
            raise RuntimeError(
                "模型2必须包含 local_id=0 和 local_id=1。"
            )
        if len(model2_names) != 3:
            rospy.logwarn(
                "Model2 expected 3 classes, actual=%d: %s. "
                "本节点仍只使用local_id 0和1。",
                len(model2_names),
                str(model2_names),
            )

        rospy.loginfo(
            "Model1 mapping: local 4 (%s) -> global 8 (park)",
            model1_names[4],
        )
        rospy.loginfo(
            "Model2 mapping: local 0 (%s) -> global 13 "
            "(parking_occupied)",
            model2_names[0],
        )
        rospy.loginfo(
            "Model2 mapping: local 1 (%s) -> global 14 "
            "(parking_empty)",
            model2_names[1],
        )
        if 2 in model2_names:
            rospy.loginfo(
                "Model2 local 2 (%s) is ignored",
                model2_names[2],
            )

    # ============================================================
    # ROS图像与相机参数
    # ============================================================
    @staticmethod
    def bgr_numpy_to_ros_image(
        image: np.ndarray,
        header,
    ) -> Image:
        """手动构造bgr8消息，绕开部分Jetson cv_bridge输出兼容问题。"""
        if image is None:
            raise ValueError("输出图像为空")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"BGR图像必须是HxWx3，当前shape={image.shape}"
            )

        image = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = image.shape[:2]

        msg = Image()
        msg.header = header
        msg.height = height
        msg.width = width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = image.tobytes()
        return msg

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if msg.K[0] <= 0.0 or msg.K[4] <= 0.0:
            return

        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])

    def synced_callback(
        self,
        rgb_msg: Image,
        depth_msg: Image,
    ) -> None:
        with self.frame_lock:
            self.latest_rgb_msg = rgb_msg
            self.latest_depth_msg = depth_msg

    # ============================================================
    # 深度处理
    # ============================================================
    def convert_depth_values_to_meters(
        self,
        values: np.ndarray,
        encoding: str,
    ) -> np.ndarray:
        original_dtype = values.dtype
        values = values.astype(np.float32, copy=False)

        if encoding in ("16UC1", "mono16"):
            values = values * self.depth_scale_16u
        elif encoding == "32FC1":
            pass
        elif np.issubdtype(original_dtype, np.integer):
            values = values * self.depth_scale_16u

        return values

    def filter_depth_values(
        self,
        roi: np.ndarray,
        encoding: str,
    ) -> np.ndarray:
        if roi.size == 0:
            return np.empty((0,), dtype=np.float32)

        values = self.convert_depth_values_to_meters(
            roi.reshape(-1),
            encoding,
        )
        mask = (
            np.isfinite(values)
            & (values >= self.min_depth_m)
            & (values <= self.max_depth_m)
        )
        return values[mask]

    def estimate_depth(
        self,
        depth_image: np.ndarray,
        encoding: str,
        center_u: int,
        center_v: int,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        image_h, image_w = depth_image.shape[:2]
        radius = self.depth_window_radius

        u0 = max(0, center_u - radius)
        u1 = min(image_w, center_u + radius + 1)
        v0 = max(0, center_v - radius)
        v1 = min(image_h, center_v + radius + 1)

        center_values = self.filter_depth_values(
            depth_image[v0:v1, u0:u1],
            encoding,
        )
        if center_values.size >= self.min_valid_depth_pixels:
            return float(np.median(center_values))

        x_min, y_min, x_max, y_max = bbox
        bbox_w = max(1, x_max - x_min)
        bbox_h = max(1, y_max - y_min)

        inner_x0 = int(x_min + 0.25 * bbox_w)
        inner_x1 = int(x_max - 0.25 * bbox_w)
        inner_y0 = int(y_min + 0.25 * bbox_h)
        inner_y1 = int(y_max - 0.25 * bbox_h)

        inner_x0 = max(0, min(image_w - 1, inner_x0))
        inner_x1 = max(inner_x0 + 1, min(image_w, inner_x1))
        inner_y0 = max(0, min(image_h - 1, inner_y0))
        inner_y1 = max(inner_y0 + 1, min(image_h, inner_y1))

        inner_values = self.filter_depth_values(
            depth_image[inner_y0:inner_y1, inner_x0:inner_x1],
            encoding,
        )
        if inner_values.size < self.min_valid_depth_pixels:
            return None

        return float(np.median(inner_values))

    def deproject_pixel(
        self,
        u: int,
        v: int,
        depth_m: float,
    ) -> Tuple[float, float, float]:
        if None in (self.fx, self.fy, self.cx, self.cy):
            raise RuntimeError("尚未收到有效CameraInfo")

        x_m = (float(u) - self.cx) * depth_m / self.fx
        y_m = (float(v) - self.cy) * depth_m / self.fy
        return x_m, y_m, depth_m

    # ============================================================
    # 检测结果处理
    # ============================================================
    def parse_model_result(
        self,
        result,
        source_model: str,
        local_to_global: Dict[int, int],
        base_conf_threshold: float,
        class_conf_thresholds: Dict[str, float],
    ) -> List[CandidateDetection]:
        candidates: List[CandidateDetection] = []
        boxes = result.boxes

        if boxes is None:
            return candidates

        xyxy_array = boxes.xyxy.detach().cpu().numpy()
        class_array = boxes.cls.detach().cpu().numpy()
        confidence_array = boxes.conf.detach().cpu().numpy()

        for xyxy, class_value, confidence_value in zip(
            xyxy_array,
            class_array,
            confidence_array,
        ):
            local_id = int(class_value)

            # 尽管predict已经用classes过滤，这里仍做防御性检查。
            if local_id not in local_to_global:
                rospy.logwarn_throttle(
                    5.0,
                    "%s returned ignored local class id=%d",
                    source_model,
                    local_id,
                )
                continue

            global_id = local_to_global[local_id]
            class_name = GLOBAL_CLASS_ID_TO_NAME[global_id]
            confidence = float(confidence_value)

            final_threshold = class_conf_thresholds.get(
                class_name,
                base_conf_threshold,
            )
            if confidence < final_threshold:
                continue

            candidates.append(
                CandidateDetection(
                    source_model=source_model,
                    local_class_id=local_id,
                    global_class_id=global_id,
                    class_name=class_name,
                    confidence=confidence,
                    xyxy=(
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                    ),
                )
            )

        return candidates

    def select_target(
        self,
        objects,
    ) -> Optional[DetectedObject]:
        candidates = []

        for obj in objects:
            if not obj.depth_valid:
                continue
            if (
                self.target_classes
                and obj.class_name not in self.target_classes
            ):
                continue
            candidates.append(obj)

        if not candidates:
            return None

        if self.target_policy == "highest_confidence":
            return max(
                candidates,
                key=lambda item: item.confidence,
            )

        return min(
            candidates,
            key=lambda item: item.depth_m,
        )

    @staticmethod
    def class_color(global_class_id: int) -> Tuple[int, int, int]:
        return (
            int((37 * global_class_id + 80) % 255),
            int((97 * global_class_id + 130) % 255),
            int((53 * global_class_id + 200) % 255),
        )

    def draw_detection(
        self,
        image: np.ndarray,
        candidate: CandidateDetection,
        bbox: Tuple[int, int, int, int],
        depth_valid: bool,
        depth_m: float,
        x_m: float,
        y_m: float,
    ) -> None:
        x_min, y_min, x_max, y_max = bbox
        color = self.class_color(candidate.global_class_id)

        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            color,
            2,
        )

        source_text = (
            f"[{candidate.source_model}] "
            if self.show_model_source
            else ""
        )
        label_text = (
            f"{source_text}{candidate.class_name} "
            f"{candidate.confidence:.2f}"
        )
        if depth_valid:
            label_text += (
                f" {depth_m:.2f}m X{x_m:.2f} Y{y_m:.2f}"
            )

        text_y = max(20, y_min - 7)
        cv2.putText(
            image,
            label_text,
            (x_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # ============================================================
    # 双模型推理与统一发布
    # ============================================================
    def inference_timer_callback(self, _event) -> None:
        with self.frame_lock:
            if self.latest_rgb_msg is None or self.latest_depth_msg is None:
                return

            rgb_msg = self.latest_rgb_msg
            depth_msg = self.latest_depth_msg
            self.latest_rgb_msg = None
            self.latest_depth_msg = None

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="bgr8",
            )
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except CvBridgeError as exc:
            rospy.logerr("CvBridge conversion failed: %s", str(exc))
            return

        if depth_image.ndim != 2:
            rospy.logerr_throttle(
                5.0,
                "深度图不是单通道，encoding=%s shape=%s",
                depth_msg.encoding,
                str(depth_image.shape),
            )
            return

        total_start = time.perf_counter()

        try:
            model1_start = time.perf_counter()
            model1_results = self.model1.predict(
                source=rgb_image,
                conf=self.model1_predict_conf,
                iou=self.model1_iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                classes=MODEL1_PREDICT_CLASSES,
                max_det=self.max_det,
                verbose=False,
            )
            model1_ms = (
                time.perf_counter() - model1_start
            ) * 1000.0

            model2_start = time.perf_counter()
            model2_results = self.model2.predict(
                source=rgb_image,
                conf=self.model2_predict_conf,
                iou=self.model2_iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                classes=MODEL2_PREDICT_CLASSES,
                max_det=self.max_det,
                verbose=False,
            )
            model2_ms = (
                time.perf_counter() - model2_start
            ) * 1000.0

        except Exception as exc:
            rospy.logerr("Parking dual YOLO inference failed: %r", exc)
            return

        model1_candidates = self.parse_model_result(
            model1_results[0],
            source_model="PARK",
            local_to_global=MODEL1_LOCAL_TO_GLOBAL,
            base_conf_threshold=self.model1_conf_threshold,
            class_conf_thresholds=self.model1_class_conf_thresholds,
        )
        model2_candidates = self.parse_model_result(
            model2_results[0],
            source_model="SLOT",
            local_to_global=MODEL2_LOCAL_TO_GLOBAL,
            base_conf_threshold=self.model2_conf_threshold,
            class_conf_thresholds=self.model2_class_conf_thresholds,
        )

        # 两个模型独立，结果只做拼接，不做覆盖、互斥或跨模型NMS。
        all_candidates = model1_candidates + model2_candidates
        all_candidates.sort(
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )

        output_msg = DetectedObjectArray()
        output_msg.header = rgb_msg.header

        annotated_image = np.ascontiguousarray(
            rgb_image.copy(),
            dtype=np.uint8,
        )
        image_h, image_w = rgb_image.shape[:2]

        for candidate in all_candidates:
            x1, y1, x2, y2 = candidate.xyxy
            x_min = int(max(0, min(image_w - 1, round(x1))))
            y_min = int(max(0, min(image_h - 1, round(y1))))
            x_max = int(max(0, min(image_w - 1, round(x2))))
            y_max = int(max(0, min(image_h - 1, round(y2))))

            if x_max <= x_min or y_max <= y_min:
                continue

            center_u = int(round((x_min + x_max) / 2.0))
            center_v = int(round((y_min + y_max) / 2.0))

            detection = DetectedObject()
            detection.header = rgb_msg.header
            detection.class_id = candidate.global_class_id
            detection.class_name = candidate.class_name
            detection.confidence = candidate.confidence
            detection.x_min = x_min
            detection.y_min = y_min
            detection.x_max = x_max
            detection.y_max = y_max
            detection.center_u = center_u
            detection.center_v = center_v

            depth_m = self.estimate_depth(
                depth_image,
                depth_msg.encoding,
                center_u,
                center_v,
                (x_min, y_min, x_max, y_max),
            )

            depth_valid = False
            draw_depth_m = float("nan")
            draw_x_m = float("nan")
            draw_y_m = float("nan")

            if depth_m is not None and self.fx is not None:
                try:
                    x_m, y_m, z_m = self.deproject_pixel(
                        center_u,
                        center_v,
                        depth_m,
                    )
                    detection.depth_valid = True
                    detection.depth_m = depth_m
                    detection.position.x = x_m
                    detection.position.y = y_m
                    detection.position.z = z_m

                    depth_valid = True
                    draw_depth_m = depth_m
                    draw_x_m = x_m
                    draw_y_m = y_m
                except Exception as exc:
                    rospy.logwarn_throttle(
                        2.0,
                        "3D反投影失败: %s",
                        str(exc),
                    )
                    detection.depth_valid = False
            else:
                detection.depth_valid = False

            if not detection.depth_valid:
                detection.depth_m = float("nan")
                detection.position.x = float("nan")
                detection.position.y = float("nan")
                detection.position.z = float("nan")

            output_msg.objects.append(detection)
            self.draw_detection(
                annotated_image,
                candidate,
                (x_min, y_min, x_max, y_max),
                depth_valid,
                draw_depth_m,
                draw_x_m,
                draw_y_m,
            )

        self.detections_pub.publish(output_msg)

        selected_target = self.select_target(output_msg.objects)
        if selected_target is not None:
            target_point_msg = PointStamped()
            target_point_msg.header = selected_target.header
            target_point_msg.point = selected_target.position
            self.target_point_pub.publish(target_point_msg)

        try:
            annotated_msg = self.bgr_numpy_to_ros_image(
                annotated_image,
                rgb_msg.header,
            )
            self.annotated_pub.publish(annotated_msg)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Annotated image publish failed: %r",
                exc,
            )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            occupied_count = sum(
                1
                for item in model2_candidates
                if item.class_name == "parking_occupied"
            )
            empty_count = sum(
                1
                for item in model2_candidates
                if item.class_name == "parking_empty"
            )

            rospy.loginfo(
                "park=%d occupied=%d empty=%d final=%d | "
                "M1=%.1fms M2=%.1fms total=%.1fms depth=%s",
                len(model1_candidates),
                occupied_count,
                empty_count,
                len(output_msg.objects),
                model1_ms,
                model2_ms,
                total_ms,
                depth_msg.encoding,
            )


def main() -> None:
    rospy.init_node(
        "r300_autonomous_parking_perception",
        anonymous=False,
    )
    ParkingYoloDepthNode()
    rospy.loginfo("R300 isolated parking perception node started")
    rospy.spin()


if __name__ == "__main__":
    main()
