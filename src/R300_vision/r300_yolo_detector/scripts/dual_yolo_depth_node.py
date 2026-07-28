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


# 统一对外发布的全局类别编号。下游模块继续使用这套编号，无需修改。
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
    "crater":12
}

GLOBAL_CLASS_ID_TO_NAME: Dict[int, str] = {
    class_id: class_name
    for class_name, class_id in GLOBAL_CLASS_NAME_TO_ID.items()
}

# 模型1原始类别编号等于全局编号，但只允许发布这些类别。
# 其余 0、1、5、7、8、11 交给模型2负责，模型1结果直接忽略。
MODEL1_ALLOWED_GLOBAL_IDS: Set[int] = {
    2,   # vehicle
    3,   # smoke
    4,   # trench
    6,   # person
    9,   # chevro_left
    10,  # chevro_right
}

# 模型1训练时将六个类别重新映射为 0~5，需要恢复到全局编号。
MODEL1_LOCAL_TO_GLOBAL: Dict[int, int] = {
    0: 2,   # vehicle
    1: 3,   # smoke
    2: 4,   # trench
    3: 6,   # person
    4: 8,   # park
    5: 12,  # crater
}

# 模型2训练时将六个类别重新映射为 0~5，需要恢复到全局编号。
MODEL2_LOCAL_TO_GLOBAL: Dict[int, int] = {
    0: 0,   # tire
    1: 1,   # barrel
    2: 5,   # puddle
    3: 7,   # rockfall
    4: 9,   # chevro_left
    5: 10,  # chevro_right
    6: 11,  # sandbag
}


@dataclass(frozen=True)
class CandidateDetection:
    """两个模型统一后的候选检测框。"""

    source_model: str
    local_class_id: int
    global_class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


class DualYoloDepthNode:
    """
    D435i + 双 YOLO 模型 + 深度定位节点。

    模型1：c11_260711.pt，包含完整12类，但只保留：
        vehicle、smoke、trench、person、chevro_left、chevro_right

    模型2：model0722.pt，包含六个重映射类别，负责：
        tire、barrel、puddle、rockfall、park、sandbag

    最终仍只发布：
        /r300_vision/detections
        /r300_vision/annotated_image
        /r300_vision/target_point
    """

    def __init__(self) -> None:
        # ============================================================
        # 1. 模型参数
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

        self.model2_conf_threshold = float(
            rospy.get_param("~model2_conf_threshold", 0.20)
        )
        self.model2_iou_threshold = float(
            rospy.get_param("~model2_iou_threshold", 0.45)
        )


        # 模型2支持按类别设置最终置信度阈值。
        raw_model1_thresholds = rospy.get_param(
            "~model1class_conf_thresholds",
            {},
        )
        self.model1_class_conf_thresholds: Dict[str, float] = {
            str(class_name): float(threshold)
            for class_name, threshold in raw_model1_thresholds.items()
        }
        # 模型2支持按类别设置最终置信度阈值。
        raw_model2_thresholds = rospy.get_param(
            "~model2_class_conf_thresholds",
            {},
        )
        self.model2_class_conf_thresholds: Dict[str, float] = {
            str(class_name): float(threshold)
            for class_name, threshold in raw_model2_thresholds.items()
        }

        # 只要模型2框与模型1框的 IoU 高于该值，就认为模型2覆盖模型1。
        self.specialist_override_iou = float(
            rospy.get_param("~specialist_override_iou", 0.50)
        )

        # 可限制只覆盖模型1中的某些类别；空列表表示允许覆盖全部模型1类别。
        raw_override_ids = rospy.get_param(
            "~override_model1_global_ids",
            [],
        )
        self.override_model1_global_ids: Set[int] = {
            int(class_id)
            for class_id in raw_override_ids
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
                "/r300_vision/annotated_image",
            )
        )
        self.detections_topic = str(
            rospy.get_param(
                "~detections_topic",
                "/r300_vision/detections",
            )
        )
        self.target_point_topic = str(
            rospy.get_param(
                "~target_point_topic",
                "/r300_vision/target_point",
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
            [],
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
            "Loading model1: %s",
            self.model1_path,
        )
        self.model1 = YOLO(self.model1_path)

        rospy.loginfo(
            "Loading model2: %s",
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

        self._validate_model_class_mappings()

        # 模型2 predict() 使用所有类别最终阈值中的最小值，之后再按类别过滤。
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
        # 7. 只保留最新同步帧，避免双模型推理造成队列积压
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
            "Dual YOLO node ready: infer_hz=%.2f, imgsz=%d",
            self.infer_hz,
            self.imgsz,
        )
        rospy.loginfo("RGB topic: %s", self.rgb_topic)
        rospy.loginfo("Depth topic: %s", self.depth_topic)
        rospy.loginfo("CameraInfo topic: %s", self.camera_info_topic)

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
        if not 0.0 <= self.specialist_override_iou <= 1.0:
            raise ValueError(
                "参数 ~specialist_override_iou 必须在[0, 1]范围内"
            )
        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "配置要求GPU推理，但torch.cuda.is_available()为False"
            )

    @staticmethod
    def _model_class_name(model: YOLO, class_id: int) -> str:
        names = model.names
        if isinstance(names, dict):
            return str(names[class_id])
        return str(names[class_id])

    def _validate_model_class_mappings(self) -> None:
        # # 模型1按全局ID读取，检查关键类名称是否一致。
        # for class_id in sorted(MODEL1_ALLOWED_GLOBAL_IDS):
        #     expected_name = GLOBAL_CLASS_ID_TO_NAME[class_id]
        #     try:
        #         actual_name = self._model_class_name(
        #             self.model1,
        #             class_id,
        #         )
        #     except (KeyError, IndexError, TypeError):
        #         raise RuntimeError(
        #             f"模型1缺少类别ID {class_id} ({expected_name})"
        #         )

        #     if actual_name != expected_name:
        #         rospy.logwarn(
        #             "Model1 class mismatch: id=%d expected=%s actual=%s",
        #             class_id,
        #             expected_name,
        #             actual_name,
        #         )

        # 模型1严格按本地ID映射回全局ID。
        for local_id, global_id in MODEL1_LOCAL_TO_GLOBAL.items():
            expected_name = GLOBAL_CLASS_ID_TO_NAME[global_id]
            try:
                actual_name = self._model_class_name(
                    self.model1,
                    local_id,
                )
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(
                    f"模型1缺少本地类别ID {local_id}，期望类别 {expected_name}"
                )

            if actual_name != expected_name:
                rospy.logwarn(
                    "Model1 class mismatch: local_id=%d expected=%s actual=%s; "
                    "仍按预设ID映射发布",
                    local_id,
                    expected_name,
                    actual_name,
                )

        # 模型2严格按本地ID映射回全局ID。
        for local_id, global_id in MODEL2_LOCAL_TO_GLOBAL.items():
            expected_name = GLOBAL_CLASS_ID_TO_NAME[global_id]
            try:
                actual_name = self._model_class_name(
                    self.model2,
                    local_id,
                )
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(
                    f"模型2缺少本地类别ID {local_id}，期望类别 {expected_name}"
                )

            if actual_name != expected_name:
                rospy.logwarn(
                    "Model2 class mismatch: local_id=%d expected=%s actual=%s; "
                    "仍按预设ID映射发布",
                    local_id,
                    expected_name,
                    actual_name,
                )

    @staticmethod
    def bgr_numpy_to_ros_image(
        image: np.ndarray,
        header,
    ) -> Image:
        """绕开 cv_bridge 输出端兼容问题，手动构造 bgr8 ROS 图像。"""
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
        """读取RGB相机内参矩阵K。"""
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
        """只缓存最新同步帧，不在订阅回调中执行耗时推理。"""
        with self.frame_lock:
            self.latest_rgb_msg = rgb_msg
            self.latest_depth_msg = depth_msg

    def convert_depth_values_to_meters(
        self,
        values: np.ndarray,
        encoding: str,
    ) -> np.ndarray:
        """将深度数据统一转换为米。"""
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
        """筛选有效深度值。"""
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
        """先使用中心窗口，失败后使用检测框中央50%区域。"""
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
        """像素坐标反投影到相机光学坐标系：X右、Y下、Z前。"""
        if None in (self.fx, self.fy, self.cx, self.cy):
            raise RuntimeError("尚未收到有效CameraInfo")

        x_m = (float(u) - self.cx) * depth_m / self.fx
        y_m = (float(v) - self.cy) * depth_m / self.fy
        return x_m, y_m, depth_m

    @staticmethod
    def compute_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """计算两个 xyxy 框的 IoU。"""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def parse_model1_result(self, result) -> List[CandidateDetection]:
        """解析模型1，只保留非专用类别。"""
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
            global_id = int(class_value)
            if global_id not in MODEL1_ALLOWED_GLOBAL_IDS:
                continue

            class_name = GLOBAL_CLASS_ID_TO_NAME[global_id]
            candidates.append(
                CandidateDetection(
                    source_model="M1",
                    local_class_id=global_id,
                    global_class_id=global_id,
                    class_name=class_name,
                    confidence=float(confidence_value),
                    xyxy=(
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                    ),
                )
            )

        return candidates
    
    def parse_model2_result(self, result) -> List[CandidateDetection]:
        """解析模型2，恢复到12类全局编号并应用类别独立阈值。"""
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
            if local_id not in MODEL2_LOCAL_TO_GLOBAL:
                rospy.logwarn_throttle(
                    5.0,
                    "Model2 returned unknown local class id=%d",
                    local_id,
                )
                continue

            global_id = MODEL2_LOCAL_TO_GLOBAL[local_id]
            class_name = GLOBAL_CLASS_ID_TO_NAME[global_id]
            confidence = float(confidence_value)
            class_threshold = self.model2_class_conf_thresholds.get(
                class_name,
                self.model2_conf_threshold,
            )

            if confidence < class_threshold:
                continue

            candidates.append(
                CandidateDetection(
                    source_model="M2",
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

    def parse_model1_result_v2(self, result) -> List[CandidateDetection]:
        """解析模型1，恢复到12类全局编号并应用类别独立阈值。"""
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
            if local_id not in MODEL1_LOCAL_TO_GLOBAL:
                rospy.logwarn_throttle(
                    5.0,
                    "Model1 returned unknown local class id=%d",
                    local_id,
                )
                continue

            global_id = MODEL1_LOCAL_TO_GLOBAL[local_id]
            class_name = GLOBAL_CLASS_ID_TO_NAME[global_id]
            confidence = float(confidence_value)
            class_threshold = self.model1_class_conf_thresholds.get(
                class_name,
                self.model1_conf_threshold,
            )

            if confidence < class_threshold:
                continue

            candidates.append(
                CandidateDetection(
                    source_model="M1",
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

    def merge_candidates(
        self,
        model1_candidates: List[CandidateDetection],
        model2_candidates: List[CandidateDetection],
    ) -> Tuple[List[CandidateDetection], int]:
        """
        模型2是六个少数类的权威模型。

        规则：
        1. 模型1的六个专用类别从源头就不参与发布；
        2. 模型2结果达到阈值后直接保留；
        3. 如果模型2框与模型1允许类别框高度重叠，删除模型1框，避免
           同一物体同时显示为 vehicle 和 tire/sandbag 等。
        """
        kept_model1: List[CandidateDetection] = []
        suppressed_count = 0

        for general_candidate in model1_candidates:
            may_override = (
                not self.override_model1_global_ids
                or general_candidate.global_class_id
                in self.override_model1_global_ids
            )

            if may_override:
                overridden = any(
                    self.compute_iou(
                        general_candidate.xyxy,
                        specialist_candidate.xyxy,
                    ) >= self.specialist_override_iou
                    for specialist_candidate in model2_candidates
                )
                if overridden:
                    suppressed_count += 1
                    continue

            kept_model1.append(general_candidate)

        merged = kept_model1 + model2_candidates
        merged.sort(
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )
        return merged, suppressed_count

    def select_target(
        self,
        objects,
    ) -> Optional[DetectedObject]:
        """从最终融合结果中选择控制目标。"""
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
        """根据全局类别ID生成稳定的BGR颜色。"""
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
        """在最终融合图像上绘制检测框、类别、来源和深度。"""
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

    def inference_timer_callback(self, _event) -> None:
        """对同一帧顺序执行两个模型，融合后统一发布。"""
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
                conf=self.model1_conf_threshold,
                iou=self.model1_iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                # classes=sorted(MODEL1_ALLOWED_GLOBAL_IDS),
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
                max_det=self.max_det,
                verbose=False,
            )
            model2_ms = (
                time.perf_counter() - model2_start
            ) * 1000.0

        except Exception as exc:
            rospy.logerr("Dual YOLO inference failed: %r", exc)
            return

        model1_candidates = self.parse_model1_result_v2(
            model1_results[0]
        )
        model2_candidates = self.parse_model2_result(
            model2_results[0]
        )
        merged_candidates, suppressed_count = self.merge_candidates(
            model1_candidates,
            model2_candidates,
        )

        output_msg = DetectedObjectArray()
        output_msg.header = rgb_msg.header

        annotated_image = np.ascontiguousarray(
            rgb_image.copy(),
            dtype=np.uint8,
        )
        image_h, image_w = rgb_image.shape[:2]

        for candidate in merged_candidates:
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
            rospy.loginfo(
                "M1=%d M2=%d suppressed=%d final=%d | "
                "M1=%.1fms M2=%.1fms total=%.1fms depth=%s",
                len(model1_candidates),
                len(model2_candidates),
                suppressed_count,
                len(output_msg.objects),
                model1_ms,
                model2_ms,
                total_ms,
                depth_msg.encoding,
            )


def main() -> None:
    rospy.init_node(
        "r300_dual_yolo_depth_node",
        anonymous=False,
    )
    DualYoloDepthNode()
    rospy.loginfo("R300 dual YOLO depth node started")
    rospy.spin()


if __name__ == "__main__":
    main()
