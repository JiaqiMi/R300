#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Jetson/aarch64环境必须优先导入torch，避免libgomp TLS加载问题
import torch
from ultralytics import YOLO

import threading
import time
from typing import Optional, Set, Tuple

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


class YoloDepthNode:
    """
    D435i + YOLO26目标检测和三维定位节点。

    输入：
        RGB图像
        对齐到RGB的深度图像
        RGB相机内参

    输出：
        /r300_vision/detections
        /r300_vision/annotated_image
        /r300_vision/target_point
    """

    def __init__(self) -> None:
        # ============================================================
        # 1. 参数
        # ============================================================
        self.model_path = str(
            rospy.get_param("~model_path", "")
        )

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

        self.conf_threshold = float(
            # rospy.get_param("~conf_threshold", 0.25)
            rospy.get_param("~conf_threshold", 0.1)
        )

        self.iou_threshold = float(
            # rospy.get_param("~iou_threshold", 0.70)
            rospy.get_param("~iou_threshold", 0.30)
        )

        self.imgsz = int(
            rospy.get_param("~imgsz", 640)
        )

        self.device = str(
            rospy.get_param("~device", "0")
        )

        self.use_half = bool(
            rospy.get_param("~half", False)
        )

        self.infer_hz = float(
            rospy.get_param("~infer_hz", 10.0)
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
            str(name)
            for name in target_classes
        }

        self.target_policy = str(
            rospy.get_param(
                "~target_policy",
                "nearest",
            )
        )

        if not self.model_path:
            raise RuntimeError(
                "参数 ~model_path 不能为空"
            )

        if self.infer_hz <= 0.0:
            raise ValueError(
                "参数 ~infer_hz 必须大于0"
            )

        if self.device != "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "配置要求GPU推理，但"
                    "torch.cuda.is_available()为False"
                )

        # ============================================================
        # 2. 加载YOLO模型
        # ============================================================
        rospy.loginfo(
            "Loading YOLO model: %s",
            self.model_path,
        )

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

        self.model = YOLO(
            self.model_path
        )

        rospy.loginfo(
            "Model classes: %s",
            str(self.model.names),
        )

        # ============================================================
        # 3. CameraInfo内参
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
        # 4. ROS发布接口
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
        # 5. RGB与深度同步
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

        self.sync = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self.rgb_sub,
                    self.depth_sub,
                ],
                queue_size=self.sync_queue_size,
                slop=self.sync_slop,
                allow_headerless=False,
            )
        )

        self.sync.registerCallback(
            self.synced_callback
        )

        # ============================================================
        # 6. 只保留最新同步帧
        # ============================================================
        self.frame_lock = threading.Lock()

        self.latest_rgb_msg: Optional[Image] = None
        self.latest_depth_msg: Optional[Image] = None

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.infer_hz
            ),
            self.inference_timer_callback,
        )

        self.frame_counter = 0

        rospy.loginfo(
            "RGB topic: %s",
            self.rgb_topic,
        )

        rospy.loginfo(
            "Aligned depth topic: %s",
            self.depth_topic,
        )

        rospy.loginfo(
            "CameraInfo topic: %s",
            self.camera_info_topic,
        )

    # fix cv_bridge error: CvBridgeError: OpenCV(4.5.5) 
    @staticmethod
    def bgr_numpy_to_ros_image(
        image: np.ndarray,
        header,
    ) -> Image:
        """
        将 uint8 BGR NumPy图像手动转换为 sensor_msgs/Image。

        这样可以绕开 pip OpenCV 与 ROS cv_bridge
        之间的类型编号兼容问题。
        """
        if image is None:
            raise ValueError(
                "输出图像为空"
            )

        if image.ndim != 3:
            raise ValueError(
                "BGR图像必须是三维数组，"
                f"当前shape={image.shape}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                "BGR图像必须具有3个通道，"
                f"当前shape={image.shape}"
            )

        # 保证类型、内存连续性正确
        image = np.ascontiguousarray(
            image,
            dtype=np.uint8,
        )

        height, width = image.shape[:2]

        msg = Image()

        msg.header = header
        msg.height = height
        msg.width = width

        msg.encoding = "bgr8"
        msg.is_bigendian = 0

        # 每行字节数 = 宽度 × 3通道 × uint8单字节
        msg.step = width * 3

        msg.data = image.tobytes()

        return msg

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ) -> None:
        """读取RGB相机内参矩阵K。"""
        if (
            msg.K[0] <= 0.0
            or msg.K[4] <= 0.0
        ):
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
        """
        保存最新RGB和对齐深度消息。
        不在订阅回调里执行耗时推理。
        """
        with self.frame_lock:
            self.latest_rgb_msg = rgb_msg
            self.latest_depth_msg = depth_msg

    def convert_depth_values_to_meters(
        self,
        values: np.ndarray,
        encoding: str,
    ) -> np.ndarray:
        """将深度数据统一转换为米。"""
        values = values.astype(
            np.float32,
            copy=False,
        )

        if encoding in (
            "16UC1",
            "mono16",
        ):
            values = (
                values
                * self.depth_scale_16u
            )

        elif encoding == "32FC1":
            pass

        elif np.issubdtype(
            values.dtype,
            np.integer,
        ):
            values = (
                values
                * self.depth_scale_16u
            )

        return values

    def filter_depth_values(
        self,
        roi: np.ndarray,
        encoding: str,
    ) -> np.ndarray:
        """筛选有效深度值。"""
        if roi.size == 0:
            return np.empty(
                (0,),
                dtype=np.float32,
            )

        values = roi.reshape(-1)

        values = (
            self.convert_depth_values_to_meters(
                values,
                encoding,
            )
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
        """
        先使用目标中心小窗口估计深度。
        若中心存在空洞，再使用目标框中心区域。
        """
        image_h, image_w = (
            depth_image.shape[:2]
        )

        radius = (
            self.depth_window_radius
        )

        u0 = max(
            0,
            center_u - radius,
        )
        u1 = min(
            image_w,
            center_u + radius + 1,
        )

        v0 = max(
            0,
            center_v - radius,
        )
        v1 = min(
            image_h,
            center_v + radius + 1,
        )

        center_roi = depth_image[
            v0:v1,
            u0:u1,
        ]

        center_values = (
            self.filter_depth_values(
                center_roi,
                encoding,
            )
        )

        if (
            center_values.size
            >= self.min_valid_depth_pixels
        ):
            return float(
                np.median(center_values)
            )

        # 中心深度无效时，使用检测框内中央50%区域
        x_min, y_min, x_max, y_max = bbox

        bbox_w = max(
            1,
            x_max - x_min,
        )
        bbox_h = max(
            1,
            y_max - y_min,
        )

        inner_x0 = int(
            x_min + 0.25 * bbox_w
        )
        inner_x1 = int(
            x_max - 0.25 * bbox_w
        )

        inner_y0 = int(
            y_min + 0.25 * bbox_h
        )
        inner_y1 = int(
            y_max - 0.25 * bbox_h
        )

        inner_x0 = max(
            0,
            min(image_w - 1, inner_x0),
        )
        inner_x1 = max(
            inner_x0 + 1,
            min(image_w, inner_x1),
        )

        inner_y0 = max(
            0,
            min(image_h - 1, inner_y0),
        )
        inner_y1 = max(
            inner_y0 + 1,
            min(image_h, inner_y1),
        )

        inner_roi = depth_image[
            inner_y0:inner_y1,
            inner_x0:inner_x1,
        ]

        inner_values = (
            self.filter_depth_values(
                inner_roi,
                encoding,
            )
        )

        if (
            inner_values.size
            < self.min_valid_depth_pixels
        ):
            return None

        return float(
            np.median(inner_values)
        )

    def deproject_pixel(
        self,
        u: int,
        v: int,
        depth_m: float,
    ) -> Tuple[float, float, float]:
        """
        通过针孔模型把像素坐标和深度反投影到相机光学坐标系。

        相机光学坐标：
            X 向右
            Y 向下
            Z 向前
        """
        if None in (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        ):
            raise RuntimeError(
                "尚未收到有效CameraInfo"
            )

        x_m = (
            (float(u) - self.cx)
            * depth_m
            / self.fx
        )

        y_m = (
            (float(v) - self.cy)
            * depth_m
            / self.fy
        )

        z_m = depth_m

        return x_m, y_m, z_m

    def select_target(
        self,
        objects,
    ) -> Optional[DetectedObject]:
        """从有效检测结果中选择控制目标。"""
        candidates = []

        for obj in objects:
            if not obj.depth_valid:
                continue

            if (
                self.target_classes
                and obj.class_name
                not in self.target_classes
            ):
                continue

            candidates.append(obj)

        if not candidates:
            return None

        if self.target_policy == (
            "highest_confidence"
        ):
            return max(
                candidates,
                key=lambda item: (
                    item.confidence
                ),
            )

        # 默认选择距离最近的目标
        return min(
            candidates,
            key=lambda item: item.depth_m,
        )

    def inference_timer_callback(
        self,
        _event,
    ) -> None:
        """执行一次最新帧推理。"""
        with self.frame_lock:
            if (
                self.latest_rgb_msg is None
                or self.latest_depth_msg
                is None
            ):
                return

            rgb_msg = (
                self.latest_rgb_msg
            )
            depth_msg = (
                self.latest_depth_msg
            )

            self.latest_rgb_msg = None
            self.latest_depth_msg = None

        try:
            rgb_image = (
                self.bridge.imgmsg_to_cv2(
                    rgb_msg,
                    desired_encoding="bgr8",
                )
            )

            depth_image = (
                self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding=(
                        "passthrough"
                    ),
                )
            )

        except CvBridgeError as exc:
            rospy.logerr(
                "CvBridge conversion failed: %s",
                str(exc),
            )
            return

        if depth_image.ndim != 2:
            rospy.logerr_throttle(
                5.0,
                "深度图不是单通道，"
                "encoding=%s shape=%s",
                depth_msg.encoding,
                str(depth_image.shape),
            )
            return

        start_time = (
            time.perf_counter()
        )

        try:
            # results = self.model.predict(
            #     source=rgb_image,
            #     conf=self.conf_threshold,
            #     iou=self.iou_threshold,
            #     imgsz=self.imgsz,
            #     device=self.device,
            #     half=self.use_half,
            #     verbose=False,
            # )
            results = self.model.predict(
                source=rgb_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

        except Exception as exc:
            rospy.logerr(
                "YOLO inference failed: %r",
                exc,
            )
            return

        result = results[0]

        output_msg = (
            DetectedObjectArray()
        )

        output_msg.header = (
            rgb_msg.header
        )

        annotated_image = (
            result.plot()
        )

        annotated_image = np.ascontiguousarray(
            annotated_image,
            dtype=np.uint8,
        )

        image_h, image_w = (
            rgb_image.shape[:2]
        )

        boxes = result.boxes

        if boxes is not None:
            xyxy_array = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            class_array = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
            )

            confidence_array = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for (
                xyxy,
                class_value,
                confidence_value,
            ) in zip(
                xyxy_array,
                class_array,
                confidence_array,
            ):
                x_min = int(
                    max(
                        0,
                        min(
                            image_w - 1,
                            round(xyxy[0]),
                        ),
                    )
                )

                y_min = int(
                    max(
                        0,
                        min(
                            image_h - 1,
                            round(xyxy[1]),
                        ),
                    )
                )

                x_max = int(
                    max(
                        0,
                        min(
                            image_w - 1,
                            round(xyxy[2]),
                        ),
                    )
                )

                y_max = int(
                    max(
                        0,
                        min(
                            image_h - 1,
                            round(xyxy[3]),
                        ),
                    )
                )

                if (
                    x_max <= x_min
                    or y_max <= y_min
                ):
                    continue

                center_u = int(
                    round(
                        (x_min + x_max)
                        / 2.0
                    )
                )

                center_v = int(
                    round(
                        (y_min + y_max)
                        / 2.0
                    )
                )

                class_id = int(
                    class_value
                )

                confidence = float(
                    confidence_value
                )

                class_name = str(
                    self.model.names[
                        class_id
                    ]
                )

                detection = (
                    DetectedObject()
                )

                detection.header = (
                    rgb_msg.header
                )

                detection.class_id = (
                    class_id
                )

                detection.class_name = (
                    class_name
                )

                detection.confidence = (
                    confidence
                )

                detection.x_min = x_min
                detection.y_min = y_min
                detection.x_max = x_max
                detection.y_max = y_max

                detection.center_u = (
                    center_u
                )

                detection.center_v = (
                    center_v
                )

                depth_m = (
                    self.estimate_depth(
                        depth_image,
                        depth_msg.encoding,
                        center_u,
                        center_v,
                        (
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                        ),
                    )
                )

                if (
                    depth_m is not None
                    and self.fx is not None
                ):
                    try:
                        (
                            x_m,
                            y_m,
                            z_m,
                        ) = self.deproject_pixel(
                            center_u,
                            center_v,
                            depth_m,
                        )

                        detection.depth_valid = (
                            True
                        )

                        detection.depth_m = (
                            depth_m
                        )

                        detection.position.x = (
                            x_m
                        )

                        detection.position.y = (
                            y_m
                        )

                        detection.position.z = (
                            z_m
                        )

                        depth_text = (
                            "{} {:.2f}m "
                            "X{:.2f} Y{:.2f}"
                        ).format(
                            class_name,
                            depth_m,
                            x_m,
                            y_m,
                        )

                        cv2.putText(
                            annotated_image,
                            depth_text,
                            (
                                x_min,
                                min(
                                    image_h - 10,
                                    y_max + 20,
                                ),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

                    except Exception as exc:
                        rospy.logwarn_throttle(
                            2.0,
                            "3D反投影失败: %s",
                            str(exc),
                        )

                        detection.depth_valid = (
                            False
                        )

                else:
                    detection.depth_valid = (
                        False
                    )

                    detection.depth_m = (
                        float("nan")
                    )

                    detection.position.x = (
                        float("nan")
                    )

                    detection.position.y = (
                        float("nan")
                    )

                    detection.position.z = (
                        float("nan")
                    )

                output_msg.objects.append(
                    detection
                )

        self.detections_pub.publish(
            output_msg
        )

        # 发布被选中的单个控制目标位置
        selected_target = (
            self.select_target(
                output_msg.objects
            )
        )

        if selected_target is not None:
            target_point_msg = (
                PointStamped()
            )

            target_point_msg.header = (
                selected_target.header
            )

            target_point_msg.point = (
                selected_target.position
            )

            self.target_point_pub.publish(
                target_point_msg
            )

        # try:
        #     annotated_msg = (
        #         self.bridge.cv2_to_imgmsg(
        #             annotated_image,
        #             encoding="bgr8",
        #         )
        #     )

        #     annotated_msg.header = (
        #         rgb_msg.header
        #     )

        #     self.annotated_pub.publish(
        #         annotated_msg
        #     )

        # except CvBridgeError as exc:
        #     rospy.logerr(
        #         "Annotated image publish failed: %s",
        #         str(exc),
        #     )
        try:
            annotated_msg = (
                self.bgr_numpy_to_ros_image(
                    annotated_image,
                    rgb_msg.header,
                )
            )

            self.annotated_pub.publish(
                annotated_msg
            )

        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Annotated image publish failed: %r",
                exc,
            )

        elapsed_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000.0
        )

        self.frame_counter += 1

        if self.frame_counter % 30 == 0:
            rospy.loginfo(
                "detections=%d, "
                "inference=%.1f ms, "
                "depth_encoding=%s",
                len(output_msg.objects),
                elapsed_ms,
                depth_msg.encoding,
            )


def main() -> None:
    rospy.init_node(
        "r300_yolo_depth_node",
        anonymous=False,
    )

    YoloDepthNode()

    rospy.loginfo(
        "R300 YOLO depth node started"
    )

    rospy.spin()


if __name__ == "__main__":
    main()