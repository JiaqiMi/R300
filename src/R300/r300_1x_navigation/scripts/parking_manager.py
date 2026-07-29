#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""科目一保守型自主泊车状态机。

航点完成后，在视觉、类别能力、TF、move_base 和泊车低速配置全部就绪时，
依次搜索 P 牌、靠近停车区域、连续确认空车位、规划并低速驶入，最后验证
实际位置、朝向和速度。任何不确定状态均停止运动并进入 ERROR。
"""

import copy
import json
import math
import threading
import time

import actionlib
import dynamic_reconfigure.client
import rospy
import tf.transformations as tft
import tf2_geometry_msgs  # noqa: F401  注册 PoseStamped 的 tf2 转换
import tf2_ros

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan, GetPlanRequest
from r300_vision_msgs.msg import DetectedObjectArray
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse


class ParkingManager:
    WAIT_MISSION = "WAIT_MISSION"
    SEARCH_PARK = "SEARCH_PARK"
    APPROACH_PARK = "APPROACH_PARK"
    SEARCH_SLOT = "SEARCH_SLOT"
    ENTER_SLOT = "ENTER_SLOT"
    VERIFY_PARKING = "VERIFY_PARKING"
    FINISH = "FINISH"
    ERROR = "ERROR"

    def __init__(self):
        self.lock = threading.RLock()
        self.dwa_lock = threading.Lock()

        # ROS 接口
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.detections_topic = str(rospy.get_param(
            "~detections_topic", "/r300_vision/detections"))
        self.available_classes_topic = str(rospy.get_param(
            "~available_classes_topic", "/r300_vision/available_classes"))
        self.waypoint_status_topic = str(rospy.get_param(
            "~waypoint_status_topic", "/subject1/waypoint_status"))
        self.odom_topic = str(rospy.get_param(
            "~odom_topic", "/subject1/dwa_odom"))
        self.cmd_vel_topic = str(rospy.get_param(
            "~cmd_vel_topic", "/subject1/cmd_vel_raw"))
        self.move_base_action = str(rospy.get_param(
            "~move_base_action", "/move_base"))
        self.make_plan_service = str(rospy.get_param(
            "~make_plan_service", "/move_base/make_plan"))
        self.dwa_reconfigure_namespace = str(rospy.get_param(
            "~dwa_reconfigure_namespace", "/move_base/DWAPlannerROS"))
        self.map_frame = str(rospy.get_param("~map_frame", "map"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))

        self.auto_start_on_waypoint_completed = bool(rospy.get_param(
            "~auto_start_on_waypoint_completed", True))
        self.single_use = bool(rospy.get_param("~single_use", True))

        # 类别与视觉健康
        self.park_classes = self._class_set(rospy.get_param(
            "~park_classes", ["park"]))
        self.slot_classes = self._class_set(rospy.get_param(
            "~slot_classes", ["parking_slot"]))
        self.vehicle_classes = self._class_set(rospy.get_param(
            "~vehicle_classes", ["vehicle"]))
        self.required_classes = self._class_set(rospy.get_param(
            "~required_classes", ["park", "parking_slot", "vehicle"]))
        self.vision_ready_timeout_s = max(0.1, float(rospy.get_param(
            "~vision_ready_timeout_s", 8.0)))
        self.max_detection_age_s = max(0.05, float(rospy.get_param(
            "~max_detection_age_s", 1.0)))
        self.odom_max_age_s = max(0.05, float(rospy.get_param(
            "~odom_max_age_s", 0.5)))

        self.park_min_confidence = float(rospy.get_param(
            "~park_min_confidence", 0.65))
        self.slot_min_confidence = float(rospy.get_param(
            "~slot_min_confidence", 0.55))
        self.vehicle_min_confidence = float(rospy.get_param(
            "~vehicle_min_confidence", 0.45))
        self.confirm_frames = max(1, int(rospy.get_param(
            "~confirm_frames", 3)))
        self.confirm_max_gap_s = max(0.05, float(rospy.get_param(
            "~confirm_max_gap_s", 0.8)))
        self.candidate_match_px = max(5, int(rospy.get_param(
            "~candidate_match_px", 90)))
        self.min_bbox_width_px = max(1, int(rospy.get_param(
            "~min_bbox_width_px", 20)))
        self.min_bbox_height_px = max(1, int(rospy.get_param(
            "~min_bbox_height_px", 20)))
        self.min_target_distance_m = max(0.1, float(rospy.get_param(
            "~min_target_distance_m", 0.4)))
        self.max_target_distance_m = max(
            self.min_target_distance_m,
            float(rospy.get_param("~max_target_distance_m", 15.0)),
        )

        # 搜索动作
        self.search_angular_speed_radps = abs(float(rospy.get_param(
            "~search_angular_speed_radps", 0.28)))
        self.search_direction = 1.0 if float(rospy.get_param(
            "~search_direction", 1.0)) >= 0.0 else -1.0
        self.search_rotate_s = max(0.1, float(rospy.get_param(
            "~search_rotate_s", 0.9)))
        self.search_pause_s = max(0.1, float(rospy.get_param(
            "~search_pause_s", 1.0)))
        self.search_timeout_s = max(5.0, float(rospy.get_param(
            "~search_timeout_s", 90.0)))

        # 目标执行与规划
        self.park_sign_standoff_m = max(0.0, float(rospy.get_param(
            "~park_sign_standoff_m", 1.0)))
        self.min_approach_travel_m = max(0.0, float(rospy.get_param(
            "~min_approach_travel_m", 0.6)))
        self.slot_goal_extension_m = max(0.0, float(rospy.get_param(
            "~slot_goal_extension_m", 0.8)))
        self.goal_timeout_s = max(2.0, float(rospy.get_param(
            "~goal_timeout_s", 35.0)))
        self.action_server_timeout_s = max(0.5, float(rospy.get_param(
            "~action_server_timeout_s", 8.0)))
        self.tf_timeout_s = max(0.1, float(rospy.get_param(
            "~tf_timeout_s", 1.0)))
        self.make_plan_tolerance_m = max(0.0, float(rospy.get_param(
            "~make_plan_tolerance_m", 0.2)))
        self.make_plan_timeout_s = max(0.1, float(rospy.get_param(
            "~make_plan_timeout_s", 2.0)))

        # 连续空车位判断
        self.slot_confirm_frames = max(1, int(rospy.get_param(
            "~slot_confirm_frames", 3)))
        self.empty_confirm_frames = max(1, int(rospy.get_param(
            "~empty_confirm_frames", 3)))
        self.slot_match_center_px = max(5.0, float(rospy.get_param(
            "~slot_match_center_px", 80.0)))
        self.slot_match_depth_m = max(0.05, float(rospy.get_param(
            "~slot_match_depth_m", 1.0)))
        self.vehicle_center_margin_ratio = min(0.45, max(0.0, float(
            rospy.get_param("~vehicle_center_margin_ratio", 0.08))))
        self.vehicle_slot_overlap_threshold = min(1.0, max(0.0, float(
            rospy.get_param("~vehicle_slot_overlap_threshold", 0.15))))
        self.occupied_distance_m = max(0.1, float(rospy.get_param(
            "~occupied_distance_m", 1.8)))

        # 泊车专用 DWA dynamic_reconfigure 参数
        self.dwa_reconfigure_timeout_s = max(0.5, float(rospy.get_param(
            "~dwa_reconfigure_timeout_s", 3.0)))
        self.parking_dwa_config = {
            "max_vel_x": float(rospy.get_param("~parking_max_vel_x", 0.30)),
            "min_vel_x": float(rospy.get_param("~parking_min_vel_x", 0.0)),
            "max_vel_trans": float(rospy.get_param(
                "~parking_max_trans_vel", 0.30)),
            "min_vel_trans": float(rospy.get_param(
                "~parking_min_trans_vel", 0.0)),
            "max_vel_theta": float(rospy.get_param(
                "~parking_max_rot_vel", 0.35)),
            "min_vel_theta": float(rospy.get_param(
                "~parking_min_rot_vel", 0.05)),
            "xy_goal_tolerance": float(rospy.get_param(
                "~parking_xy_goal_tolerance_m", 0.40)),
            "yaw_goal_tolerance": float(rospy.get_param(
                "~parking_yaw_goal_tolerance_rad", 0.52)),
        }
        self._validate_parking_dwa_values()
        self.search_angular_speed_radps = min(
            self.search_angular_speed_radps,
            self.parking_dwa_config["max_vel_theta"],
        )

        # 最终停车验证
        self.final_position_tolerance_m = max(0.01, float(rospy.get_param(
            "~final_position_tolerance_m", 0.50)))
        self.final_yaw_tolerance_rad = max(0.01, float(rospy.get_param(
            "~final_yaw_tolerance_rad", 0.52)))
        self.final_speed_tolerance_mps = max(0.0, float(rospy.get_param(
            "~final_speed_tolerance_mps", 0.10)))
        self.final_stable_time_s = max(0.1, float(rospy.get_param(
            "~final_stable_time_s", 1.0)))
        self.final_verify_timeout_s = max(0.5, float(rospy.get_param(
            "~final_verify_timeout_s", 5.0)))

        # 相机光学坐标 -> base_link(FLU)。兼容旧参数名。
        default_rotation = [
            0.0, 0.0, 1.0,
            -1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
        ]
        default_translation = [0.30, 0.00, 0.10]
        rotation_values = rospy.get_param(
            "~camera_to_body_rotation",
            rospy.get_param("~camera_to_base_rotation", default_rotation),
        )
        translation_values = rospy.get_param(
            "~camera_in_body_translation_m",
            rospy.get_param(
                "~camera_in_base_translation_m", default_translation),
        )
        self.camera_to_base_rotation = [float(v) for v in rotation_values]
        self.camera_in_base_translation_m = [
            float(v) for v in translation_values]
        self._validate_extrinsics()

        # 共享运行状态
        self.state = self.WAIT_MISSION
        self.parking_active = False
        self.start_in_progress = False
        self.execution_id = 0
        self.used = False
        self.executing_goal = False
        self.move_base_active = False
        self.last_error = ""
        self.search_started_monotonic = 0.0
        self.search_cycle_started_monotonic = 0.0
        self.search_motion_was_active = False

        self.candidate_kind = None
        self.candidate_u = None
        self.candidate_depth_m = None
        self.candidate_count = 0
        self.last_candidate_monotonic = 0.0

        self.slot_candidate = None
        self.slot_seen_count = 0
        self.empty_streak = 0
        self.occupied_streak = 0
        self.last_slot_candidate_monotonic = 0.0
        self.last_slot_total = 0
        self.last_slot_occupied = 0
        self.last_slot_empty = 0

        self.selected_park = None
        self.selected_slot = None
        self.selected_goal_pose = None

        self.received_detections = False
        self.last_detection_received_monotonic = 0.0
        self.last_detection_stamp = rospy.Time(0)
        self.latest_detections = None
        self.available_classes_received = False
        self.available_classes = set()

        self.latest_odom = None
        self.last_odom_received_monotonic = 0.0
        self.current_speed_mps = float("nan")
        self.final_position_error_m = float("nan")
        self.final_yaw_error_rad = float("nan")

        self.dwa_client = None
        self.original_dwa_config = None
        self.low_speed_applied = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.move_base = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction)
        self.make_plan = rospy.ServiceProxy(self.make_plan_service, GetPlan)

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=2)
        self.state_pub = rospy.Publisher(
            "/subject1/parking/state", String, queue_size=5, latch=True)
        self.done_pub = rospy.Publisher(
            "/subject1/parking/done", Bool, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher(
            "/subject1/parking/selected_goal", PoseStamped,
            queue_size=2, latch=True)

        rospy.Subscriber(
            self.waypoint_status_topic, String,
            self.waypoint_status_cb, queue_size=5)
        rospy.Subscriber(
            self.detections_topic, DetectedObjectArray,
            self.detections_cb, queue_size=5)
        rospy.Subscriber(
            self.available_classes_topic, String,
            self.available_classes_cb, queue_size=1)
        rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_cb, queue_size=5)

        rospy.Service(
            "/subject1/parking/start", Trigger, self.start_service_cb)
        rospy.Service(
            "/subject1/parking/reset", Trigger, self.reset_service_cb)

        rospy.Timer(rospy.Duration(0.1), self.control_timer_cb)
        rospy.Timer(rospy.Duration(1.0), lambda _event: self.publish_state())
        rospy.on_shutdown(self.shutdown)

        self.done_pub.publish(Bool(data=False))
        self.publish_state()
        rospy.logwarn(
            "Parking manager ready: detections=%s classes=%s required=%s",
            self.detections_topic,
            self.available_classes_topic,
            sorted(self.required_classes),
        )

    # 基础工具
    @staticmethod
    def _class_set(values):
        if not isinstance(values, list):
            raise ValueError("类别参数必须是 YAML 列表")
        return {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }

    @staticmethod
    def _parse_status_state(text):
        for token in str(text).split():
            if token.startswith("state="):
                return token.split("=", 1)[1]
        return "UNKNOWN"

    @staticmethod
    def _bbox(obj):
        return int(obj.x_min), int(obj.y_min), int(obj.x_max), int(obj.y_max)

    @staticmethod
    def _bbox_area(box):
        x0, y0, x1, y1 = box
        return max(0, x1 - x0) * max(0, y1 - y0)

    @staticmethod
    def _intersection_area(box_a, box_b):
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
            0, min(ay1, by1) - max(ay0, by0))

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _pose_yaw(pose):
        q = pose.orientation
        return tft.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]

    def _validate_extrinsics(self):
        if len(self.camera_to_base_rotation) != 9:
            raise ValueError("camera_to_body_rotation 必须包含 9 个数")
        if len(self.camera_in_base_translation_m) != 3:
            raise ValueError("camera_in_body_translation_m 必须包含 3 个数")
        values = self.camera_to_base_rotation + self.camera_in_base_translation_m
        if not all(math.isfinite(value) for value in values):
            raise ValueError("相机外参必须全部为有限数值")
        rows = [
            self.camera_to_base_rotation[index:index + 3]
            for index in (0, 3, 6)
        ]
        for row in rows:
            norm = math.sqrt(sum(value * value for value in row))
            if abs(norm - 1.0) > 1.0e-3:
                raise ValueError("camera_to_body_rotation 行向量必须为单位向量")
        for first, second in ((0, 1), (0, 2), (1, 2)):
            dot = sum(rows[first][i] * rows[second][i] for i in range(3))
            if abs(dot) > 1.0e-3:
                raise ValueError("camera_to_body_rotation 必须为正交矩阵")

    def _validate_parking_dwa_values(self):
        if any(value < 0.0 for value in self.parking_dwa_config.values()):
            raise ValueError("泊车 DWA 速度和容差参数不能为负数")
        if (self.parking_dwa_config["min_vel_x"]
                > self.parking_dwa_config["max_vel_x"]):
            raise ValueError("parking_min_vel_x 不能大于 parking_max_vel_x")
        if (self.parking_dwa_config["min_vel_trans"]
                > self.parking_dwa_config["max_vel_trans"]):
            raise ValueError(
                "parking_min_trans_vel 不能大于 parking_max_trans_vel")
        if (self.parking_dwa_config["min_vel_theta"]
                > self.parking_dwa_config["max_vel_theta"]):
            raise ValueError(
                "parking_min_rot_vel 不能大于 parking_max_rot_vel")

    def detection_to_base_xyz(self, obj):
        if not bool(obj.depth_valid):
            return None
        camera = [
            float(obj.position.x),
            float(obj.position.y),
            float(obj.position.z),
        ]
        if not all(math.isfinite(value) for value in camera):
            return None
        r = self.camera_to_base_rotation
        t = self.camera_in_base_translation_m
        cx, cy, cz = camera
        return (
            r[0] * cx + r[1] * cy + r[2] * cz + t[0],
            r[3] * cx + r[4] * cy + r[5] * cz + t[1],
            r[6] * cx + r[7] * cy + r[8] * cz + t[2],
        )

    def valid_detection(self, obj, classes, min_confidence):
        name = str(obj.class_name).strip().lower()
        confidence = float(obj.confidence)
        if name not in classes or not math.isfinite(confidence):
            return False
        if confidence < min_confidence:
            return False
        if int(obj.x_max) - int(obj.x_min) < self.min_bbox_width_px:
            return False
        if int(obj.y_max) - int(obj.y_min) < self.min_bbox_height_px:
            return False
        base_xyz = self.detection_to_base_xyz(obj)
        if base_xyz is None:
            return False
        distance = math.hypot(base_xyz[0], base_xyz[1])
        return self.min_target_distance_m <= distance <= self.max_target_distance_m

    def _current_map_pose(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time(0)
        pose.header.frame_id = self.base_frame
        pose.pose.orientation.w = 1.0
        return self.tf_buffer.transform(
            pose, self.map_frame, rospy.Duration(self.tf_timeout_s))

    def local_target_to_map_pose(self, local_x, local_y, local_yaw):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time(0)
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = float(local_x)
        pose.pose.position.y = float(local_y)
        q = tft.quaternion_from_euler(0.0, 0.0, float(local_yaw))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return self.tf_buffer.transform(
            pose, self.map_frame, rospy.Duration(self.tf_timeout_s))

    def is_execution_valid(self, execution_id):
        with self.lock:
            return (
                execution_id == self.execution_id
                and self.parking_active
                and not rospy.is_shutdown()
            )

    # 视觉、里程计与启动检查
    def available_classes_cb(self, msg):
        try:
            decoded = json.loads(msg.data)
            if not isinstance(decoded, list):
                raise ValueError("available_classes JSON 必须是数组")
            classes = self._class_set(decoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            classes = {
                value.strip().lower()
                for value in str(msg.data).split(",")
                if value.strip()
            }
        with self.lock:
            self.available_classes = classes
            self.available_classes_received = True
            self.publish_state()

    def odom_cb(self, msg):
        linear = msg.twist.twist.linear
        speed = math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2)
        with self.lock:
            self.latest_odom = copy.deepcopy(msg)
            self.last_odom_received_monotonic = time.monotonic()
            self.current_speed_mps = speed

    def _detection_age_locked(self):
        if not self.received_detections:
            return float("inf")
        receipt_age = time.monotonic() - self.last_detection_received_monotonic
        if self.last_detection_stamp == rospy.Time(0):
            return receipt_age
        stamp_age = (rospy.Time.now() - self.last_detection_stamp).to_sec()
        if stamp_age < -0.2:
            return float("inf")
        return max(receipt_age, stamp_age)

    def _vision_preflight_locked(self):
        if not self.received_detections:
            return False, "尚未收到视觉 detections 消息"
        age = self._detection_age_locked()
        if not math.isfinite(age) or age > self.max_detection_age_s:
            return False, "视觉 detections 已过期 age=%.2fs" % age
        if not self.available_classes_received:
            return False, "尚未收到 available_classes"
        missing = sorted(self.required_classes - self.available_classes)
        if missing:
            return False, "视觉模型缺少必要类别: %s" % ",".join(missing)
        return True, "视觉与类别能力就绪"

    def _preflight(self):
        deadline = time.monotonic() + self.vision_ready_timeout_s
        while not rospy.is_shutdown():
            with self.lock:
                ok, message = self._vision_preflight_locked()
            if ok:
                break
            if time.monotonic() >= deadline:
                return False, "%s（等待 %.1fs）" % (
                    message, self.vision_ready_timeout_s)
            rospy.sleep(0.05)
        if rospy.is_shutdown():
            return False, "ROS 正在关闭"
        if not self.move_base.wait_for_server(
                rospy.Duration(self.action_server_timeout_s)):
            return False, "move_base action server 不可用"
        if not self.tf_buffer.can_transform(
            self.map_frame,
            self.base_frame,
            rospy.Time(0),
            rospy.Duration(self.tf_timeout_s),
        ):
            return False, "缺少 %s -> %s TF" % (
                self.map_frame, self.base_frame)
        try:
            self._current_map_pose()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException) as exc:
            return False, "TF 检查失败: %s" % exc
        return True, "启动检查通过"

    # DWA 泊车低速模式
    def _apply_parking_dwa(self):
        with self.dwa_lock:
            return self._apply_parking_dwa_unlocked()

    def _apply_parking_dwa_unlocked(self):
        original = None
        try:
            if self.dwa_client is None:
                self.dwa_client = dynamic_reconfigure.client.Client(
                    self.dwa_reconfigure_namespace,
                    timeout=self.dwa_reconfigure_timeout_s,
                )
            current = self.dwa_client.get_configuration(
                timeout=self.dwa_reconfigure_timeout_s)
            missing_keys = sorted(set(self.parking_dwa_config) - set(current))
            if missing_keys:
                return False, "DWA 动态参数缺失: %s" % ",".join(missing_keys)
            original = {
                key: current[key]
                for key in self.parking_dwa_config
            }
            updated = self.dwa_client.update_configuration(
                self.parking_dwa_config)
            for key, expected in self.parking_dwa_config.items():
                if key not in updated or abs(float(updated[key]) - expected) > 1.0e-6:
                    self.dwa_client.update_configuration(original)
                    return False, "DWA 泊车参数未生效: %s" % key
            with self.lock:
                self.original_dwa_config = original
                self.low_speed_applied = True
            return True, "泊车低速参数已生效"
        except Exception as exc:
            if self.dwa_client is not None and original:
                try:
                    self.dwa_client.update_configuration(original)
                except Exception as restore_exc:
                    rospy.logerr(
                        "DWA 低速切换异常后恢复也失败: %s", restore_exc)
            return False, "DWA dynamic_reconfigure 失败: %s" % exc

    def _restore_dwa(self):
        with self.dwa_lock:
            return self._restore_dwa_unlocked()

    def _restore_dwa_unlocked(self):
        with self.lock:
            if not self.low_speed_applied:
                return True, "无需恢复 DWA"
            original = dict(self.original_dwa_config or {})
        try:
            if self.dwa_client is None or not original:
                return False, "缺少 DWA 原始参数缓存"
            restored = self.dwa_client.update_configuration(original)
            for key, expected in original.items():
                if key not in restored or abs(float(restored[key]) - float(expected)) > 1.0e-6:
                    return False, "DWA 原始参数恢复失败: %s" % key
            with self.lock:
                self.low_speed_applied = False
                self.original_dwa_config = None
            return True, "DWA 原始参数已恢复"
        except Exception as exc:
            return False, "恢复 DWA 参数异常: %s" % exc

    # 启动、复位和错误处理
    def waypoint_status_cb(self, msg):
        if not self.enabled or not self.auto_start_on_waypoint_completed:
            return
        if self._parse_status_state(msg.data) != "COMPLETED":
            return
        with self.lock:
            if self.start_in_progress or self.parking_active:
                return
            if self.single_use and self.used:
                return
            if self.state != self.WAIT_MISSION:
                return
            self.start_in_progress = True
        threading.Thread(
            target=self._automatic_start_worker,
            daemon=True,
        ).start()

    def _automatic_start_worker(self):
        try:
            success, message = self.start_parking(
                cancel_existing_goal=False,
                start_flag_already_set=True,
            )
            if not success and message != "泊车启动已取消":
                self.set_error("自动泊车未启动: %s" % message)
        finally:
            with self.lock:
                self.start_in_progress = False

    def start_service_cb(self, _request):
        success, message = self.start_parking(cancel_existing_goal=True)
        return TriggerResponse(success, message)

    def start_parking(self, cancel_existing_goal, start_flag_already_set=False):
        with self.lock:
            if not self.enabled:
                return False, "自主泊车已禁用"
            if self.start_in_progress and not start_flag_already_set:
                return False, "泊车启动检查正在执行"
            if self.parking_active or self.executing_goal:
                return False, "泊车流程正在执行"
            if self.single_use and self.used:
                return False, "泊车流程已使用；请先 reset"
            start_generation = self.execution_id
            if not start_flag_already_set:
                self.start_in_progress = True

        low_speed_applied_here = False
        try:
            ok, message = self._preflight()
            if not ok:
                return False, message
            with self.lock:
                if self.execution_id != start_generation:
                    return False, "泊车启动已取消"
            ok, message = self._apply_parking_dwa()
            if not ok:
                return False, message
            low_speed_applied_here = True

            with self.lock:
                if self.execution_id != start_generation:
                    return False, "泊车启动已取消"

            if cancel_existing_goal:
                self.move_base.cancel_all_goals()
                self._publish_stop_once()

            with self.lock:
                if self.execution_id != start_generation:
                    return False, "泊车启动已取消"
                if self.parking_active:
                    return False, "泊车流程已被其他请求启动"
                self.execution_id += 1
                execution_id = self.execution_id
                self.parking_active = True
                self.executing_goal = False
                self.move_base_active = False
                self.last_error = ""
                self.selected_park = None
                self.selected_slot = None
                self.selected_goal_pose = None
                self.final_position_error_m = float("nan")
                self.final_yaw_error_rad = float("nan")
                self.done_pub.publish(Bool(data=False))
                self._enter_search_state_locked(self.SEARCH_PARK)
                self.publish_state()
            rospy.logwarn("自主泊车已启动 execution_id=%d", execution_id)
            return True, "自主泊车已启动 execution_id=%d" % execution_id
        finally:
            with self.lock:
                if not start_flag_already_set:
                    self.start_in_progress = False
                active = self.parking_active
            if low_speed_applied_here and not active:
                self._restore_dwa()

    def reset_service_cb(self, _request):
        with self.lock:
            self.execution_id += 1
            self.parking_active = False
            self.executing_goal = False
            self.move_base_active = False
            self.state = self.WAIT_MISSION
            self.used = False
            self.last_error = ""
            self.selected_park = None
            self.selected_slot = None
            self.selected_goal_pose = None
            self._clear_confirmation_locked()
            self._clear_slot_confirmation_locked()
            self.done_pub.publish(Bool(data=False))
            self._publish_stop_once()
            self.publish_state()
        self.move_base.cancel_all_goals()
        restored, restore_message = self._restore_dwa()
        if not restored:
            self.set_error(restore_message)
            return TriggerResponse(False, restore_message)
        return TriggerResponse(True, "停车状态已复位，旧执行线程已失效")

    def set_error(self, message, execution_id=None):
        with self.lock:
            if execution_id is not None and execution_id != self.execution_id:
                return False
            self.execution_id += 1
            self.parking_active = False
            self.executing_goal = False
            self.move_base_active = False
            self.state = self.ERROR
            self.last_error = str(message)
            self.used = True
            self._publish_stop_once()
            self.publish_state()
        self.move_base.cancel_all_goals()
        restored, restore_message = self._restore_dwa()
        if not restored:
            rospy.logerr("%s；%s", message, restore_message)
        else:
            rospy.logerr("停车流程失败：%s", message)
        return True

    # 搜索控制
    def _enter_search_state_locked(self, state, reset_timeout=True):
        self.state = state
        now = time.monotonic()
        if reset_timeout or self.search_started_monotonic <= 0.0:
            self.search_started_monotonic = now
        self.search_cycle_started_monotonic = now
        self.search_motion_was_active = False
        self._clear_confirmation_locked()
        self._clear_slot_confirmation_locked()
        rospy.logwarn("停车状态切换：%s", state)

    def _clear_confirmation_locked(self):
        self.candidate_kind = None
        self.candidate_u = None
        self.candidate_depth_m = None
        self.candidate_count = 0
        self.last_candidate_monotonic = 0.0

    def _clear_slot_confirmation_locked(self):
        self.slot_candidate = None
        self.slot_seen_count = 0
        self.empty_streak = 0
        self.occupied_streak = 0
        self.last_slot_candidate_monotonic = 0.0

    def _search_is_paused(self):
        period = self.search_rotate_s + self.search_pause_s
        elapsed = time.monotonic() - self.search_cycle_started_monotonic
        return elapsed % period >= self.search_rotate_s

    def _publish_search_rotation(self):
        cmd = Twist()
        cmd.angular.z = self.search_direction * self.search_angular_speed_radps
        self.cmd_pub.publish(cmd)
        self.search_motion_was_active = True

    def _publish_stop_once(self):
        self.cmd_pub.publish(Twist())
        self.search_motion_was_active = False

    def control_timer_cb(self, _event):
        failure = None
        execution_id = None
        with self.lock:
            if not self.parking_active:
                return
            execution_id = self.execution_id
            vision_ok, vision_message = self._vision_preflight_locked()
            if not vision_ok:
                failure = "泊车期间视觉失效: %s" % vision_message
            elif self.state not in (self.SEARCH_PARK, self.SEARCH_SLOT):
                return
            elif self.executing_goal or self.move_base_active:
                return
            elif time.monotonic() - self.search_started_monotonic > self.search_timeout_s:
                target = "P 牌" if self.state == self.SEARCH_PARK else "空车位"
                failure = "搜索%s超时 %.1f 秒" % (target, self.search_timeout_s)
            elif self._search_is_paused():
                if self.search_motion_was_active:
                    self._publish_stop_once()
            else:
                if not self.search_motion_was_active:
                    # 旋转期间图像不参与连续确认；新一轮停顿必须重新
                    # 积累完整的连续帧，避免跨旋转阶段拼接空闲证据。
                    self._clear_confirmation_locked()
                    self._clear_slot_confirmation_locked()
                self._publish_search_rotation()
        if failure is not None:
            self.set_error(failure, execution_id)

    # 视觉候选与空车位连续确认
    def detections_cb(self, msg):
        now = time.monotonic()
        with self.lock:
            self.received_detections = True
            self.last_detection_received_monotonic = now
            self.last_detection_stamp = msg.header.stamp
            self.latest_detections = copy.deepcopy(msg)
            if not self.parking_active or self.executing_goal:
                return
            if self.state not in (self.SEARCH_PARK, self.SEARCH_SLOT):
                return
            if not self._search_is_paused():
                return
            if self._detection_age_locked() > self.max_detection_age_s:
                return

            if self.state == self.SEARCH_PARK:
                candidate = self._best_park_candidate(msg)
                if candidate is None:
                    self._maybe_clear_stale_confirmation_locked()
                    return
                if not self._confirm_park_candidate_locked(candidate):
                    return
                execution_id = self.execution_id
                self.selected_park = copy.deepcopy(candidate)
                self._publish_stop_once()
                self.state = self.APPROACH_PARK
                self.executing_goal = True
                self.publish_state()
                threading.Thread(
                    target=self.execute_park_approach,
                    args=(self.selected_park, execution_id),
                    daemon=True,
                ).start()
                return

            confirmed_slot = self._update_slot_confirmation_locked(msg)
            if confirmed_slot is None:
                return
            execution_id = self.execution_id
            self.selected_slot = copy.deepcopy(confirmed_slot)
            self._publish_stop_once()
            self.state = self.ENTER_SLOT
            self.executing_goal = True
            self.publish_state()
            threading.Thread(
                target=self.execute_slot_entry,
                args=(self.selected_slot, execution_id),
                daemon=True,
            ).start()

    def _best_park_candidate(self, msg):
        candidates = [
            obj for obj in msg.objects
            if self.valid_detection(
                obj, self.park_classes, self.park_min_confidence)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda obj: (
            float(obj.confidence), -float(obj.depth_m)))

    def _confirm_park_candidate_locked(self, obj):
        now = time.monotonic()
        u = int(obj.center_u)
        depth = float(obj.depth_m)
        gap = (float("inf") if self.last_candidate_monotonic <= 0.0
               else now - self.last_candidate_monotonic)
        same = (
            self.candidate_kind == "PARK"
            and self.candidate_u is not None
            and abs(u - self.candidate_u) <= self.candidate_match_px
            and self.candidate_depth_m is not None
            and abs(depth - self.candidate_depth_m) <= 2.0
            and gap <= self.confirm_max_gap_s
        )
        self.candidate_count = self.candidate_count + 1 if same else 1
        self.candidate_kind = "PARK"
        self.candidate_u = u
        self.candidate_depth_m = depth
        self.last_candidate_monotonic = now
        self.publish_state()
        return self.candidate_count >= self.confirm_frames

    def _maybe_clear_stale_confirmation_locked(self):
        if (self.last_candidate_monotonic > 0.0 and
                time.monotonic() - self.last_candidate_monotonic
                > self.confirm_max_gap_s):
            self._clear_confirmation_locked()
            self.publish_state()

    def _slot_matches(self, first, second):
        center_distance = math.hypot(
            float(first.center_u) - float(second.center_u),
            float(first.center_v) - float(second.center_v),
        )
        if center_distance > self.slot_match_center_px:
            return False
        first_depth = float(first.depth_m)
        second_depth = float(second.depth_m)
        return (
            math.isfinite(first_depth)
            and math.isfinite(second_depth)
            and abs(first_depth - second_depth) <= self.slot_match_depth_m
        )

    def _valid_vehicles(self, msg):
        return [
            obj for obj in msg.objects
            if (
                str(obj.class_name).strip().lower() in self.vehicle_classes
                and math.isfinite(float(obj.confidence))
                and float(obj.confidence) >= self.vehicle_min_confidence
                and int(obj.x_max) > int(obj.x_min)
                and int(obj.y_max) > int(obj.y_min)
            )
        ]

    def slot_is_occupied(self, slot, vehicles):
        slot_box = self._bbox(slot)
        sx0, sy0, sx1, sy1 = slot_box
        slot_w = max(1, sx1 - sx0)
        slot_h = max(1, sy1 - sy0)
        margin_x = self.vehicle_center_margin_ratio * slot_w
        margin_y = self.vehicle_center_margin_ratio * slot_h
        slot_area = max(1, self._bbox_area(slot_box))
        slot_base = self.detection_to_base_xyz(slot)

        for vehicle in vehicles:
            vehicle_box = self._bbox(vehicle)
            vehicle_area = max(1, self._bbox_area(vehicle_box))
            center_inside = (
                sx0 - margin_x <= float(vehicle.center_u) <= sx1 + margin_x
                and sy0 - margin_y <= float(vehicle.center_v) <= sy1 + margin_y
            )
            overlap = float(self._intersection_area(slot_box, vehicle_box))
            overlap_ratio = overlap / float(min(slot_area, vehicle_area))

            close_in_3d = False
            vehicle_base = self.detection_to_base_xyz(vehicle)
            if slot_base is not None and vehicle_base is not None:
                close_in_3d = math.hypot(
                    slot_base[0] - vehicle_base[0],
                    slot_base[1] - vehicle_base[1],
                ) <= self.occupied_distance_m
            if (center_inside
                    or overlap_ratio >= self.vehicle_slot_overlap_threshold
                    or close_in_3d):
                return True
        return False

    def _update_slot_confirmation_locked(self, msg):
        slots = [
            obj for obj in msg.objects
            if self.valid_detection(
                obj, self.slot_classes, self.slot_min_confidence)
        ]
        vehicles = self._valid_vehicles(msg)
        occupied_flags = {
            id(slot): self.slot_is_occupied(slot, vehicles)
            for slot in slots
        }
        self.last_slot_total = len(slots)
        self.last_slot_occupied = sum(
            1 for slot in slots if occupied_flags[id(slot)])
        self.last_slot_empty = len(slots) - self.last_slot_occupied

        matched = None
        if self.slot_candidate is not None:
            matches = [
                slot for slot in slots
                if self._slot_matches(slot, self.slot_candidate)
            ]
            if matches:
                matched = min(matches, key=lambda slot: math.hypot(
                    float(slot.center_u) - float(self.slot_candidate.center_u),
                    float(slot.center_v) - float(self.slot_candidate.center_v)))

        if matched is None:
            empty_slots = [
                slot for slot in slots if not occupied_flags[id(slot)]
            ]
            if not empty_slots:
                self._clear_slot_confirmation_locked()
                self.publish_state()
                return None
            matched = min(empty_slots, key=lambda obj: float(obj.depth_m))
            self.slot_candidate = copy.deepcopy(matched)
            self.slot_seen_count = 0
            self.empty_streak = 0
            self.occupied_streak = 0

        self.slot_candidate = copy.deepcopy(matched)
        self.slot_seen_count += 1
        self.last_slot_candidate_monotonic = time.monotonic()
        if occupied_flags[id(matched)]:
            self.empty_streak = 0
            self.occupied_streak += 1
            self.publish_state()
            return None

        self.empty_streak += 1
        self.occupied_streak = 0
        self.publish_state()
        if (self.slot_seen_count >= self.slot_confirm_frames
                and self.empty_streak >= self.empty_confirm_frames):
            return copy.deepcopy(matched)
        return None

    def _latest_slot_is_empty(self, selected_slot):
        with self.lock:
            if self._detection_age_locked() > self.max_detection_age_s:
                return False, "最终车位复查时视觉数据已过期"
            msg = copy.deepcopy(self.latest_detections)
        if msg is None:
            return False, "最终车位复查缺少 detections"
        slots = [
            obj for obj in msg.objects
            if self.valid_detection(
                obj, self.slot_classes, self.slot_min_confidence)
            and self._slot_matches(obj, selected_slot)
        ]
        if not slots:
            return False, "最终车位复查未找到同一车位"
        slot = min(slots, key=lambda obj: math.hypot(
            float(obj.center_u) - float(selected_slot.center_u),
            float(obj.center_v) - float(selected_slot.center_v)))
        if self.slot_is_occupied(slot, self._valid_vehicles(msg)):
            return False, "最终车位复查发现 vehicle 占用"
        return True, "最终车位仍为空"

    def _abandon_slot(self, execution_id, reason):
        with self.lock:
            if not self.is_execution_valid(execution_id):
                return
            self.executing_goal = False
            self.move_base_active = False
            self.selected_slot = None
            self.selected_goal_pose = None
            self._enter_search_state_locked(
                self.SEARCH_SLOT, reset_timeout=False)
            self.publish_state()
        rospy.logwarn("放弃当前车位并继续搜索：%s", reason)

    # move_base 目标、make_plan 与最终验证
    def execute_park_approach(self, park_obj, execution_id):
        try:
            if not self.is_execution_valid(execution_id):
                return
            base_xyz = self.detection_to_base_xyz(park_obj)
            if base_xyz is None:
                self.set_error("P 牌缺少有效三维位置", execution_id)
                return
            x, y, _ = base_xyz
            distance = math.hypot(x, y)
            travel = max(0.0, distance - self.park_sign_standoff_m)
            if not self.is_execution_valid(execution_id):
                return
            if travel < self.min_approach_travel_m:
                with self.lock:
                    if not self.is_execution_valid(execution_id):
                        return
                    self.executing_goal = False
                    self._enter_search_state_locked(self.SEARCH_SLOT)
                    self.publish_state()
                return
            scale = travel / distance
            goal_pose = self.local_target_to_map_pose(
                x * scale, y * scale, math.atan2(y, x))
            if not self.is_execution_valid(execution_id):
                return
            if not self.execute_move_base_goal(
                    goal_pose, "APPROACH_PARK", execution_id):
                return
            with self.lock:
                if not self.is_execution_valid(execution_id):
                    return
                self.executing_goal = False
                self._enter_search_state_locked(self.SEARCH_SLOT)
                self.publish_state()
        except (tf2_ros.TransformException, ValueError, RuntimeError) as exc:
            self.set_error("靠近 P 牌异常: %s" % exc, execution_id)
        except Exception as exc:
            self.set_error("靠近 P 牌未处理异常: %s" % exc, execution_id)

    def _make_plan_available(self, goal_pose, execution_id):
        if not self.is_execution_valid(execution_id):
            return False, "执行已取消"
        try:
            rospy.wait_for_service(
                self.make_plan_service, timeout=self.make_plan_timeout_s)
        except rospy.ROSException as exc:
            return False, "make_plan 服务不可用: %s" % exc
        if not self.is_execution_valid(execution_id):
            return False, "执行已取消"
        try:
            start = self._current_map_pose()
            request = GetPlanRequest()
            request.start = start
            request.goal = goal_pose
            request.tolerance = self.make_plan_tolerance_m
        except tf2_ros.TransformException as exc:
            return False, "make_plan 调用失败: %s" % exc
        result = {}
        completed = threading.Event()

        def call_make_plan():
            try:
                result["response"] = self.make_plan(request)
            except rospy.ServiceException as exc:
                result["error"] = str(exc)
            finally:
                completed.set()

        threading.Thread(target=call_make_plan, daemon=True).start()
        deadline = time.monotonic() + self.make_plan_timeout_s
        while not completed.wait(0.05):
            if not self.is_execution_valid(execution_id):
                return False, "执行已取消"
            if time.monotonic() >= deadline:
                return False, "make_plan 调用超时 %.1fs" % self.make_plan_timeout_s
        if not self.is_execution_valid(execution_id):
            return False, "执行已取消"
        if "error" in result:
            return False, "make_plan 调用失败: %s" % result["error"]
        response = result.get("response")
        if response is None:
            return False, "make_plan 未返回结果"
        if not response.plan.poses:
            return False, "make_plan 返回空路径"
        return True, "make_plan 路径点=%d" % len(response.plan.poses)

    def execute_slot_entry(self, slot_obj, execution_id):
        try:
            if not self.is_execution_valid(execution_id):
                return
            empty, reason = self._latest_slot_is_empty(slot_obj)
            if not empty:
                self._abandon_slot(execution_id, reason)
                return
            base_xyz = self.detection_to_base_xyz(slot_obj)
            if base_xyz is None:
                self.set_error("空车位缺少有效三维位置", execution_id)
                return
            x, y, _ = base_xyz
            distance = math.hypot(x, y)
            if distance < 1.0e-3:
                self.set_error("空车位目标距离接近 0", execution_id)
                return
            goal_x = x + x / distance * self.slot_goal_extension_m
            goal_y = y + y / distance * self.slot_goal_extension_m
            goal_pose = self.local_target_to_map_pose(
                goal_x, goal_y, math.atan2(y, x))
            if not self.is_execution_valid(execution_id):
                return
            plan_ok, plan_message = self._make_plan_available(
                goal_pose, execution_id)
            if not plan_ok:
                if self.is_execution_valid(execution_id):
                    self._abandon_slot(execution_id, plan_message)
                return
            empty, reason = self._latest_slot_is_empty(slot_obj)
            if not empty:
                self._abandon_slot(execution_id, reason)
                return
            if not self.execute_move_base_goal(
                    goal_pose, "ENTER_SLOT", execution_id):
                return
            with self.lock:
                if not self.is_execution_valid(execution_id):
                    return
                self.executing_goal = False
                self.state = self.VERIFY_PARKING
                self.publish_state()
            self.verify_final_parking(goal_pose, execution_id)
        except (tf2_ros.TransformException, ValueError, RuntimeError) as exc:
            self.set_error("驶入车位异常: %s" % exc, execution_id)
        except Exception as exc:
            self.set_error("驶入车位未处理异常: %s" % exc, execution_id)

    def execute_move_base_goal(self, goal_pose, purpose, execution_id):
        if not self.is_execution_valid(execution_id):
            return False
        if not self.move_base.wait_for_server(
                rospy.Duration(self.action_server_timeout_s)):
            self.set_error("等待 move_base action server 超时", execution_id)
            return False
        if not self.is_execution_valid(execution_id):
            return False

        goal = MoveBaseGoal()
        goal.target_pose = copy.deepcopy(goal_pose)
        goal.target_pose.header.stamp = rospy.Time.now()
        with self.lock:
            if not self.is_execution_valid(execution_id):
                return False
            self.selected_goal_pose = copy.deepcopy(goal.target_pose)
            self.goal_pub.publish(goal.target_pose)
            self.move_base_active = True
            self.publish_state()
        if not self.is_execution_valid(execution_id):
            return False
        self.move_base.send_goal(goal)
        if not self.is_execution_valid(execution_id):
            self.move_base.cancel_goal()
            return False
        finished = self.move_base.wait_for_result(
            rospy.Duration(self.goal_timeout_s))
        if not self.is_execution_valid(execution_id):
            return False
        with self.lock:
            self.move_base_active = False
            self.publish_state()
        if not finished:
            self.move_base.cancel_goal()
            self.set_error("%s 目标执行超时 %.1f 秒" % (
                purpose, self.goal_timeout_s), execution_id)
            return False
        status = self.move_base.get_state()
        if status != GoalStatus.SUCCEEDED:
            self.set_error("%s 目标失败 status=%d" % (
                purpose, status), execution_id)
            return False
        return True

    def verify_final_parking(self, goal_pose, execution_id):
        target_x = float(goal_pose.pose.position.x)
        target_y = float(goal_pose.pose.position.y)
        target_yaw = self._pose_yaw(goal_pose.pose)
        started = time.monotonic()
        stable_started = None
        rate = rospy.Rate(10.0)

        while time.monotonic() - started <= self.final_verify_timeout_s:
            if not self.is_execution_valid(execution_id):
                return
            try:
                current = self._current_map_pose()
            except tf2_ros.TransformException:
                rate.sleep()
                continue
            with self.lock:
                odom_fresh = (
                    self.latest_odom is not None
                    and time.monotonic() - self.last_odom_received_monotonic
                    <= self.odom_max_age_s
                )
                speed = self.current_speed_mps
            position_error = math.hypot(
                current.pose.position.x - target_x,
                current.pose.position.y - target_y,
            )
            yaw_error = abs(self._wrap_angle(
                self._pose_yaw(current.pose) - target_yaw))
            with self.lock:
                self.final_position_error_m = position_error
                self.final_yaw_error_rad = yaw_error
                self.publish_state()
            conditions_ok = (
                odom_fresh
                and math.isfinite(speed)
                and position_error <= self.final_position_tolerance_m
                and yaw_error <= self.final_yaw_tolerance_rad
                and speed <= self.final_speed_tolerance_mps
            )
            if conditions_ok:
                if stable_started is None:
                    stable_started = time.monotonic()
                elif time.monotonic() - stable_started >= self.final_stable_time_s:
                    restored, message = self._restore_dwa()
                    if not restored:
                        self.set_error(message, execution_id)
                        return
                    with self.lock:
                        if not self.is_execution_valid(execution_id):
                            return
                        self.parking_active = False
                        self.executing_goal = False
                        self.move_base_active = False
                        self.state = self.FINISH
                        self.used = True
                        self.last_error = ""
                        self.done_pub.publish(Bool(data=True))
                        self._publish_stop_once()
                        self.publish_state()
                    rospy.logwarn("停车最终验证通过，流程完成")
                    return
            else:
                stable_started = None
            rate.sleep()

        self.set_error(
            "最终验证超时 position=%.3f yaw=%.3f speed=%s" % (
                self.final_position_error_m,
                self.final_yaw_error_rad,
                "nan" if not math.isfinite(self.current_speed_mps)
                else "%.3f" % self.current_speed_mps,
            ),
            execution_id,
        )

    # 状态与退出
    def publish_state(self):
        with self.lock:
            detection_age = self._detection_age_locked()
            vision_ready, _ = self._vision_preflight_locked()
            missing = sorted(self.required_classes - self.available_classes)
            selected_slot = "NONE"
            status_slot = self.selected_slot or self.slot_candidate
            if status_slot is not None:
                selected_slot = "%d,%d,%.2f" % (
                    int(status_slot.center_u),
                    int(status_slot.center_v),
                    float(status_slot.depth_m),
                )
            text = (
                "state=%s active=%s execution_id=%d vision_ready=%s "
                "detection_age=%s available_classes=%s missing_classes=%s "
                "selected_slot=%s slot_seen_count=%d empty_streak=%d "
                "occupied_streak=%d move_base_active=%s "
                "final_position_error=%s final_yaw_error=%s current_speed=%s "
                "error=%s" % (
                    self.state,
                    self.parking_active,
                    self.execution_id,
                    vision_ready,
                    "inf" if not math.isfinite(detection_age)
                    else "%.3f" % detection_age,
                    ",".join(sorted(self.available_classes)) or "NONE",
                    ",".join(missing) or "NONE",
                    selected_slot,
                    self.slot_seen_count,
                    self.empty_streak,
                    self.occupied_streak,
                    self.move_base_active,
                    "nan" if not math.isfinite(self.final_position_error_m)
                    else "%.3f" % self.final_position_error_m,
                    "nan" if not math.isfinite(self.final_yaw_error_rad)
                    else "%.3f" % self.final_yaw_error_rad,
                    "nan" if not math.isfinite(self.current_speed_mps)
                    else "%.3f" % self.current_speed_mps,
                    self.last_error.replace(" ", "_")
                    if self.last_error else "NONE",
                )
            )
            self.state_pub.publish(String(data=text))

    def shutdown(self):
        with self.lock:
            self.execution_id += 1
            self.parking_active = False
            self.executing_goal = False
            self.move_base_active = False
            self._publish_stop_once()
        try:
            self.move_base.cancel_all_goals()
        except Exception as exc:
            rospy.logwarn("shutdown 取消 move_base 目标失败: %s", exc)
        restored, message = self._restore_dwa()
        if not restored:
            rospy.logerr("shutdown 时%s", message)


def main():
    rospy.init_node("parking_manager", anonymous=False)
    ParkingManager()
    rospy.spin()


if __name__ == "__main__":
    main()
