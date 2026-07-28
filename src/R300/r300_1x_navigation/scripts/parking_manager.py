#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛保底版自主泊车状态机。

流程：
    GPS 航点全部完成
      -> 原地间歇旋转搜索 park 标志
      -> 根据视觉三维点生成 map 下的 move_base 临时目标并靠近标志
      -> 原地间歇旋转搜索 parking_slot
      -> 使用 vehicle 检测结果排除被占用车位
      -> 选择最近的空车位，前进驶向车位中心附近
      -> 停车完成

设计原则：
1. 不做倒车，不写单独的泊车规划器。
2. 只有搜索阶段直接发布低速角速度；正常移动全部交给 move_base/DWA。
3. RealSense 没有接入 TF，因此沿用视觉记录节点的相机外参思路，
   先把相机光学坐标转换到 base_link，再通过现有 TF 转到 map。
4. 不修改 waypoint_executor：直接监听其已有的
   /subject1/waypoint_status，并在 state=COMPLETED 时启动泊车。
"""

import copy
import math
import threading
import time

import actionlib
import rospy
import tf.transformations as tft
import tf2_geometry_msgs  # noqa: F401  注册 PoseStamped 的 tf2 转换
import tf2_ros

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from r300_vision_msgs.msg import DetectedObjectArray
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse


class ParkingManager:
    WAIT_MISSION = "WAIT_MISSION"
    SEARCH_PARK = "SEARCH_PARK"
    GO_PARK = "GO_PARK"
    SEARCH_SLOT = "SEARCH_SLOT"
    ENTER_SLOT = "ENTER_SLOT"
    FINISH = "FINISH"
    ERROR = "ERROR"

    def __init__(self):
        self.lock = threading.RLock()

        # ------------------------------------------------------------------
        # 1. ROS 接口
        # ------------------------------------------------------------------
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.detections_topic = str(rospy.get_param(
            "~detections_topic", "/r300_vision/detections"))
        self.waypoint_status_topic = str(rospy.get_param(
            "~waypoint_status_topic", "/subject1/waypoint_status"))
        self.cmd_vel_topic = str(rospy.get_param(
            "~cmd_vel_topic", "/subject1/cmd_vel_raw"))
        self.move_base_action = str(rospy.get_param(
            "~move_base_action", "/move_base"))
        self.map_frame = str(rospy.get_param("~map_frame", "map"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))

        self.auto_start_on_waypoint_completed = bool(rospy.get_param(
            "~auto_start_on_waypoint_completed", True))
        self.single_use = bool(rospy.get_param("~single_use", True))

        # ------------------------------------------------------------------
        # 2. YOLO 类别和检测阈值
        # ------------------------------------------------------------------
        self.park_classes = self._class_set(rospy.get_param(
            "~park_classes", ["park"]))
        self.slot_classes = self._class_set(rospy.get_param(
            "~slot_classes", ["parking_slot"]))
        self.vehicle_classes = self._class_set(rospy.get_param(
            "~vehicle_classes", ["vehicle"]))

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
        self.max_detection_age_s = max(0.05, float(rospy.get_param(
            "~max_detection_age_s", 0.8)))
        self.min_bbox_width_px = max(1, int(rospy.get_param(
            "~min_bbox_width_px", 20)))
        self.min_bbox_height_px = max(1, int(rospy.get_param(
            "~min_bbox_height_px", 20)))
        self.min_target_distance_m = max(0.1, float(rospy.get_param(
            "~min_target_distance_m", 0.4)))
        self.max_target_distance_m = max(
            self.min_target_distance_m,
            float(rospy.get_param("~max_target_distance_m", 15.0)))

        # ------------------------------------------------------------------
        # 3. 搜索动作
        # ------------------------------------------------------------------
        self.search_angular_speed_radps = abs(float(rospy.get_param(
            "~search_angular_speed_radps", 0.28)))
        self.search_direction = 1.0 if float(rospy.get_param(
            "~search_direction", 1.0)) >= 0.0 else -1.0
        self.search_rotate_s = max(0.1, float(rospy.get_param(
            "~search_rotate_s", 0.9)))
        self.search_pause_s = max(0.1, float(rospy.get_param(
            "~search_pause_s", 0.55)))
        self.search_timeout_s = max(5.0, float(rospy.get_param(
            "~search_timeout_s", 90.0)))

        # ------------------------------------------------------------------
        # 4. 靠近 P 牌和驶入车位
        # ------------------------------------------------------------------
        # 由于当前 DWA 的 xy_goal_tolerance 较大，P 牌目标设置得稍近一些，
        # 实际车辆通常会在更远处判定到达，仍能保留安全距离。
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

        # ------------------------------------------------------------------
        # 5. 空车位判断
        # ------------------------------------------------------------------
        self.vehicle_center_margin_ratio = min(0.45, max(0.0, float(
            rospy.get_param("~vehicle_center_margin_ratio", 0.08))))
        self.vehicle_overlap_ratio_threshold = min(1.0, max(0.0, float(
            rospy.get_param("~vehicle_overlap_ratio_threshold", 0.45))))
        self.occupied_distance_m = max(0.1, float(rospy.get_param(
            "~occupied_distance_m", 1.8)))

        # ------------------------------------------------------------------
        # 6. 相机光学坐标 -> ROS base_link(FLU) 外参
        # 相机光学系：X 右、Y 下、Z 前
        # base_link： X 前、Y 左、Z 上
        # 默认：Xb=Zc, Yb=-Xc, Zb=-Yc
        # ------------------------------------------------------------------
        rotation_values = rospy.get_param(
            "~camera_to_base_rotation",
            [
                0.0, 0.0, 1.0,
                -1.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
            ],
        )
        translation_values = rospy.get_param(
            "~camera_in_base_translation_m", [0.30, 0.00, 0.10])
        if len(rotation_values) != 9:
            raise ValueError("camera_to_base_rotation 必须包含 9 个数")
        if len(translation_values) != 3:
            raise ValueError("camera_in_base_translation_m 必须包含 3 个数")
        self.camera_to_base_rotation = [float(v) for v in rotation_values]
        self.camera_in_base_translation_m = [
            float(v) for v in translation_values]

        # ------------------------------------------------------------------
        # 7. 运行状态
        # ------------------------------------------------------------------
        self.state = self.WAIT_MISSION
        self.used = False
        self.executing_goal = False
        self.last_error = ""
        self.search_started_monotonic = 0.0
        self.search_cycle_started_monotonic = 0.0
        self.search_motion_was_active = False

        self.candidate_kind = None
        self.candidate_u = None
        self.candidate_depth_m = None
        self.candidate_count = 0
        self.last_candidate_monotonic = 0.0

        self.selected_park = None
        self.selected_slot = None
        self.selected_goal_pose = None
        self.last_slot_total = 0
        self.last_slot_occupied = 0
        self.last_slot_empty = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.move_base = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction)

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

        rospy.Service(
            "/subject1/parking/start", Trigger, self.start_service_cb)
        rospy.Service(
            "/subject1/parking/reset", Trigger, self.reset_service_cb)

        rospy.Timer(rospy.Duration(0.1), self.control_timer_cb)
        rospy.Timer(rospy.Duration(1.0), lambda _event: self.publish_state())
        rospy.on_shutdown(self.shutdown)

        self.publish_state()
        rospy.logwarn(
            "Parking manager ready: enabled=%s detections=%s "
            "park=%s slot=%s vehicle=%s",
            self.enabled,
            self.detections_topic,
            sorted(self.park_classes),
            sorted(self.slot_classes),
            sorted(self.vehicle_classes),
        )

    # ======================================================================
    # 基础工具
    # ======================================================================
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
    def _finite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _bbox(obj):
        return (
            int(obj.x_min), int(obj.y_min),
            int(obj.x_max), int(obj.y_max),
        )

    @staticmethod
    def _bbox_area(box):
        x0, y0, x1, y1 = box
        return max(0, x1 - x0) * max(0, y1 - y0)

    @staticmethod
    def _intersection_area(box_a, box_b):
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        x0 = max(ax0, bx0)
        y0 = max(ay0, by0)
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        return max(0, x1 - x0) * max(0, y1 - y0)

    def detection_to_base_xyz(self, obj):
        """把 DetectedObject.position 从相机光学系转到 base_link。"""
        if not bool(obj.depth_valid):
            return None
        camera = [
            float(obj.position.x),
            float(obj.position.y),
            float(obj.position.z),
        ]
        if not all(math.isfinite(v) for v in camera):
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
        if name not in classes:
            return False
        if float(obj.confidence) < min_confidence:
            return False
        if int(obj.x_max) - int(obj.x_min) < self.min_bbox_width_px:
            return False
        if int(obj.y_max) - int(obj.y_min) < self.min_bbox_height_px:
            return False
        base_xyz = self.detection_to_base_xyz(obj)
        if base_xyz is None:
            return False
        horizontal_distance = math.hypot(base_xyz[0], base_xyz[1])
        return (
            self.min_target_distance_m
            <= horizontal_distance
            <= self.max_target_distance_m
        )

    def local_target_to_map_pose(self, local_x, local_y, local_yaw):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time(0)
        pose.header.frame_id = self.base_frame
        pose.pose.position.x = float(local_x)
        pose.pose.position.y = float(local_y)
        pose.pose.position.z = 0.0
        q = tft.quaternion_from_euler(0.0, 0.0, float(local_yaw))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        return self.tf_buffer.transform(
            pose,
            self.map_frame,
            rospy.Duration(self.tf_timeout_s),
        )

    # ======================================================================
    # 启动、复位和状态
    # ======================================================================
    def waypoint_status_cb(self, msg):
        if not self.enabled or not self.auto_start_on_waypoint_completed:
            return
        waypoint_state = self._parse_status_state(msg.data)
        if waypoint_state != "COMPLETED":
            return
        with self.lock:
            if self.executing_goal:
                return
            if self.single_use and self.used:
                return
            if self.state != self.WAIT_MISSION:
                return
        rospy.logwarn("全部 GPS 航点完成，自动进入停车搜索模式")
        self.start_parking(cancel_existing_goal=False)

    def start_service_cb(self, _request):
        with self.lock:
            if self.executing_goal:
                return TriggerResponse(False, "停车 move_base 目标正在执行")
        self.start_parking(cancel_existing_goal=True)
        return TriggerResponse(True, "已手动启动停车搜索")

    def reset_service_cb(self, _request):
        with self.lock:
            if self.executing_goal:
                self.move_base.cancel_all_goals()
            self.executing_goal = False
            self.state = self.WAIT_MISSION
            self.used = False
            self.last_error = ""
            self.selected_park = None
            self.selected_slot = None
            self.selected_goal_pose = None
            self._clear_confirmation()
            self.done_pub.publish(Bool(data=False))
            self._publish_stop_once()
            self.publish_state()
        return TriggerResponse(True, "停车状态已复位")

    def start_parking(self, cancel_existing_goal):
        with self.lock:
            if self.single_use and self.used:
                rospy.logwarn("停车流程已执行过；请先调用 /subject1/parking/reset")
                return
            if cancel_existing_goal:
                self.move_base.cancel_all_goals()
            self.last_error = ""
            self.selected_park = None
            self.selected_slot = None
            self.selected_goal_pose = None
            self.done_pub.publish(Bool(data=False))
            self._enter_search_state(self.SEARCH_PARK)
            self.publish_state()

    def _enter_search_state(self, state):
        self.state = state
        self.search_started_monotonic = time.monotonic()
        self.search_cycle_started_monotonic = self.search_started_monotonic
        self.search_motion_was_active = False
        self._clear_confirmation()
        rospy.logwarn("停车状态切换：%s", state)

    def _clear_confirmation(self):
        self.candidate_kind = None
        self.candidate_u = None
        self.candidate_depth_m = None
        self.candidate_count = 0
        self.last_candidate_monotonic = 0.0

    def set_error(self, message):
        with self.lock:
            self.state = self.ERROR
            self.executing_goal = False
            self.last_error = str(message)
            self.used = True
            self._publish_stop_once()
            self.publish_state()
        rospy.logerr("停车流程失败：%s", message)

    # ======================================================================
    # 搜索阶段：间歇旋转，停顿时确认 YOLO
    # ======================================================================
    def _search_is_paused(self):
        period = self.search_rotate_s + self.search_pause_s
        elapsed = time.monotonic() - self.search_cycle_started_monotonic
        phase = elapsed % period
        return phase >= self.search_rotate_s

    def _publish_search_rotation(self):
        cmd = Twist()
        cmd.angular.z = (
            self.search_direction * self.search_angular_speed_radps)
        self.cmd_pub.publish(cmd)
        self.search_motion_was_active = True

    def _publish_stop_once(self):
        self.cmd_pub.publish(Twist())
        self.search_motion_was_active = False

    def control_timer_cb(self, _event):
        with self.lock:
            if self.state not in (self.SEARCH_PARK, self.SEARCH_SLOT):
                return
            if self.executing_goal:
                return
            search_elapsed = time.monotonic() - self.search_started_monotonic
            if search_elapsed > self.search_timeout_s:
                search_name = "P 牌" if self.state == self.SEARCH_PARK else "空车位"
                self.set_error("搜索%s超时 %.1f 秒" % (
                    search_name, self.search_timeout_s))
                return

            if self._search_is_paused():
                if self.search_motion_was_active:
                    self._publish_stop_once()
            else:
                self._publish_search_rotation()

    # ======================================================================
    # YOLO 回调和稳定目标确认
    # ======================================================================
    def detections_cb(self, msg):
        if msg.header.stamp != rospy.Time(0):
            age = (rospy.Time.now() - msg.header.stamp).to_sec()
            if age < -0.2 or age > self.max_detection_age_s:
                return

        with self.lock:
            if self.executing_goal:
                return
            if self.state not in (self.SEARCH_PARK, self.SEARCH_SLOT):
                return
            # 旋转阶段不确认目标，只在短暂停顿阶段看稳定图像。
            if not self._search_is_paused():
                return

            if self.state == self.SEARCH_PARK:
                candidate = self._best_park_candidate(msg)
                if candidate is None:
                    self._maybe_clear_stale_confirmation()
                    return
                if not self._confirm_candidate("PARK", candidate):
                    return

                self.selected_park = copy.deepcopy(candidate)
                self._publish_stop_once()
                self.state = self.GO_PARK
                self.executing_goal = True
                self.publish_state()
                threading.Thread(
                    target=self.execute_park_approach,
                    args=(self.selected_park,),
                    daemon=True,
                ).start()
                return

            candidate, slot_total, occupied_count, empty_count = (
                self._best_empty_slot(msg))
            self.last_slot_total = slot_total
            self.last_slot_occupied = occupied_count
            self.last_slot_empty = empty_count
            if candidate is None:
                self._maybe_clear_stale_confirmation()
                return
            if not self._confirm_candidate("SLOT", candidate):
                return

            self.selected_slot = copy.deepcopy(candidate)
            self._publish_stop_once()
            self.state = self.ENTER_SLOT
            self.executing_goal = True
            self.publish_state()
            threading.Thread(
                target=self.execute_slot_entry,
                args=(self.selected_slot,),
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
        # 优先高置信度；置信度接近时选择距离更近的 P 牌。
        return max(
            candidates,
            key=lambda obj: (
                float(obj.confidence),
                -float(obj.depth_m),
            ),
        )

    def _confirm_candidate(self, kind, obj):
        now = time.monotonic()
        u = int(obj.center_u)
        depth = float(obj.depth_m)
        gap = (
            float("inf")
            if self.last_candidate_monotonic <= 0.0
            else now - self.last_candidate_monotonic
        )
        same_candidate = (
            self.candidate_kind == kind
            and self.candidate_u is not None
            and abs(u - self.candidate_u) <= self.candidate_match_px
            and self.candidate_depth_m is not None
            and abs(depth - self.candidate_depth_m) <= 2.0
            and gap <= self.confirm_max_gap_s
        )
        if same_candidate:
            self.candidate_count += 1
        else:
            self.candidate_kind = kind
            self.candidate_count = 1

        self.candidate_u = u
        self.candidate_depth_m = depth
        self.last_candidate_monotonic = now
        self.publish_state()
        return self.candidate_count >= self.confirm_frames

    def _maybe_clear_stale_confirmation(self):
        if (
            self.last_candidate_monotonic > 0.0
            and time.monotonic() - self.last_candidate_monotonic
            > self.confirm_max_gap_s
        ):
            self._clear_confirmation()
            self.publish_state()

    # ======================================================================
    # 空车位筛选
    # ======================================================================
    def _best_empty_slot(self, msg):
        slots = [
            obj for obj in msg.objects
            if self.valid_detection(
                obj, self.slot_classes, self.slot_min_confidence)
        ]
        vehicles = [
            obj for obj in msg.objects
            if (
                str(obj.class_name).strip().lower() in self.vehicle_classes
                and float(obj.confidence) >= self.vehicle_min_confidence
            )
        ]

        empty_slots = []
        occupied_count = 0
        for slot in slots:
            if self.slot_is_occupied(slot, vehicles):
                occupied_count += 1
            else:
                empty_slots.append(slot)

        if not empty_slots:
            return None, len(slots), occupied_count, 0

        # 最简单、最容易驶入的选择：当前视野中距离最近的空车位。
        best = min(empty_slots, key=lambda obj: float(obj.depth_m))
        return best, len(slots), occupied_count, len(empty_slots)

    def slot_is_occupied(self, slot, vehicles):
        slot_box = self._bbox(slot)
        sx0, sy0, sx1, sy1 = slot_box
        slot_w = max(1, sx1 - sx0)
        slot_h = max(1, sy1 - sy0)
        margin_x = self.vehicle_center_margin_ratio * slot_w
        margin_y = self.vehicle_center_margin_ratio * slot_h
        slot_base = self.detection_to_base_xyz(slot)

        for vehicle in vehicles:
            vehicle_box = self._bbox(vehicle)
            vehicle_area = max(1, self._bbox_area(vehicle_box))
            center_x = float(vehicle.center_u)
            center_y = float(vehicle.center_v)

            center_inside = (
                sx0 - margin_x <= center_x <= sx1 + margin_x
                and sy0 - margin_y <= center_y <= sy1 + margin_y
            )
            overlap_ratio = (
                float(self._intersection_area(slot_box, vehicle_box))
                / float(vehicle_area)
            )

            close_in_3d = False
            vehicle_base = self.detection_to_base_xyz(vehicle)
            if slot_base is not None and vehicle_base is not None:
                close_in_3d = math.hypot(
                    slot_base[0] - vehicle_base[0],
                    slot_base[1] - vehicle_base[1],
                ) <= self.occupied_distance_m

            if (
                center_inside
                or overlap_ratio >= self.vehicle_overlap_ratio_threshold
                or close_in_3d
            ):
                return True
        return False

    # ======================================================================
    # move_base 目标执行
    # ======================================================================
    def execute_park_approach(self, park_obj):
        try:
            base_xyz = self.detection_to_base_xyz(park_obj)
            if base_xyz is None:
                self.set_error("P 牌缺少有效深度或三维位置")
                return
            x, y, _ = base_xyz
            distance = math.hypot(x, y)
            travel = max(0.0, distance - self.park_sign_standoff_m)

            if travel < self.min_approach_travel_m:
                rospy.logwarn(
                    "P 牌已足够近：distance=%.2fm，直接开始搜索白框",
                    distance,
                )
                with self.lock:
                    self.executing_goal = False
                    self._enter_search_state(self.SEARCH_SLOT)
                    self.publish_state()
                return

            scale = travel / distance
            goal_x = x * scale
            goal_y = y * scale
            goal_yaw = math.atan2(y, x)
            goal_pose = self.local_target_to_map_pose(
                goal_x, goal_y, goal_yaw)

            rospy.logwarn(
                "发现 P 牌：distance=%.2fm，向其方向前进 %.2fm，"
                "局部目标 x=%.2f y=%.2f",
                distance, travel, goal_x, goal_y,
            )
            if not self.execute_move_base_goal(goal_pose, "GO_PARK"):
                return

            with self.lock:
                self.executing_goal = False
                self._enter_search_state(self.SEARCH_SLOT)
                self.publish_state()
        except Exception as exc:
            self.set_error("靠近 P 牌异常：%s" % exc)

    def execute_slot_entry(self, slot_obj):
        try:
            base_xyz = self.detection_to_base_xyz(slot_obj)
            if base_xyz is None:
                self.set_error("空车位缺少有效深度或三维位置")
                return
            x, y, _ = base_xyz
            distance = math.hypot(x, y)
            if distance < 1.0e-3:
                self.set_error("空车位目标距离接近 0，拒绝发送")
                return

            # 沿“车辆 -> 白框中心”方向再延伸一点。当前 DWA 的
            # xy_goal_tolerance 较大，这个延伸可提高车头进入白框的概率。
            unit_x = x / distance
            unit_y = y / distance
            goal_x = x + unit_x * self.slot_goal_extension_m
            goal_y = y + unit_y * self.slot_goal_extension_m
            goal_yaw = math.atan2(y, x)
            goal_pose = self.local_target_to_map_pose(
                goal_x, goal_y, goal_yaw)

            rospy.logwarn(
                "选择空车位：slot_distance=%.2fm，局部进入目标 "
                "x=%.2f y=%.2f extension=%.2fm",
                distance, goal_x, goal_y, self.slot_goal_extension_m,
            )
            if not self.execute_move_base_goal(goal_pose, "ENTER_SLOT"):
                return

            with self.lock:
                self.executing_goal = False
                self.state = self.FINISH
                self.used = True
                self.last_error = ""
                self.done_pub.publish(Bool(data=True))
                self._publish_stop_once()
                self.publish_state()
            rospy.logwarn("停车流程完成：无人车已驶向选定空车位")
        except Exception as exc:
            self.set_error("驶入车位异常：%s" % exc)

    def execute_move_base_goal(self, goal_pose, purpose):
        if not self.move_base.wait_for_server(
            rospy.Duration(self.action_server_timeout_s)
        ):
            self.set_error("等待 move_base action server 超时")
            return False

        goal = MoveBaseGoal()
        goal.target_pose = goal_pose
        goal.target_pose.header.stamp = rospy.Time.now()
        self.selected_goal_pose = copy.deepcopy(goal.target_pose)
        self.goal_pub.publish(goal.target_pose)
        self.publish_state()

        rospy.logwarn(
            "发送停车 move_base 目标：purpose=%s frame=%s x=%.3f y=%.3f",
            purpose,
            goal.target_pose.header.frame_id,
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
        )
        self.move_base.send_goal(goal)
        finished = self.move_base.wait_for_result(
            rospy.Duration(self.goal_timeout_s))
        if not finished:
            self.move_base.cancel_goal()
            self.set_error("%s 目标执行超时 %.1f 秒" % (
                purpose, self.goal_timeout_s))
            return False

        status = self.move_base.get_state()
        if status != GoalStatus.SUCCEEDED:
            self.set_error("%s 目标失败，move_base status=%d" % (
                purpose, status))
            return False
        return True

    # ======================================================================
    # 状态输出和退出
    # ======================================================================
    def publish_state(self):
        with self.lock:
            goal_x = float("nan")
            goal_y = float("nan")
            if self.selected_goal_pose is not None:
                goal_x = float(self.selected_goal_pose.pose.position.x)
                goal_y = float(self.selected_goal_pose.pose.position.y)
            text = (
                "state=%s enabled=%s used=%s executing_goal=%s "
                "candidate=%s count=%d/%d slots=%d occupied=%d empty=%d "
                "goal_x=%s goal_y=%s error=%s" % (
                    self.state,
                    self.enabled,
                    self.used,
                    self.executing_goal,
                    self.candidate_kind or "NONE",
                    self.candidate_count,
                    self.confirm_frames,
                    self.last_slot_total,
                    self.last_slot_occupied,
                    self.last_slot_empty,
                    "nan" if not math.isfinite(goal_x) else "%.3f" % goal_x,
                    "nan" if not math.isfinite(goal_y) else "%.3f" % goal_y,
                    self.last_error.replace(" ", "_")
                    if self.last_error else "NONE",
                )
            )
            self.state_pub.publish(String(data=text))

    def shutdown(self):
        try:
            self._publish_stop_once()
            if self.executing_goal:
                self.move_base.cancel_goal()
        except Exception:
            pass


def main():
    rospy.init_node("parking_manager", anonymous=False)
    ParkingManager()
    rospy.spin()


if __name__ == "__main__":
    main()
