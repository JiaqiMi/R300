#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous parking controller and end-of-route search coordinator for R300.

Integrated radar-navigation behaviour:

1. Wait for ``/subject1/waypoint_status`` to report ``state=COMPLETED``.
2. Reuse or start the RealSense camera, then start
   ``R300_vision/r300_yolo_detector/parking_yolo_depth.launch``.
3. Rotate only in the positive yaw direction in 30-degree steps, holding each
   direction for five seconds.
4. Accept one ``parking_empty`` target after five consecutive valid frames,
   publish its WGS84/map representations, stop the search, temporarily reduce
   local/global radar inflation to the parking radius, and hand a single new
   goal to the existing ``move_base`` action.
5. Restore the original inflation/tolerance after the parking goal, then remain
   ``FINISHED`` after arrival; no automatic rearm or repeated parking
   mission is performed.

The standalone launch keeps its historical immediate-detection behaviour by
default; waypoint gating and active rotation are enabled only by the radar
navigation integration arguments.
"""

from __future__ import annotations

import math
import os
import signal
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import actionlib
import rospy
import tf.transformations as tft
import tf2_ros

from actionlib_msgs.msg import GoalStatus
from dynamic_reconfigure.client import Client as DynamicReconfigureClient
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from r300_autonomous_parking.msg import ParkingTarget
from r300_vision_msgs.msg import DetectedObject, DetectedObjectArray
from sensor_msgs.msg import CameraInfo, Image, NavSatFix
from std_msgs.msg import Bool, Float64, Int32, String
from std_srvs.srv import Trigger, TriggerResponse


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    x, y, z, w = tft.quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


def quaternion_to_yaw(q: Quaternion) -> float:
    return tft.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return (
        (n + alt_m) * cos_lat * cos_lon,
        (n + alt_m) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
    )


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1.0e-9:
        lat = math.copysign(math.pi / 2.0, z)
        alt = abs(z) - WGS84_A * math.sqrt(1.0 - WGS84_E2)
        return math.degrees(lat), math.degrees(lon), alt

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    alt = 0.0
    for _ in range(12):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        cos_lat = math.cos(lat)
        alt = p / max(abs(cos_lat), 1.0e-12) - n
        denom = p * (1.0 - WGS84_E2 * n / max(n + alt, 1.0))
        next_lat = math.atan2(z, denom)
        if abs(next_lat - lat) < 1.0e-13:
            lat = next_lat
            break
        lat = next_lat
    return math.degrees(lat), math.degrees(lon), alt


def enu_to_geodetic(
    east_m: float,
    north_m: float,
    up_m: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_m: float,
) -> Tuple[float, float, float]:
    x0, y0, z0 = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
    lat0 = math.radians(origin_lat_deg)
    lon0 = math.radians(origin_lon_deg)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    dx = -sin_lon * east_m - sin_lat * cos_lon * north_m + cos_lat * cos_lon * up_m
    dy = cos_lon * east_m - sin_lat * sin_lon * north_m + cos_lat * sin_lon * up_m
    dz = cos_lat * north_m + sin_lat * up_m
    return ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)


@dataclass(frozen=True)
class ParkingSample:
    map_x: float
    map_y: float
    confidence: float
    depth_m: float
    camera_x_right_m: float
    camera_y_down_m: float
    camera_z_forward_m: float
    east_offset_m: float
    north_offset_m: float
    latitude: float
    longitude: float
    altitude: float
    wall_time: float


class ImmediateParkingNode:
    WAITING_WAYPOINTS = "WAITING_WAYPOINTS"
    PREPARING = "PREPARING"
    ARMED = "ARMED"
    SEARCHING = "SEARCHING"
    COUNTING = "COUNTING"
    TARGET_PUBLISHED = "TARGET_PUBLISHED"
    NAVIGATING = "NAVIGATING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process_lock = threading.RLock()
        self.dwa_lock = threading.RLock()
        self.inflation_lock = threading.RLock()

        # Essential ROS interfaces.
        self.goal_frame = str(rospy.get_param("~goal_frame", "map"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.move_base_action = str(rospy.get_param("~move_base_action", "/move_base"))
        self.fix_topic = str(rospy.get_param("~fix_topic", "/one_x/fix"))
        self.heading_topic = str(rospy.get_param("~heading_topic", "/one_x/heading_deg"))
        self.detections_topic = str(
            rospy.get_param("~parking_detections_topic", "/r300_parking_vision/detections")
        )
        self.pause_waypoints_service = str(
            rospy.get_param("~pause_waypoints_service", "/subject1/pause_waypoints")
        )

        # Mission orchestration.  In the integrated radar-navigation mode this
        # node remains lightweight until the normal GPS waypoint executor
        # reports state=COMPLETED.  Standalone launch can keep the historical
        # immediate behaviour by setting wait_for_waypoint_completion=false.
        self.wait_for_waypoint_completion = bool(
            rospy.get_param("~wait_for_waypoint_completion", False)
        )
        self.waypoint_status_topic = str(
            rospy.get_param("~waypoint_status_topic", "/subject1/waypoint_status")
        )
        self.search_enabled = bool(rospy.get_param("~search_enabled", False))
        self.search_cmd_vel_topic = str(
            rospy.get_param("~search_cmd_vel_topic", "/subject1/cmd_vel_raw")
        )
        self.search_step_deg = float(rospy.get_param("~search_step_deg", 30.0))
        self.search_hold_s = float(rospy.get_param("~search_hold_s", 5.0))
        # Integrated parking keeps repeating the 30-degree scan indefinitely.
        # search_cycles is retained only as a finite fallback for standalone tests
        # when search_continuous is explicitly disabled.
        self.search_continuous = bool(rospy.get_param("~search_continuous", True))
        self.search_cycles = max(1, int(rospy.get_param("~search_cycles", 1)))
        self.search_angular_speed_radps = float(
            rospy.get_param("~search_angular_speed_radps", 0.25)
        )
        self.search_min_angular_speed_radps = float(
            rospy.get_param("~search_min_angular_speed_radps", 0.08)
        )
        self.search_yaw_kp = float(rospy.get_param("~search_yaw_kp", 1.2))
        self.search_yaw_tolerance_deg = float(
            rospy.get_param("~search_yaw_tolerance_deg", 2.0)
        )
        self.search_control_hz = float(rospy.get_param("~search_control_hz", 20.0))
        self.search_step_timeout_s = float(
            rospy.get_param("~search_step_timeout_s", 10.0)
        )

        # Exact requested trigger: conf >= 0.20 for five consecutive frames.
        self.empty_class_name = str(rospy.get_param("~empty_class_name", "parking_empty"))
        self.min_confidence = float(rospy.get_param("~min_empty_confidence", 0.20))
        self.required_frames = max(1, int(rospy.get_param("~required_frames", 5)))
        self.max_frame_gap_s = float(rospy.get_param("~max_frame_gap_s", 1.0))
        self.max_target_jump_m = float(rospy.get_param("~max_target_jump_m", 1.5))
        self.min_depth_m = float(rospy.get_param("~min_target_depth_m", 0.40))
        self.max_depth_m = float(rospy.get_param("~max_target_depth_m", 15.0))
        self.navigation_freshness_s = float(rospy.get_param("~navigation_data_freshness_s", 1.5))
        self.reject_zero_fix = bool(rospy.get_param("~reject_zero_fix", True))
        self.move_base_wait_s = float(rospy.get_param("~move_base_wait_s", 15.0))
        self.goal_timeout_s = float(rospy.get_param("~parking_goal_timeout_s", 60.0))

        # The normal GPS waypoint mission intentionally uses a loose tolerance,
        # but the final parking goal requires precise arrival.  This node updates
        # only the live DWA dynamic-reconfigure server for the parking goal and
        # restores the previous value afterwards.
        self.dwa_reconfigure_namespace = str(
            rospy.get_param("~dwa_reconfigure_namespace", "/move_base/DWAPlannerROS")
        )
        self.parking_xy_goal_tolerance_m = float(
            rospy.get_param("~parking_xy_goal_tolerance_m", 0.30)
        )
        self.dwa_reconfigure_timeout_s = float(
            rospy.get_param("~dwa_reconfigure_timeout_s", 3.0)
        )
        self.restore_dwa_goal_tolerance = bool(
            rospy.get_param("~restore_dwa_goal_tolerance_after_parking", True)
        )

        # The normal radar route keeps its configured safety inflation.  Only
        # after a stable parking target has been frozen, immediately before the
        # final move_base goal, reduce both radar-backed costmaps to the parking
        # radius.  The original live values are restored on success, failure,
        # timeout, reset, or shutdown.
        self.parking_inflation_enabled = bool(
            rospy.get_param("~parking_inflation_enabled", True)
        )
        raw_inflation_namespaces = rospy.get_param(
            "~parking_inflation_reconfigure_namespaces",
            [
                "/move_base/local_costmap/inflation_layer",
                "/move_base/global_costmap/inflation_layer",
            ],
        )
        if isinstance(raw_inflation_namespaces, str):
            raw_inflation_namespaces = [raw_inflation_namespaces]
        self.parking_inflation_namespaces = [
            str(value).strip()
            for value in raw_inflation_namespaces
            if str(value).strip()
        ]
        self.parking_inflation_radius_m = float(
            rospy.get_param("~parking_inflation_radius_m", 0.30)
        )
        self.inflation_reconfigure_timeout_s = float(
            rospy.get_param("~inflation_reconfigure_timeout_s", 3.0)
        )
        self.parking_inflation_settle_s = float(
            rospy.get_param("~parking_inflation_settle_s", 0.50)
        )
        self.restore_inflation_after_parking = bool(
            rospy.get_param("~restore_inflation_after_parking", True)
        )

        # Camera/detector preparation retained, but it is automatic and has no
        # effect on the five-frame trigger once detections are already present.
        self.rgb_topic = str(rospy.get_param("~rgb_topic", "/camera/color/image_raw"))
        self.depth_topic = str(
            rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        )
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        )
        self.annotated_topic = str(
            rospy.get_param("~parking_annotated_topic", "/r300_parking_vision/annotated_image")
        )
        self.target_point_topic = str(
            rospy.get_param("~parking_target_point_topic", "/r300_parking_vision/target_point")
        )
        self.camera_freshness_s = float(rospy.get_param("~camera_freshness_s", 1.5))
        self.camera_probe_s = float(rospy.get_param("~camera_probe_s", 2.0))
        self.camera_start_timeout_s = float(rospy.get_param("~camera_start_timeout_s", 20.0))
        self.detector_start_timeout_s = float(rospy.get_param("~detector_start_timeout_s", 45.0))
        self.detections_freshness_s = float(rospy.get_param("~detections_freshness_s", 1.0))
        self.auto_start_camera = bool(rospy.get_param("~auto_start_camera", True))
        self.auto_start_detector = bool(rospy.get_param("~auto_start_parking_detector", True))
        self.camera_serial_no = str(rospy.get_param("~camera_serial_no", ""))
        self.camera_device_type = str(rospy.get_param("~camera_device_type", "d435i"))
        self.camera_initial_reset = bool(rospy.get_param("~camera_initial_reset", False))
        self.camera_width = int(rospy.get_param("~camera_width", 640))
        self.camera_height = int(rospy.get_param("~camera_height", 480))
        self.camera_fps = int(rospy.get_param("~camera_fps", 30))
        self.model1_path = os.path.expanduser(str(rospy.get_param("~model1_path", "")))
        self.model2_path = os.path.expanduser(str(rospy.get_param("~model2_path", "")))
        self.detector_config = os.path.expanduser(
            str(rospy.get_param("~parking_detector_config", ""))
        )

        # If start_r300.sh web was used, the D435i and the regular dual-YOLO
        # detector share one parent roslaunch.  At parking transition we stop
        # only the heavy regular detector node, not the parent launch, camera,
        # target-feedback node, or web_video_server.  r300_system_dual.launch
        # therefore launches this detector as non-required.
        self.stop_regular_detector_before_parking = bool(
            rospy.get_param("~stop_regular_detector_before_parking", True)
        )
        self.regular_detector_node = str(
            rospy.get_param("~regular_detector_node", "/r300_dual_yolo_depth_node")
        )
        self.regular_detector_stop_timeout_s = float(
            rospy.get_param("~regular_detector_stop_timeout_s", 6.0)
        )
        self.regular_detector_release_wait_s = float(
            rospy.get_param("~regular_detector_release_wait_s", 1.0)
        )

        self._validate_parameters()

        self.state = self.PREPARING
        self.detail = "initializing"
        self.armed = False
        self.triggered = False
        self.samples: List[ParkingSample] = []
        self.last_frame_key: Optional[Tuple[int, int]] = None
        self.last_sample_wall = 0.0
        self.latest_fix: Optional[NavSatFix] = None
        self.latest_heading_deg: Optional[float] = None
        self.last_fix_wall = 0.0
        self.last_heading_wall = 0.0
        self.last_rgb_wall = 0.0
        self.last_depth_wall = 0.0
        self.last_camera_info_wall = 0.0
        self.last_detection_wall = 0.0
        self.navigation_thread: Optional[threading.Thread] = None
        self.generation = 0
        self.prepare_thread: Optional[threading.Thread] = None
        self.search_thread: Optional[threading.Thread] = None
        self.mission_started = False
        self.camera_process: Optional[subprocess.Popen] = None
        self.detector_process: Optional[subprocess.Popen] = None
        self.dwa_client: Optional[DynamicReconfigureClient] = None
        self.saved_xy_goal_tolerance: Optional[float] = None
        self.parking_tolerance_applied = False
        self.inflation_clients: Dict[str, DynamicReconfigureClient] = {}
        self.saved_inflation_radii: Dict[str, float] = {}
        self.parking_inflation_applied = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.move_base = actionlib.SimpleActionClient(self.move_base_action, MoveBaseAction)

        self.state_pub = rospy.Publisher(
            "/subject1/autonomous_parking/state", String, queue_size=10, latch=True
        )
        self.active_pub = rospy.Publisher(
            "/subject1/autonomous_parking/active", Bool, queue_size=2, latch=True
        )
        self.stable_count_pub = rospy.Publisher(
            "/subject1/autonomous_parking/stable_count", Int32, queue_size=10, latch=True
        )
        self.target_fix_pub = rospy.Publisher(
            "/subject1/autonomous_parking/target_fix", NavSatFix, queue_size=2, latch=True
        )
        self.target_pose_pub = rospy.Publisher(
            "/subject1/autonomous_parking/target_pose", PoseStamped, queue_size=2, latch=True
        )
        self.parking_target_pub = rospy.Publisher(
            "/subject1/autonomous_parking/parking_target",
            ParkingTarget,
            queue_size=2,
            latch=True,
        )
        self.search_cmd_pub = rospy.Publisher(
            self.search_cmd_vel_topic, Twist, queue_size=2
        )
        self.search_active_pub = rospy.Publisher(
            "/subject1/autonomous_parking/search_active",
            Bool,
            queue_size=2,
            latch=True,
        )

        rospy.Subscriber(self.fix_topic, NavSatFix, self._fix_cb, queue_size=20)
        rospy.Subscriber(self.heading_topic, Float64, self._heading_cb, queue_size=20)
        rospy.Subscriber(self.detections_topic, DetectedObjectArray, self._detections_cb, queue_size=10)
        rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(
            self.waypoint_status_topic, String, self._waypoint_status_cb, queue_size=10
        )

        rospy.Service("/subject1/autonomous_parking/reset", Trigger, self._reset_service)
        rospy.Service("/subject1/autonomous_parking/rearm", Trigger, self._reset_service)

        self.status_timer = rospy.Timer(rospy.Duration(0.5), self._status_timer_cb)
        rospy.on_shutdown(self._shutdown)
        self.active_pub.publish(Bool(data=False))
        self.search_active_pub.publish(Bool(data=False))
        self._publish_count(0)

        if self.wait_for_waypoint_completion:
            self._set_state(
                self.WAITING_WAYPOINTS,
                "waiting for normal waypoint task state=COMPLETED",
            )
        else:
            self._start_parking_mission("standalone immediate start")

    def _validate_parameters(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_empty_confidence must be in [0,1]")
        if self.required_frames < 1:
            raise ValueError("required_frames must be >= 1")
        if self.max_frame_gap_s <= 0.0 or self.max_target_jump_m <= 0.0:
            raise ValueError("frame gap and jump limits must be positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_target_depth_m must exceed min_target_depth_m")
        if not 0.0 < self.search_step_deg <= 180.0:
            raise ValueError("search_step_deg must be in (0,180]")
        if self.search_hold_s < 0.0:
            raise ValueError("search_hold_s must be >= 0")
        if self.search_angular_speed_radps <= 0.0:
            raise ValueError("search_angular_speed_radps must be > 0")
        if not 0.0 < self.search_min_angular_speed_radps <= self.search_angular_speed_radps:
            raise ValueError("search_min_angular_speed_radps must be in (0,max]")
        if self.search_yaw_tolerance_deg <= 0.0:
            raise ValueError("search_yaw_tolerance_deg must be > 0")
        if self.search_control_hz <= 0.0 or self.search_step_timeout_s <= 0.0:
            raise ValueError("search control frequency and timeout must be > 0")
        if self.parking_xy_goal_tolerance_m <= 0.0:
            raise ValueError("parking_xy_goal_tolerance_m must be > 0")
        if self.dwa_reconfigure_timeout_s <= 0.0:
            raise ValueError("dwa_reconfigure_timeout_s must be > 0")
        if self.parking_inflation_enabled and not self.parking_inflation_namespaces:
            raise ValueError(
                "parking_inflation_reconfigure_namespaces must not be empty when enabled"
            )
        if self.parking_inflation_radius_m <= 0.0:
            raise ValueError("parking_inflation_radius_m must be > 0")
        if self.inflation_reconfigure_timeout_s <= 0.0:
            raise ValueError("inflation_reconfigure_timeout_s must be > 0")
        if self.parking_inflation_settle_s < 0.0:
            raise ValueError("parking_inflation_settle_s must be >= 0")
        if self.regular_detector_stop_timeout_s <= 0.0:
            raise ValueError("regular_detector_stop_timeout_s must be > 0")
        if self.regular_detector_release_wait_s < 0.0:
            raise ValueError("regular_detector_release_wait_s must be >= 0")

    # ------------------------------------------------------------------
    # Mission trigger and search odometry
    # ------------------------------------------------------------------
    def _waypoint_status_cb(self, msg: String) -> None:
        if not self.wait_for_waypoint_completion:
            return
        fields = {}
        for token in str(msg.data).split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        if fields.get("state") != "COMPLETED":
            return
        self._start_parking_mission("normal waypoint task completed")

    def _start_parking_mission(self, reason: str) -> None:
        with self.lock:
            if self.mission_started:
                return
            self.mission_started = True
        rospy.logwarn("Autonomous parking mission triggered: %s", reason)
        self.active_pub.publish(Bool(data=True))
        self._set_state(self.PREPARING, "preparing camera and parking detector")
        self.prepare_thread = threading.Thread(
            target=self._prepare_and_arm,
            name="r300_parking_prepare",
            daemon=True,
        )
        self.prepare_thread.start()

    # ------------------------------------------------------------------
    # Basic inputs
    # ------------------------------------------------------------------
    def _fix_cb(self, msg: NavSatFix) -> None:
        valid = (
            math.isfinite(msg.latitude)
            and math.isfinite(msg.longitude)
            and abs(msg.latitude) <= 90.0
            and abs(msg.longitude) <= 180.0
        )
        if self.reject_zero_fix and abs(msg.latitude) < 1.0e-9 and abs(msg.longitude) < 1.0e-9:
            valid = False
        if not valid:
            rospy.logwarn_throttle(
                2.0,
                "parking ignored invalid fix lat=%.10f lon=%.10f",
                float(msg.latitude),
                float(msg.longitude),
            )
            return
        with self.lock:
            self.latest_fix = msg
            self.last_fix_wall = time.monotonic()

    def _heading_cb(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            return
        with self.lock:
            self.latest_heading_deg = value % 360.0
            self.last_heading_wall = time.monotonic()

    def _rgb_cb(self, _msg: Image) -> None:
        self.last_rgb_wall = time.monotonic()

    def _depth_cb(self, _msg: Image) -> None:
        self.last_depth_wall = time.monotonic()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if msg.K[0] > 0.0 and msg.K[4] > 0.0:
            self.last_camera_info_wall = time.monotonic()

    # ------------------------------------------------------------------
    # Five consecutive frame trigger
    # ------------------------------------------------------------------
    def _detections_cb(self, msg: DetectedObjectArray) -> None:
        now = time.monotonic()
        self.last_detection_wall = now
        frame_key = (int(msg.header.stamp.secs), int(msg.header.stamp.nsecs))

        with self.lock:
            if not self.armed or self.triggered:
                return
            if frame_key == self.last_frame_key:
                return
            self.last_frame_key = frame_key

        candidates = [
            obj
            for obj in msg.objects
            if str(obj.class_name) == self.empty_class_name
            and float(obj.confidence) >= self.min_confidence
            and bool(obj.depth_valid)
            and math.isfinite(float(obj.depth_m))
            and self.min_depth_m <= float(obj.depth_m) <= self.max_depth_m
            and all(
                math.isfinite(float(value))
                for value in (obj.position.x, obj.position.y, obj.position.z)
            )
        ]

        if not candidates:
            self._clear_samples("frame_without_valid_parking_empty")
            return

        # Nearest empty bay is deterministic and matches the existing perception target policy.
        selected = min(candidates, key=lambda item: float(item.depth_m))
        sample = self._make_sample(selected, now)
        if sample is None:
            self._clear_samples("navigation_or_map_pose_not_ready")
            return

        frozen: Optional[ParkingSample] = None
        with self.lock:
            if self.samples and now - self.last_sample_wall > self.max_frame_gap_s:
                self.samples = []

            if self.samples:
                ref_x = statistics.median(item.map_x for item in self.samples)
                ref_y = statistics.median(item.map_y for item in self.samples)
                jump = math.hypot(sample.map_x - ref_x, sample.map_y - ref_y)
                if jump > self.max_target_jump_m:
                    rospy.logwarn(
                        "parking_empty changed by %.3fm (> %.3fm); restart at frame 1/%d",
                        jump,
                        self.max_target_jump_m,
                        self.required_frames,
                    )
                    self.samples = [sample]
                else:
                    self.samples.append(sample)
            else:
                self.samples = [sample]

            self.last_sample_wall = now
            if len(self.samples) > self.required_frames:
                self.samples = self.samples[-self.required_frames :]
            count = len(self.samples)
            self._publish_count(count)
            self.state = self.COUNTING
            self.detail = "parking_empty conf>=%.2f consecutive=%d/%d" % (
                self.min_confidence,
                count,
                self.required_frames,
            )

            if count >= self.required_frames:
                frozen = self._freeze_samples_locked()
                self.triggered = True
                self.armed = False
                generation = self.generation

        self._publish_status()
        if frozen is not None:
            self.search_active_pub.publish(Bool(data=False))
            self._publish_zero_search_cmd()
            # Wait briefly for the positive-yaw search loop to observe
            # triggered=true. This prevents a last raw angular command from
            # racing with the newly submitted move_base parking goal.
            search_thread = self.search_thread
            if (
                search_thread is not None
                and search_thread.is_alive()
                and search_thread is not threading.current_thread()
            ):
                search_thread.join(timeout=0.50)
            self._publish_zero_search_cmd()
            self._publish_target(frozen)
            self.navigation_thread = threading.Thread(
                target=self._send_move_base_goal,
                args=(frozen, generation),
                name="r300_parking_navigation",
                daemon=True,
            )
            self.navigation_thread.start()

    def _make_sample(self, obj: DetectedObject, now: float) -> Optional[ParkingSample]:
        nav = self._navigation_snapshot(now)
        pose = self._current_map_pose(timeout_s=0.05)
        if nav is None or pose is None:
            return None
        fix, heading_deg = nav
        map_x, map_y, map_yaw = pose

        # Camera optical: x right, y down, z forward. Installation/lever-arm
        # errors are intentionally zero. Only the mandatory axis convention
        # conversion is applied.
        right = float(obj.position.x)
        down = float(obj.position.y)
        forward = float(obj.position.z)

        # Body FLU uses left positive, therefore camera right is -left.
        target_map_x = map_x + math.cos(map_yaw) * forward + math.sin(map_yaw) * right
        target_map_y = map_y + math.sin(map_yaw) * forward - math.cos(map_yaw) * right

        heading = math.radians(heading_deg)
        east = forward * math.sin(heading) + right * math.cos(heading)
        north = forward * math.cos(heading) - right * math.sin(heading)
        origin_alt = float(fix.altitude) if math.isfinite(float(fix.altitude)) else 0.0
        lat, lon, alt = enu_to_geodetic(
            east,
            north,
            0.0,
            float(fix.latitude),
            float(fix.longitude),
            origin_alt,
        )

        return ParkingSample(
            map_x=target_map_x,
            map_y=target_map_y,
            confidence=float(obj.confidence),
            depth_m=float(obj.depth_m),
            camera_x_right_m=right,
            camera_y_down_m=down,
            camera_z_forward_m=forward,
            east_offset_m=east,
            north_offset_m=north,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            wall_time=now,
        )

    def _navigation_snapshot(self, now: float) -> Optional[Tuple[NavSatFix, float]]:
        with self.lock:
            fix = self.latest_fix
            heading = self.latest_heading_deg
            fix_age = now - self.last_fix_wall
            heading_age = now - self.last_heading_wall
        if fix is None or heading is None:
            rospy.logwarn_throttle(2.0, "waiting for /one_x/fix and /one_x/heading_deg")
            return None
        if fix_age > self.navigation_freshness_s or heading_age > self.navigation_freshness_s:
            rospy.logwarn_throttle(
                2.0,
                "parking navigation data stale: fix_age=%.2fs heading_age=%.2fs",
                fix_age,
                heading_age,
            )
            return None
        return fix, heading

    def _current_map_pose(self, timeout_s: float) -> Optional[Tuple[float, float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.goal_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(timeout_s),
            )
            t = transform.transform.translation
            yaw = quaternion_to_yaw(transform.transform.rotation)
            return float(t.x), float(t.y), float(yaw)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "waiting for TF %s -> %s: %s",
                self.goal_frame,
                self.base_frame,
                exc,
            )
            return None

    def _freeze_samples_locked(self) -> ParkingSample:
        items = list(self.samples[-self.required_frames :])
        return ParkingSample(
            map_x=statistics.median(item.map_x for item in items),
            map_y=statistics.median(item.map_y for item in items),
            confidence=statistics.median(item.confidence for item in items),
            depth_m=statistics.median(item.depth_m for item in items),
            camera_x_right_m=statistics.median(item.camera_x_right_m for item in items),
            camera_y_down_m=statistics.median(item.camera_y_down_m for item in items),
            camera_z_forward_m=statistics.median(item.camera_z_forward_m for item in items),
            east_offset_m=statistics.median(item.east_offset_m for item in items),
            north_offset_m=statistics.median(item.north_offset_m for item in items),
            latitude=statistics.median(item.latitude for item in items),
            longitude=statistics.median(item.longitude for item in items),
            altitude=statistics.median(item.altitude for item in items),
            wall_time=time.monotonic(),
        )

    def _clear_samples(self, reason: str) -> None:
        with self.lock:
            had_samples = bool(self.samples)
            self.samples = []
            self.last_sample_wall = 0.0
        self._publish_count(0)
        if had_samples:
            rospy.loginfo("parking five-frame count reset: %s", reason)

    # ------------------------------------------------------------------
    # Publication and move_base handoff
    # ------------------------------------------------------------------
    def _publish_target(self, target: ParkingSample) -> None:
        pose = self._current_map_pose(timeout_s=0.20)
        yaw = 0.0
        if pose is not None:
            yaw = math.atan2(target.map_y - pose[1], target.map_x - pose[0])

        stamp = rospy.Time.now()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.goal_frame
        pose_msg.pose.position.x = target.map_x
        pose_msg.pose.position.y = target.map_y
        pose_msg.pose.orientation = yaw_to_quaternion(yaw)
        self.target_pose_pub.publish(pose_msg)

        fix_msg = NavSatFix()
        fix_msg.header.stamp = stamp
        fix_msg.header.frame_id = "wgs84"
        fix_msg.latitude = target.latitude
        fix_msg.longitude = target.longitude
        fix_msg.altitude = target.altitude
        self.target_fix_pub.publish(fix_msg)

        msg = ParkingTarget()
        msg.header.stamp = stamp
        msg.header.frame_id = "wgs84"
        msg.park = 1
        msg.latitude = target.latitude
        msg.longitude = target.longitude
        msg.altitude = target.altitude
        msg.confidence = target.confidence
        msg.stable_frames = self.required_frames
        msg.camera_x_right_m = target.camera_x_right_m
        msg.camera_y_down_m = target.camera_y_down_m
        msg.camera_z_forward_m = target.camera_z_forward_m
        msg.east_offset_m = target.east_offset_m
        msg.north_offset_m = target.north_offset_m
        self.parking_target_pub.publish(msg)

        self._set_state(
            self.TARGET_PUBLISHED,
            "parking_empty 5/5 published park=1 lat=%.10f lon=%.10f" % (
                target.latitude,
                target.longitude,
            ),
        )
        rospy.logwarn(
            "PARKING TARGET PUBLISHED: conf=%.3f frames=%d map=(%.3f,%.3f) "
            "WGS84=(%.10f,%.10f)",
            target.confidence,
            self.required_frames,
            target.map_x,
            target.map_y,
            target.latitude,
            target.longitude,
        )

    def _send_move_base_goal(self, target: ParkingSample, generation: int) -> None:
        self.active_pub.publish(Bool(data=True))
        try:
            with self.lock:
                if generation != self.generation:
                    return
            # Best effort only: no dependency on the normal waypoint state.
            self._pause_waypoints_best_effort()
            self.move_base.cancel_all_goals()
            if not self.move_base.wait_for_server(rospy.Duration(self.move_base_wait_s)):
                raise RuntimeError("move_base action server is unavailable")

            # Keep normal-route clearance unchanged.  Only the final parking
            # goal receives the reduced radar inflation and precise DWA arrival
            # tolerance.  Applying here (rather than during the scan) preserves
            # the original safety margin while the vehicle rotates in place.
            self._apply_parking_inflation_radius()
            self._apply_parking_goal_tolerance()

            pose = self._current_map_pose(timeout_s=0.50)
            if pose is None:
                raise RuntimeError("map->base_link TF unavailable before move_base goal")
            yaw = math.atan2(target.map_y - pose[1], target.map_x - pose[0])

            goal = MoveBaseGoal()
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.header.frame_id = self.goal_frame
            goal.target_pose.pose.position.x = target.map_x
            goal.target_pose.pose.position.y = target.map_y
            goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)

            self._set_state(
                self.NAVIGATING,
                "target published; goal handed directly to move_base",
            )
            self.move_base.send_goal(goal)
            finished = self.move_base.wait_for_result(rospy.Duration(self.goal_timeout_s))
            if not finished:
                self.move_base.cancel_goal()
                raise RuntimeError("parking move_base goal timed out")
            with self.lock:
                if generation != self.generation:
                    return
            state = self.move_base.get_state()
            if state != GoalStatus.SUCCEEDED:
                raise RuntimeError("parking move_base goal ended with state=%d" % state)
            self._set_state(self.FINISHED, "move_base reached parking target")
        except Exception as exc:
            rospy.logerr("Immediate autonomous parking failed: %s", exc)
            self._set_state(self.ERROR, str(exc))
        finally:
            self._restore_dwa_goal_tolerance()
            self._restore_parking_inflation_radius()
            self.active_pub.publish(Bool(data=False))

    def _apply_parking_inflation_radius(self) -> None:
        if not self.parking_inflation_enabled:
            return
        with self.inflation_lock:
            if self.parking_inflation_applied:
                return

            # Remove duplicates while retaining the configured order.
            namespaces = list(dict.fromkeys(self.parking_inflation_namespaces))
            try:
                for namespace in namespaces:
                    client = DynamicReconfigureClient(
                        namespace,
                        timeout=self.inflation_reconfigure_timeout_s,
                    )
                    current = client.get_configuration(
                        timeout=self.inflation_reconfigure_timeout_s
                    )
                    if not current or "inflation_radius" not in current:
                        raise RuntimeError(
                            "inflation_radius missing from %s configuration" % namespace
                        )
                    previous = float(current["inflation_radius"])
                    if not math.isfinite(previous):
                        raise RuntimeError(
                            "invalid inflation_radius from %s: %s"
                            % (namespace, previous)
                        )

                    # Save before updating so a partially completed multi-layer
                    # change can always be rolled back.
                    self.inflation_clients[namespace] = client
                    self.saved_inflation_radii[namespace] = previous
                    self.parking_inflation_applied = True

                    updated = client.update_configuration(
                        {"inflation_radius": self.parking_inflation_radius_m}
                    )
                    actual = float(updated.get("inflation_radius", float("nan")))
                    if (
                        not math.isfinite(actual)
                        or abs(actual - self.parking_inflation_radius_m) > 1.0e-3
                    ):
                        raise RuntimeError(
                            "%s rejected parking inflation_radius %.3f (actual=%s)"
                            % (namespace, self.parking_inflation_radius_m, actual)
                        )
                    rospy.logwarn(
                        "Parking navigation temporarily changed %s/inflation_radius: "
                        "%.3f -> %.3f m",
                        namespace,
                        previous,
                        actual,
                    )

                if self.parking_inflation_settle_s > 0.0:
                    rospy.loginfo(
                        "Waiting %.2fs for parking inflation reinflation to settle",
                        self.parking_inflation_settle_s,
                    )
                    rospy.sleep(self.parking_inflation_settle_s)
            except Exception as exc:
                # A partial multi-layer update must always roll back, even
                # when normal post-parking restoration was explicitly disabled.
                self._restore_parking_inflation_radius(force=True)
                raise RuntimeError(
                    "failed to set parking costmap inflation radius: %s" % exc
                )

    def _restore_parking_inflation_radius(self, force: bool = False) -> None:
        with self.inflation_lock:
            if not self.parking_inflation_applied:
                return
            try:
                if force or self.restore_inflation_after_parking:
                    # Restore in reverse order of application.  Continue even if
                    # one server is unavailable so the other layer is not left
                    # in parking mode.
                    for namespace in reversed(list(self.saved_inflation_radii.keys())):
                        client = self.inflation_clients.get(namespace)
                        previous = self.saved_inflation_radii.get(namespace)
                        if client is None or previous is None:
                            continue
                        try:
                            updated = client.update_configuration(
                                {"inflation_radius": previous}
                            )
                            actual = float(updated.get("inflation_radius", previous))
                            rospy.logwarn(
                                "Restored %s/inflation_radius to %.3f m",
                                namespace,
                                actual,
                            )
                        except Exception as exc:
                            rospy.logerr(
                                "Failed to restore %s/inflation_radius: %s",
                                namespace,
                                exc,
                            )
            finally:
                self.inflation_clients = {}
                self.saved_inflation_radii = {}
                self.parking_inflation_applied = False

    def _apply_parking_goal_tolerance(self) -> None:
        with self.dwa_lock:
            if self.parking_tolerance_applied:
                return
            try:
                client = DynamicReconfigureClient(
                    self.dwa_reconfigure_namespace,
                    timeout=self.dwa_reconfigure_timeout_s,
                )
                current = client.get_configuration(timeout=self.dwa_reconfigure_timeout_s)
                if not current or "xy_goal_tolerance" not in current:
                    raise RuntimeError("xy_goal_tolerance missing from DWA configuration")
                previous = float(current["xy_goal_tolerance"])
                # Save the old value before updating.  If the service call
                # changes the server and then raises, the exception path can
                # still restore the original tolerance.
                self.dwa_client = client
                self.saved_xy_goal_tolerance = previous
                self.parking_tolerance_applied = True
                try:
                    updated = client.update_configuration(
                        {"xy_goal_tolerance": self.parking_xy_goal_tolerance_m}
                    )
                    actual = float(updated.get("xy_goal_tolerance", float("nan")))
                except Exception:
                    self._restore_dwa_goal_tolerance()
                    raise
                if not math.isfinite(actual) or abs(actual - self.parking_xy_goal_tolerance_m) > 1.0e-3:
                    self._restore_dwa_goal_tolerance()
                    raise RuntimeError(
                        "DWA rejected parking xy_goal_tolerance %.3f (actual=%s)"
                        % (self.parking_xy_goal_tolerance_m, actual)
                    )
                rospy.logwarn(
                    "Parking navigation temporarily changed %s/xy_goal_tolerance: %.3f -> %.3f m",
                    self.dwa_reconfigure_namespace,
                    previous,
                    actual,
                )
            except Exception as exc:
                raise RuntimeError("failed to set parking DWA goal tolerance: %s" % exc)

    def _restore_dwa_goal_tolerance(self) -> None:
        with self.dwa_lock:
            if not self.parking_tolerance_applied:
                return
            client = self.dwa_client
            previous = self.saved_xy_goal_tolerance
            try:
                if self.restore_dwa_goal_tolerance and client is not None and previous is not None:
                    updated = client.update_configuration({"xy_goal_tolerance": previous})
                    actual = float(updated.get("xy_goal_tolerance", previous))
                    rospy.logwarn(
                        "Restored %s/xy_goal_tolerance to %.3f m",
                        self.dwa_reconfigure_namespace,
                        actual,
                    )
            except Exception as exc:
                rospy.logerr("Failed to restore DWA xy_goal_tolerance: %s", exc)
            finally:
                self.dwa_client = None
                self.saved_xy_goal_tolerance = None
                self.parking_tolerance_applied = False

    def _pause_waypoints_best_effort(self) -> None:
        try:
            rospy.wait_for_service(self.pause_waypoints_service, timeout=0.30)
            response = rospy.ServiceProxy(self.pause_waypoints_service, Trigger)()
            if response.success:
                rospy.logwarn("Normal waypoint executor paused before parking goal")
            else:
                rospy.logwarn("Waypoint pause service returned failure: %s", response.message)
        except Exception:
            rospy.loginfo("Waypoint pause service unavailable; continuing with move_base handoff")

    # ------------------------------------------------------------------
    # Automatic camera/detector preparation
    # ------------------------------------------------------------------
    def _prepare_and_arm(self) -> None:
        try:
            # Release the regular dual-YOLO process before loading the parking
            # models.  Camera verification is deliberately performed after the
            # stop, so an old/incorrect required=true launch cannot leave the
            # parking detector without images.
            self._stop_regular_detector_for_parking()
            self._ensure_camera()
            self._ensure_detector()
            with self.lock:
                self.samples = []
                self.triggered = False
                self.armed = True
            self._publish_count(0)
            if self.search_enabled:
                self._set_state(
                    self.ARMED,
                    "parking detector ready; starting positive 30deg search",
                )
                self.search_thread = threading.Thread(
                    target=self._run_positive_search,
                    name="r300_parking_search",
                    daemon=True,
                )
                self.search_thread.start()
            else:
                self._set_state(
                    self.ARMED,
                    "waiting automatically for parking_empty conf>=0.20, 5 consecutive frames",
                )
        except Exception as exc:
            self._publish_zero_search_cmd()
            self.active_pub.publish(Bool(data=False))
            rospy.logerr("Autonomous parking preparation failed: %s", exc)
            self._set_state(self.ERROR, str(exc))

    def _publish_zero_search_cmd(self) -> None:
        try:
            self.search_cmd_pub.publish(Twist())
        except Exception:
            pass

    def _search_cancelled(self) -> bool:
        with self.lock:
            return self.triggered or not self.armed

    def _wait_for_search_yaw(self, timeout_s: float) -> float:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            pose = self._current_map_pose(timeout_s=0.05)
            if pose is not None:
                return float(pose[2])
            rospy.sleep(0.05)
        raise RuntimeError(
            "no search pose from TF %s -> %s" % (self.goal_frame, self.base_frame)
        )

    def _hold_search_heading(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        rate = rospy.Rate(max(2.0, self.search_control_hz / 2.0))
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self._search_cancelled():
                return False
            self._publish_zero_search_cmd()
            rate.sleep()
        return not self._search_cancelled()

    def _run_positive_search(self) -> None:
        self.search_active_pub.publish(Bool(data=True))
        try:
            # Waypoint executor has already reached COMPLETED, but cancel any
            # stale action goal before taking temporary cmd_vel ownership.
            self.move_base.cancel_all_goals()
            start_yaw = self._wait_for_search_yaw(5.0)
            previous_yaw = start_yaw
            current_yaw_unwrapped = start_yaw
            step_rad = math.radians(self.search_step_deg)
            tolerance_rad = math.radians(self.search_yaw_tolerance_deg)
            directions_per_cycle = max(1, int(round(360.0 / self.search_step_deg)))
            finite_total_steps = directions_per_cycle * self.search_cycles
            rate = rospy.Rate(self.search_control_hz)
            step_index = 0

            # In integrated mode this loop intentionally has no one-circle stop.
            # _detections_cb sets triggered=true and armed=false as soon as the
            # stable parking target is frozen; _search_cancelled() then stops the
            # rotation/hold loop before the move_base parking goal is submitted.
            while not rospy.is_shutdown():
                if self._search_cancelled():
                    return
                if not self.search_continuous and step_index >= finite_total_steps:
                    break

                direction_index = step_index % directions_per_cycle + 1
                cycle_index = step_index // directions_per_cycle + 1
                target_yaw = start_yaw + (step_index + 1) * step_rad
                if self.search_continuous:
                    detail = (
                        "continuous positive scan cycle=%d direction=%d/%d "
                        "target_step=%.1fdeg hold=%.1fs"
                        % (
                            cycle_index,
                            direction_index,
                            directions_per_cycle,
                            self.search_step_deg,
                            self.search_hold_s,
                        )
                    )
                else:
                    detail = (
                        "positive scan direction=%d/%d target_step=%.1fdeg hold=%.1fs"
                        % (
                            step_index + 1,
                            finite_total_steps,
                            self.search_step_deg,
                            self.search_hold_s,
                        )
                    )
                self._set_state(self.SEARCHING, detail)
                deadline = time.monotonic() + self.search_step_timeout_s

                while not rospy.is_shutdown():
                    if self._search_cancelled():
                        return
                    pose = self._current_map_pose(timeout_s=0.05)
                    if pose is None:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "search pose unavailable from TF %s -> %s" % (
                                    self.goal_frame, self.base_frame
                                )
                            )
                        rate.sleep()
                        continue
                    wrapped_yaw = float(pose[2])
                    delta = math.atan2(
                        math.sin(wrapped_yaw - previous_yaw),
                        math.cos(wrapped_yaw - previous_yaw),
                    )
                    current_yaw_unwrapped += delta
                    previous_yaw = wrapped_yaw
                    error = target_yaw - current_yaw_unwrapped
                    if error <= tolerance_rad:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "timeout rotating to parking scan cycle=%d direction=%d/%d"
                            % (cycle_index, direction_index, directions_per_cycle)
                        )

                    cmd = Twist()
                    omega = min(
                        self.search_angular_speed_radps,
                        max(self.search_min_angular_speed_radps, self.search_yaw_kp * error),
                    )
                    cmd.angular.z = max(0.0, omega)
                    self.search_cmd_pub.publish(cmd)
                    rate.sleep()

                self._publish_zero_search_cmd()
                if not self._hold_search_heading(self.search_hold_s):
                    return
                step_index += 1

            self._publish_zero_search_cmd()
            if not self._search_cancelled():
                self._set_state(
                    self.ARMED,
                    "configured finite parking scan completed; detector remains armed",
                )
                rospy.logwarn(
                    "Parking search completed %d finite direction(s) without a fixed target; "
                    "vehicle stopped and detector remains armed",
                    finite_total_steps,
                )
        except Exception as exc:
            self._publish_zero_search_cmd()
            with self.lock:
                triggered = self.triggered
            if not triggered:
                self.active_pub.publish(Bool(data=False))
                rospy.logerr("Parking search failed: %s", exc)
                self._set_state(self.ERROR, str(exc))
        finally:
            self.search_active_pub.publish(Bool(data=False))
            self._publish_zero_search_cmd()

    def _ros_node_exists(self, node_name: str) -> bool:
        try:
            result = subprocess.run(
                ["rosnode", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(1.0, self.regular_detector_stop_timeout_s),
                check=False,
            )
        except Exception as exc:
            raise RuntimeError("failed to query ROS nodes: %s" % exc)
        if result.returncode != 0:
            raise RuntimeError("rosnode list failed: %s" % result.stdout.strip())
        return node_name in {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _stop_regular_detector_for_parking(self) -> None:
        if not self.stop_regular_detector_before_parking:
            rospy.loginfo("Regular detector shutdown before parking is disabled")
            return
        node_name = self.regular_detector_node.strip()
        if not node_name:
            rospy.loginfo("No regular_detector_node configured; nothing to stop")
            return
        if not node_name.startswith("/"):
            node_name = "/" + node_name
        if not self._ros_node_exists(node_name):
            rospy.loginfo("Regular detector %s is not running; parking continues", node_name)
            return

        rospy.logwarn(
            "Stopping regular detector %s before parking; D435i and web remain active",
            node_name,
        )
        try:
            result = subprocess.run(
                ["rosnode", "kill", node_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.regular_detector_stop_timeout_s,
                check=False,
            )
        except Exception as exc:
            raise RuntimeError("failed to stop regular detector %s: %s" % (node_name, exc))

        deadline = time.monotonic() + self.regular_detector_stop_timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if not self._ros_node_exists(node_name):
                if self.regular_detector_release_wait_s > 0.0:
                    rospy.sleep(self.regular_detector_release_wait_s)
                rospy.logwarn(
                    "Regular detector stopped; GPU is reserved for parking recognition"
                )
                return
            rospy.sleep(0.10)
        raise RuntimeError(
            "regular detector did not stop within %.1fs: %s; rosnode output=%s"
            % (self.regular_detector_stop_timeout_s, node_name, result.stdout.strip())
        )

    def _camera_ready(self) -> bool:
        now = time.monotonic()
        return all(
            now - value <= self.camera_freshness_s
            for value in (
                self.last_rgb_wall,
                self.last_depth_wall,
                self.last_camera_info_wall,
            )
        )

    def _ensure_camera(self) -> None:
        deadline = time.monotonic() + self.camera_probe_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if self._camera_ready():
                rospy.loginfo("RealSense streams already active")
                return
            rospy.sleep(0.10)

        flags = (
            time.monotonic() - self.last_rgb_wall <= self.camera_freshness_s,
            time.monotonic() - self.last_depth_wall <= self.camera_freshness_s,
            time.monotonic() - self.last_camera_info_wall <= self.camera_freshness_s,
        )
        if any(flags):
            wait_deadline = time.monotonic() + self.camera_start_timeout_s
            while time.monotonic() < wait_deadline and not rospy.is_shutdown():
                if self._camera_ready():
                    return
                rospy.sleep(0.10)
            raise RuntimeError("camera partially active (rgb=%s depth=%s info=%s)" % flags)
        if not self.auto_start_camera:
            raise RuntimeError("camera is off and auto_start_camera=false")

        command = [
            "roslaunch", "realsense2_camera", "rs_camera.launch",
            "align_depth:=true", "enable_sync:=true",
            "enable_color:=true", "enable_depth:=true",
            "enable_infra1:=false", "enable_infra2:=false",
            "enable_fisheye:=false", "enable_gyro:=false",
            "enable_accel:=false", "enable_pointcloud:=false",
            "publish_tf:=true",
            "color_width:=%d" % self.camera_width,
            "color_height:=%d" % self.camera_height,
            "color_fps:=%d" % self.camera_fps,
            "depth_width:=%d" % self.camera_width,
            "depth_height:=%d" % self.camera_height,
            "depth_fps:=%d" % self.camera_fps,
            "initial_reset:=%s" % str(self.camera_initial_reset).lower(),
            "device_type:=%s" % self.camera_device_type,
        ]
        if self.camera_serial_no:
            command.append("serial_no:=%s" % self.camera_serial_no)
        self._start_owned_process("camera", command)
        self._wait_until(self._camera_ready, self.camera_start_timeout_s, "RealSense camera", "camera")

    def _detector_ready(self) -> bool:
        return time.monotonic() - self.last_detection_wall <= self.detections_freshness_s

    def _ensure_detector(self) -> None:
        if self._detector_ready():
            rospy.loginfo("parking detector already publishing")
            return
        if not self.auto_start_detector:
            raise RuntimeError("parking detector is off and auto_start_parking_detector=false")
        for label, path in (
            ("model1_path", self.model1_path),
            ("model2_path", self.model2_path),
            ("parking_detector_config", self.detector_config),
        ):
            if not path or not Path(path).is_file():
                raise RuntimeError("%s does not exist: %s" % (label, path))

        # Use the official parking detector launch from R300_vision.  The
        # camera has already been prepared/reused above, so this launch only
        # starts the parking YOLO+depth node and publishes to the isolated
        # topics from parking_perception.yaml.
        command = [
            "roslaunch", "r300_yolo_detector", "parking_yolo_depth.launch",
            "enable_camera:=false",
            "enable_web:=false",
            "model1_path:=%s" % self.model1_path,
            "model2_path:=%s" % self.model2_path,
            "config_file:=%s" % self.detector_config,
        ]
        self._start_owned_process("detector", command)
        self._wait_until(
            self._detector_ready,
            self.detector_start_timeout_s,
            "parking detector",
            "detector",
        )

    def _start_owned_process(self, name: str, command: Sequence[str]) -> None:
        with self.process_lock:
            current = self.camera_process if name == "camera" else self.detector_process
            if current is not None and current.poll() is None:
                return
            log_dir = Path.home() / ".ros" / "r300_autonomous_parking"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(str(log_dir / (name + ".log")), "ab", buffering=0)
            rospy.logwarn("Starting owned %s: %s", name, " ".join(command))
            process = subprocess.Popen(
                list(command),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=os.environ.copy(),
            )
            process._r300_log_handle = log_handle  # type: ignore[attr-defined]
            if name == "camera":
                self.camera_process = process
            else:
                self.detector_process = process

    def _wait_until(self, predicate, timeout_s: float, label: str, process_name: str) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if predicate():
                return
            process = self.camera_process if process_name == "camera" else self.detector_process
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    "%s exited with code %s; see ~/.ros/r300_autonomous_parking/%s.log"
                    % (process_name, process.returncode, process_name)
                )
            rospy.sleep(0.10)
        raise RuntimeError("timeout waiting for %s" % label)

    def _stop_owned_process(self, name: str) -> None:
        with self.process_lock:
            process = self.camera_process if name == "camera" else self.detector_process
            if process is None:
                return
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                    process.wait(timeout=5.0)
                except Exception:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except Exception:
                        pass
            handle = getattr(process, "_r300_log_handle", None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            if name == "camera":
                self.camera_process = None
            else:
                self.detector_process = None

    # ------------------------------------------------------------------
    # Reset/status/shutdown
    # ------------------------------------------------------------------
    def _reset_service(self, _req) -> TriggerResponse:
        self.move_base.cancel_all_goals()
        self._restore_dwa_goal_tolerance()
        self._restore_parking_inflation_radius()
        self.search_active_pub.publish(Bool(data=False))
        self._publish_zero_search_cmd()
        with self.lock:
            self.generation += 1
            self.samples = []
            self.last_sample_wall = 0.0
            self.last_frame_key = None
            self.triggered = False
            self.armed = True
        self.active_pub.publish(Bool(data=True))
        self._publish_count(0)
        self._set_state(
            self.ARMED,
            "manually rearmed; waiting for parking_empty conf>=0.20, 5 consecutive frames",
        )
        return TriggerResponse(success=True, message="parking cancelled and manually rearmed")

    def _publish_count(self, count: int) -> None:
        self.stable_count_pub.publish(Int32(data=int(count)))

    def _set_state(self, state: str, detail: str) -> None:
        with self.lock:
            self.state = state
            self.detail = detail
        rospy.loginfo("parking state=%s detail=%s", state, detail)
        self._publish_status()

    def _publish_status(self) -> None:
        with self.lock:
            count = len(self.samples)
            text = "state=%s stable=%d/%d detail=%s" % (
                self.state,
                count,
                self.required_frames,
                self.detail,
            )
        self.state_pub.publish(String(data=text))

    def _status_timer_cb(self, _event) -> None:
        self._publish_status()

    def _shutdown(self) -> None:
        self.search_active_pub.publish(Bool(data=False))
        self._publish_zero_search_cmd()
        try:
            self.move_base.cancel_all_goals()
        except Exception:
            pass
        self._restore_dwa_goal_tolerance()
        self._restore_parking_inflation_radius()
        self._stop_owned_process("detector")
        self._stop_owned_process("camera")


def main() -> None:
    rospy.init_node("r300_autonomous_parking", anonymous=False)
    ImmediateParkingNode()
    rospy.spin()


if __name__ == "__main__":
    main()
