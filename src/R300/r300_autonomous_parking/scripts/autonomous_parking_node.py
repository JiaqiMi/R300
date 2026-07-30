#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent autonomous parking orchestration for the R300.

This node deliberately does not modify or replace the existing navigation,
lidar, or vision source code. It coordinates already existing ROS interfaces:

* verifies that RealSense RGB/aligned-depth/camera-info streams are alive;
* starts the camera only when all required camera streams are absent;
* starts the existing parking YOLO depth node on isolated parking topics;
* uses the exact parking classes ``park``, ``parking_occupied`` and ``parking_empty``;
* confirms ``park`` first, rejects occupied bays, and only parks into ``parking_empty``;
* requires ten consecutive non-jumping detections before freezing a target;
* reads the current vehicle latitude/longitude from ``/one_x/fix`` and heading from ``/one_x/heading_deg``;
* converts the stable camera optical-frame centre directly into an absolute WGS-84 target;
* publishes one strongly typed target message containing latitude, longitude and ``park=1``;
* sends the corresponding ``move_base`` goal;
* never publishes ``cmd_vel`` directly.

The search rotation is also implemented through move_base yaw goals, so the
node remains separated from the chassis velocity command path.
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
import tf2_ros
import tf.transformations as tft

from actionlib_msgs.msg import GoalStatus
from dynamic_reconfigure.client import Client as DynamicReconfigureClient
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from r300_vision_msgs.msg import DetectedObjectArray
from sensor_msgs.msg import CameraInfo, Image, NavSatFix
from std_msgs.msg import Bool, Float64, Int32, String
from std_srvs.srv import Empty, Trigger, TriggerResponse

from r300_autonomous_parking.msg import ParkingTarget


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(q) -> float:
    return tft.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]


def yaw_to_quaternion(yaw: float):
    x, y, z, w = tft.quaternion_from_euler(0.0, 0.0, yaw)
    from geometry_msgs.msg import Quaternion

    return Quaternion(x=x, y=y, z=z, w=w)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


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
        if abs(cos_lat) < 1.0e-12:
            alt = abs(z) - n * (1.0 - WGS84_E2)
        else:
            alt = p / cos_lat - n
        denom = p * (1.0 - WGS84_E2 * n / max(n + alt, 1.0))
        next_lat = math.atan2(z, denom)
        if abs(next_lat - lat) < 1.0e-13:
            lat = next_lat
            break
        lat = next_lat

    return math.degrees(lat), math.degrees(lon), alt


def enu_to_geodetic(
    east: float,
    north: float,
    up: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_m: float,
) -> Tuple[float, float, float]:
    """Convert local ENU into WGS-84 latitude/longitude/altitude."""
    x0, y0, z0 = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
    lat0 = math.radians(origin_lat_deg)
    lon0 = math.radians(origin_lon_deg)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    return ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)


def circular_mean_deg(values: Sequence[float]) -> float:
    """Circular mean for headings in degrees, returned in [0, 360)."""
    if not values:
        return 0.0
    sin_sum = sum(math.sin(math.radians(value)) for value in values)
    cos_sum = sum(math.cos(math.radians(value)) for value in values)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def optical_to_body_frd(
    x_right_m: float,
    y_down_m: float,
    z_forward_m: float,
    roll_error_rad: float,
    pitch_error_rad: float,
    yaw_error_rad: float,
) -> Tuple[float, float, float]:
    """Camera optical XYZ -> vehicle FRD, then apply mounting-error RPY.

    Camera optical convention: x right, y down, z forward.
    Vehicle FRD convention: x forward, y right, z down.
    With all mounting errors set to zero this is only the fixed axis reorder:
    forward=z, right=x, down=y.
    """
    forward = z_forward_m
    right = x_right_m
    down = y_down_m

    cr, sr = math.cos(roll_error_rad), math.sin(roll_error_rad)
    cp, sp = math.cos(pitch_error_rad), math.sin(pitch_error_rad)
    cy, sy = math.cos(yaw_error_rad), math.sin(yaw_error_rad)

    x1 = forward
    y1 = cr * right - sr * down
    z1 = sr * right + cr * down
    x2 = cp * x1 + sp * z1
    y2 = y1
    z2 = -sp * x1 + cp * z1
    x3 = cy * x2 - sy * y2
    y3 = sy * x2 + cy * y2
    z3 = z2
    return x3, y3, z3


@dataclass
class StableSample:
    # move_base target in goal_frame
    x: float
    y: float
    z: float
    confidence: float
    depth_m: float
    center_u: int
    center_v: int
    wall_time: float

    # Original white-frame centre in camera optical coordinates.
    camera_x_right_m: float
    camera_y_down_m: float
    camera_z_forward_m: float

    # Horizontal displacement from vehicle to parking target in local ENU.
    east_offset_m: float
    north_offset_m: float

    # Vehicle navigation sample used for this visual frame.
    vehicle_latitude: float
    vehicle_longitude: float
    vehicle_altitude: float
    heading_deg: float

    # Absolute parking target.
    target_latitude: float
    target_longitude: float
    target_altitude: float


class AutonomousParkingNode:
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    SEARCH_PARK = "SEARCH_PARK"
    SEARCH_EMPTY = "SEARCH_EMPTY"
    STABILIZING = "STABILIZING"
    NAVIGATING = "NAVIGATING"
    VERIFYING = "VERIFYING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process_lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.stable_target_event = threading.Event()
        self.park_confirmed_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

        # ---------- interfaces ----------
        self.goal_frame = str(rospy.get_param("~goal_frame", "map"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.move_base_action = str(rospy.get_param("~move_base_action", "/move_base"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/subject1/dwa_odom"))
        self.fix_topic = str(rospy.get_param("~fix_topic", "/one_x/fix"))
        self.heading_topic = str(rospy.get_param("~heading_topic", "/one_x/heading_deg"))
        self.navigation_data_freshness_s = float(
            rospy.get_param("~navigation_data_freshness_s", 0.75)
        )
        self.reject_zero_fix = bool(rospy.get_param("~reject_zero_fix", True))
        self.waypoint_status_topic = str(
            rospy.get_param("~waypoint_status_topic", "/subject1/waypoint_status")
        )
        self.pause_waypoints_service = str(
            rospy.get_param("~pause_waypoints_service", "/subject1/pause_waypoints")
        )
        self.resume_waypoints_service = str(
            rospy.get_param("~resume_waypoints_service", "/subject1/resume_waypoints")
        )
        self.clear_costmaps_service = str(
            rospy.get_param("~clear_costmaps_service", "/move_base/clear_costmaps")
        )

        self.rgb_topic = str(rospy.get_param("~rgb_topic", "/camera/color/image_raw"))
        self.depth_topic = str(
            rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        )
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        )
        self.detections_topic = str(
            rospy.get_param("~parking_detections_topic", "/r300_parking_vision/detections")
        )
        self.annotated_topic = str(
            rospy.get_param("~parking_annotated_topic", "/r300_parking_vision/annotated_image")
        )
        self.target_point_topic = str(
            rospy.get_param("~parking_target_point_topic", "/r300_parking_vision/target_point")
        )

        # ---------- camera ----------
        self.camera_freshness_s = float(rospy.get_param("~camera_freshness_s", 1.5))
        self.camera_probe_s = float(rospy.get_param("~camera_probe_s", 2.0))
        self.camera_start_timeout_s = float(rospy.get_param("~camera_start_timeout_s", 20.0))
        self.camera_serial_no = str(rospy.get_param("~camera_serial_no", ""))
        self.camera_device_type = str(rospy.get_param("~camera_device_type", "d435i"))
        self.camera_initial_reset = bool(rospy.get_param("~camera_initial_reset", False))
        self.camera_width = int(rospy.get_param("~camera_width", 640))
        self.camera_height = int(rospy.get_param("~camera_height", 480))
        self.camera_fps = int(rospy.get_param("~camera_fps", 30))
        self.auto_start_camera = bool(rospy.get_param("~auto_start_camera", True))
        self.stop_owned_camera_on_shutdown = bool(
            rospy.get_param("~stop_owned_camera_on_shutdown", True)
        )
        self.stop_owned_camera_on_finish = bool(
            rospy.get_param("~stop_owned_camera_on_finish", False)
        )

        # ---------- detector ----------
        self.auto_start_parking_detector = bool(
            rospy.get_param("~auto_start_parking_detector", True)
        )
        self.detector_start_timeout_s = float(
            rospy.get_param("~detector_start_timeout_s", 45.0)
        )
        self.detections_freshness_s = float(
            rospy.get_param("~detections_freshness_s", 1.0)
        )
        self.model1_path = os.path.expanduser(str(rospy.get_param("~model1_path", "")))
        self.model2_path = os.path.expanduser(str(rospy.get_param("~model2_path", "")))
        self.parking_detector_config = os.path.expanduser(
            str(rospy.get_param("~parking_detector_config", ""))
        )

        # ---------- exact parking classes ----------
        raw_park_names = rospy.get_param("~park_class_names", ["park"])
        raw_occupied_names = rospy.get_param(
            "~occupied_class_names", ["parking_occupied"]
        )
        raw_empty_names = rospy.get_param(
            "~empty_class_names", ["parking_empty"]
        )
        self.park_class_names = {str(name) for name in raw_park_names}
        self.occupied_class_names = {str(name) for name in raw_occupied_names}
        self.empty_class_names = {str(name) for name in raw_empty_names}
        self.require_park_before_empty = bool(
            rospy.get_param("~require_park_before_empty", True)
        )
        self.min_park_confidence = float(
            rospy.get_param("~min_park_confidence", 0.30)
        )
        self.park_required_frames = max(
            1, int(rospy.get_param("~park_required_frames", 3))
        )
        self.min_occupied_confidence = float(
            rospy.get_param("~min_occupied_confidence", 0.30)
        )
        self.occupied_conflict_distance_m = float(
            rospy.get_param("~occupied_conflict_distance_m", 0.60)
        )
        self.min_empty_confidence = float(
            rospy.get_param("~min_empty_confidence", 0.60)
        )
        self.stable_required_frames = max(
            1, int(rospy.get_param("~stable_required_frames", 10))
        )
        self.max_frame_gap_s = float(rospy.get_param("~max_frame_gap_s", 0.60))
        self.max_target_jump_m = float(rospy.get_param("~max_target_jump_m", 0.25))
        self.max_target_spread_m = float(rospy.get_param("~max_target_spread_m", 0.25))
        self.min_target_depth_m = float(rospy.get_param("~min_target_depth_m", 0.40))
        self.max_target_depth_m = float(rospy.get_param("~max_target_depth_m", 15.0))
        self.min_goal_distance_m = float(rospy.get_param("~min_goal_distance_m", 0.35))
        self.max_goal_distance_m = float(rospy.get_param("~max_goal_distance_m", 20.0))
        self.candidate_policy = str(rospy.get_param("~candidate_policy", "nearest"))

        # ---------- camera/INS installation ----------
        # Requested configuration: all installation errors and lever arms are zero.
        self.camera_mount_roll_error_rad = math.radians(
            float(rospy.get_param("~camera_mount_roll_error_deg", 0.0))
        )
        self.camera_mount_pitch_error_rad = math.radians(
            float(rospy.get_param("~camera_mount_pitch_error_deg", 0.0))
        )
        self.camera_mount_yaw_error_rad = math.radians(
            float(rospy.get_param("~camera_mount_yaw_error_deg", 0.0))
        )
        self.camera_lever_forward_m = float(
            rospy.get_param("~camera_lever_forward_m", 0.0)
        )
        self.camera_lever_right_m = float(
            rospy.get_param("~camera_lever_right_m", 0.0)
        )
        self.camera_lever_down_m = float(
            rospy.get_param("~camera_lever_down_m", 0.0)
        )

        # ---------- search ----------
        self.search_enabled = bool(rospy.get_param("~search_enabled", True))
        self.search_direction = 1 if int(rospy.get_param("~search_direction", 1)) >= 0 else -1
        self.search_yaw_step = math.radians(
            float(rospy.get_param("~search_yaw_step_deg", 30.0))
        )
        self.search_max_rotation = math.radians(
            float(rospy.get_param("~search_max_rotation_deg", 360.0))
        )
        self.search_step_timeout_s = float(
            rospy.get_param("~search_step_timeout_s", 8.0)
        )
        self.search_observe_s = float(rospy.get_param("~search_observe_s", 3.5))
        self.search_settle_s = float(rospy.get_param("~search_settle_s", 0.5))
        self.search_total_timeout_s = float(
            rospy.get_param("~search_total_timeout_s", 90.0)
        )
        self.stationary_linear_mps = float(
            rospy.get_param("~stationary_linear_mps", 0.08)
        )
        self.stationary_angular_radps = float(
            rospy.get_param("~stationary_angular_radps", 0.08)
        )

        # ---------- DWA / navigation ----------
        self.use_dynamic_reconfigure = bool(
            rospy.get_param("~use_dynamic_reconfigure", True)
        )
        self.require_dwa_reconfigure_for_search = bool(
            rospy.get_param("~require_dwa_reconfigure_for_search", True)
        )
        self.dwa_namespace = str(
            rospy.get_param("~dwa_namespace", "/move_base/DWAPlannerROS")
        )
        self.search_dwa = dict(rospy.get_param("~search_dwa", {}))
        self.parking_dwa = dict(rospy.get_param("~parking_dwa", {}))
        self.parking_goal_timeout_s = float(
            rospy.get_param("~parking_goal_timeout_s", 45.0)
        )
        self.parking_retry_count = max(
            0, int(rospy.get_param("~parking_retry_count", 1))
        )
        self.retry_delay_s = float(rospy.get_param("~retry_delay_s", 1.0))
        self.goal_extension_m = float(rospy.get_param("~goal_extension_m", 0.0))

        self.final_position_tolerance_m = float(
            rospy.get_param("~final_position_tolerance_m", 0.50)
        )
        self.final_linear_speed_tolerance_mps = float(
            rospy.get_param("~final_linear_speed_tolerance_mps", 0.10)
        )
        self.final_angular_speed_tolerance_radps = float(
            rospy.get_param("~final_angular_speed_tolerance_radps", 0.10)
        )
        self.final_stable_time_s = float(rospy.get_param("~final_stable_time_s", 1.0))
        self.final_verify_timeout_s = float(
            rospy.get_param("~final_verify_timeout_s", 6.0)
        )

        self.pause_running_waypoints = bool(
            rospy.get_param("~pause_running_waypoints", True)
        )
        self.resume_waypoints_after_cancel = bool(
            rospy.get_param("~resume_waypoints_after_cancel", False)
        )
        self.auto_start_after_waypoints = bool(
            rospy.get_param("~auto_start_after_waypoints", False)
        )

        self._validate_parameters()

        # ---------- runtime ----------
        self.state = self.IDLE
        self.detail = "ready"
        self.last_rgb_wall = 0.0
        self.last_depth_wall = 0.0
        self.last_camera_info_wall = 0.0
        self.last_detections_wall = 0.0
        self.last_detection_stamp = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_fix: Optional[NavSatFix] = None
        self.latest_heading_deg: Optional[float] = None
        self.last_fix_wall = 0.0
        self.last_heading_wall = 0.0
        self.waypoint_status = ""
        self.saw_waypoint_running = False
        self.paused_waypoints_by_us = False
        self.samples: List[StableSample] = []
        self.detection_mode: Optional[str] = None
        self.park_count = 0
        self.last_park_sample_wall = 0.0
        self.frozen_target: Optional[StableSample] = None
        self.accept_samples = False
        self.last_valid_sample_wall = 0.0
        self.original_dwa_config: Optional[Dict[str, object]] = None
        self.dwa_client: Optional[DynamicReconfigureClient] = None
        self.camera_process: Optional[subprocess.Popen] = None
        self.detector_process: Optional[subprocess.Popen] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.move_base = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction
        )

        # ---------- pubs/subs/services ----------
        self.state_pub = rospy.Publisher(
            "/subject1/autonomous_parking/state", String, queue_size=10, latch=True
        )
        self.active_pub = rospy.Publisher(
            "/subject1/autonomous_parking/active", Bool, queue_size=2, latch=True
        )
        self.park_count_pub = rospy.Publisher(
            "/subject1/autonomous_parking/park_count", Int32, queue_size=10
        )
        self.stable_count_pub = rospy.Publisher(
            "/subject1/autonomous_parking/stable_count", Int32, queue_size=10
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

        rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.detections_topic, DetectedObjectArray, self._detections_cb, queue_size=5)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=20)
        rospy.Subscriber(self.fix_topic, NavSatFix, self._fix_cb, queue_size=20)
        rospy.Subscriber(self.heading_topic, Float64, self._heading_cb, queue_size=20)
        rospy.Subscriber(self.waypoint_status_topic, String, self._waypoint_status_cb, queue_size=10)

        rospy.Service(
            "/subject1/autonomous_parking/prepare", Trigger, self._prepare_service
        )
        rospy.Service(
            "/subject1/autonomous_parking/start", Trigger, self._start_service
        )
        rospy.Service(
            "/subject1/autonomous_parking/start_empty_search",
            Trigger,
            self._start_empty_search_service,
        )
        rospy.Service(
            "/subject1/autonomous_parking/reset", Trigger, self._reset_service
        )
        rospy.Service(
            "/subject1/autonomous_parking/cancel", Trigger, self._reset_service
        )

        self.status_timer = rospy.Timer(rospy.Duration(0.5), self._status_timer_cb)
        rospy.on_shutdown(self._shutdown)

        self._set_state(self.IDLE, "ready")
        rospy.loginfo(
            "Independent parking ready: park=%s occupied=%s empty=%s "
            "park_frames=%d empty_frames=%d detections=%s",
            sorted(self.park_class_names),
            sorted(self.occupied_class_names),
            sorted(self.empty_class_names),
            self.park_required_frames,
            self.stable_required_frames,
            self.detections_topic,
        )

    # ------------------------------------------------------------------
    # validation / callbacks
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        if self.stable_required_frames < 1:
            raise ValueError("stable_required_frames must be >= 1")
        for name, value in (
            ("min_park_confidence", self.min_park_confidence),
            ("min_occupied_confidence", self.min_occupied_confidence),
            ("min_empty_confidence", self.min_empty_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0,1]" % name)
        if self.occupied_conflict_distance_m <= 0.0:
            raise ValueError("occupied_conflict_distance_m must be positive")
        if self.max_target_jump_m <= 0.0 or self.max_target_spread_m <= 0.0:
            raise ValueError("target stability distances must be positive")
        if self.max_goal_distance_m <= self.min_goal_distance_m:
            raise ValueError("max_goal_distance_m must exceed min_goal_distance_m")
        if self.search_yaw_step <= 0.0:
            raise ValueError("search_yaw_step_deg must be positive")
        if self.candidate_policy not in {"nearest", "highest_confidence"}:
            raise ValueError("candidate_policy must be nearest or highest_confidence")

    def _rgb_cb(self, _msg: Image) -> None:
        self.last_rgb_wall = time.monotonic()

    def _depth_cb(self, _msg: Image) -> None:
        self.last_depth_wall = time.monotonic()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if msg.K[0] > 0.0 and msg.K[4] > 0.0:
            self.last_camera_info_wall = time.monotonic()

    def _odom_cb(self, msg: Odometry) -> None:
        with self.lock:
            self.latest_odom = msg

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
                "Ignoring invalid /one_x/fix latitude=%.10f longitude=%.10f",
                float(msg.latitude),
                float(msg.longitude),
            )
            return
        with self.lock:
            self.latest_fix = msg
            self.last_fix_wall = time.monotonic()

    def _heading_cb(self, msg: Float64) -> None:
        heading = float(msg.data)
        if not math.isfinite(heading):
            return
        with self.lock:
            self.latest_heading_deg = heading % 360.0
            self.last_heading_wall = time.monotonic()

    def _waypoint_status_cb(self, msg: String) -> None:
        data = msg.data or ""
        with self.lock:
            self.waypoint_status = data
            if "state=RUNNING" in data:
                self.saw_waypoint_running = True
            should_auto_start = (
                self.auto_start_after_waypoints
                and self.saw_waypoint_running
                and "state=COMPLETED" in data
                and self.state in {self.IDLE, self.PREPARED, self.FINISHED, self.ERROR, self.CANCELLED}
                and (self.worker is None or not self.worker.is_alive())
            )
        if should_auto_start:
            rospy.logwarn("Waypoint RUNNING->COMPLETED observed; starting autonomous parking")
            self._launch_worker("waypoint_completed", skip_park=False)

    def _detections_cb(self, msg: DetectedObjectArray) -> None:
        now = time.monotonic()
        self.last_detections_wall = now

        with self.lock:
            if self.cancel_event.is_set() or self.detection_mode is None:
                return
            mode = self.detection_mode
            latest_odom = self.latest_odom

        if not self._vehicle_stationary(latest_odom):
            if mode == "park":
                self._reset_park_count("vehicle_moving")
            else:
                self._reset_samples("vehicle_moving")
            return

        stamp_key = (msg.header.stamp.secs, msg.header.stamp.nsecs)
        with self.lock:
            if stamp_key == self.last_detection_stamp:
                return
            self.last_detection_stamp = stamp_key

        if mode == "park":
            self._process_park_frame(msg, now)
            return
        if mode != "empty":
            return

        empty_candidates: List[StableSample] = []
        occupied_candidates: List[StableSample] = []
        for obj in msg.objects:
            class_name = str(obj.class_name)
            is_empty = class_name in self.empty_class_names
            is_occupied = class_name in self.occupied_class_names
            if not is_empty and not is_occupied:
                continue

            confidence = float(obj.confidence)
            threshold = (
                self.min_empty_confidence if is_empty
                else self.min_occupied_confidence
            )
            if confidence < threshold:
                continue
            if not obj.depth_valid or not math.isfinite(float(obj.depth_m)):
                continue
            if not self.min_target_depth_m <= float(obj.depth_m) <= self.max_target_depth_m:
                continue

            sample = self._detection_to_sample(obj, confidence, now)
            if sample is None:
                continue
            if is_empty:
                empty_candidates.append(sample)
            else:
                occupied_candidates.append(sample)

        if not empty_candidates:
            self._reset_samples(
                "parking_occupied_only" if occupied_candidates else "no_valid_parking_empty"
            )
            return

        with self.lock:
            current_samples = list(self.samples)
            current_odom = self.latest_odom

        selected = self._select_candidate(
            empty_candidates, current_samples, current_odom
        )
        if selected is None:
            self._reset_samples("no_associated_parking_empty")
            return

        for occupied in occupied_candidates:
            conflict_distance = math.hypot(
                selected.x - occupied.x, selected.y - occupied.y
            )
            if conflict_distance <= self.occupied_conflict_distance_m:
                rospy.logwarn_throttle(
                    1.0,
                    "parking_empty conflicts with parking_occupied at %.3fm; "
                    "resetting 10-frame count",
                    conflict_distance,
                )
                self._reset_samples("occupied_conflict")
                return

        with self.lock:
            if self.samples and now - self.last_valid_sample_wall > self.max_frame_gap_s:
                self.samples = []

            if self.samples:
                ref_x = statistics.median(sample.x for sample in self.samples)
                ref_y = statistics.median(sample.y for sample in self.samples)
                jump = math.hypot(selected.x - ref_x, selected.y - ref_y)
                if jump > self.max_target_jump_m:
                    rospy.logwarn(
                        "parking_empty target jumped %.3fm > %.3fm; "
                        "restarting 10-frame count",
                        jump,
                        self.max_target_jump_m,
                    )
                    self.samples = [selected]
                else:
                    self.samples.append(selected)
            else:
                self.samples = [selected]

            self.last_valid_sample_wall = now
            if len(self.samples) > self.stable_required_frames:
                self.samples = self.samples[-self.stable_required_frames :]

            count = len(self.samples)
            self.stable_count_pub.publish(Int32(data=count))

            if count >= self.stable_required_frames:
                frozen = self._freeze_samples_locked()
                if frozen is not None:
                    self.frozen_target = frozen
                    self.accept_samples = False
                    self.detection_mode = None
                    self.stable_target_event.set()
                    self.detail = (
                        "parking_empty stable %d/%d conf=%.2f map=(%.3f,%.3f)"
                        % (
                            count,
                            self.stable_required_frames,
                            frozen.confidence,
                            frozen.x,
                            frozen.y,
                        )
                    )

    def _process_park_frame(
        self, msg: DetectedObjectArray, now: float
    ) -> None:
        park_candidates = [
            obj
            for obj in msg.objects
            if str(obj.class_name) in self.park_class_names
            and float(obj.confidence) >= self.min_park_confidence
        ]
        if not park_candidates:
            self._reset_park_count("no_valid_park")
            return

        best = max(park_candidates, key=lambda item: float(item.confidence))
        with self.lock:
            if (
                self.park_count > 0
                and now - self.last_park_sample_wall > self.max_frame_gap_s
            ):
                self.park_count = 0
            self.park_count += 1
            self.last_park_sample_wall = now
            count = self.park_count
            self.park_count_pub.publish(Int32(data=count))
            self.detail = (
                "park confirmed %d/%d conf=%.2f"
                % (count, self.park_required_frames, float(best.confidence))
            )
            if count >= self.park_required_frames:
                self.detection_mode = None
                self.park_confirmed_event.set()

    def _reset_park_count(self, reason: str) -> None:
        with self.lock:
            if self.park_count:
                rospy.loginfo_throttle(1.0, "reset park count: %s", reason)
            self.park_count = 0
            self.last_park_sample_wall = 0.0
            self.park_count_pub.publish(Int32(data=0))

    # ------------------------------------------------------------------
    # stable target helpers
    # ------------------------------------------------------------------
    def _navigation_snapshot(
        self,
    ) -> Optional[Tuple[NavSatFix, float, Tuple[float, float, float]]]:
        now = time.monotonic()
        with self.lock:
            fix = self.latest_fix
            heading_deg = self.latest_heading_deg
            fix_age = now - self.last_fix_wall
            heading_age = now - self.last_heading_wall
        if fix is None or heading_deg is None:
            return None
        if fix_age > self.navigation_data_freshness_s or heading_age > self.navigation_data_freshness_s:
            rospy.logwarn_throttle(
                2.0,
                "1X navigation data stale: fix_age=%.3fs heading_age=%.3fs limit=%.3fs",
                fix_age,
                heading_age,
                self.navigation_data_freshness_s,
            )
            return None
        pose = self._current_pose_in_map(timeout_s=0.05)
        if pose is None:
            return None
        return fix, heading_deg, pose

    def _detection_to_sample(
        self,
        obj,
        confidence: float,
        now: float,
    ) -> Optional[StableSample]:
        """Convert one camera optical-frame centre to map and absolute WGS-84.

        The parking vision node publishes optical coordinates as:
          x = right, y = down, z = forward.
        Requested installation settings are all zero; the parameters remain
        explicit so later calibration does not require touching control code.
        """
        x_right = float(obj.position.x)
        y_down = float(obj.position.y)
        z_forward = float(obj.position.z)
        if not all(math.isfinite(v) for v in (x_right, y_down, z_forward)):
            return None

        snapshot = self._navigation_snapshot()
        if snapshot is None:
            rospy.logwarn_throttle(
                2.0,
                "Cannot geolocate parking_empty: waiting for fresh /one_x/fix, "
                "/one_x/heading_deg and map->base_link",
            )
            return None
        fix, heading_deg, pose = snapshot

        forward, right, down = optical_to_body_frd(
            x_right,
            y_down,
            z_forward,
            self.camera_mount_roll_error_rad,
            self.camera_mount_pitch_error_rad,
            self.camera_mount_yaw_error_rad,
        )
        forward += self.camera_lever_forward_m
        right += self.camera_lever_right_m
        down += self.camera_lever_down_m

        heading_rad = math.radians(heading_deg)
        east_offset = forward * math.sin(heading_rad) + right * math.cos(heading_rad)
        north_offset = forward * math.cos(heading_rad) - right * math.sin(heading_rad)
        up_offset = -down

        vehicle_alt = float(fix.altitude) if math.isfinite(float(fix.altitude)) else 0.0
        target_lat, target_lon, target_alt = enu_to_geodetic(
            east_offset,
            north_offset,
            up_offset,
            float(fix.latitude),
            float(fix.longitude),
            vehicle_alt,
        )

        map_x, map_y, map_yaw = pose
        left = -right
        delta_map_x = math.cos(map_yaw) * forward - math.sin(map_yaw) * left
        delta_map_y = math.sin(map_yaw) * forward + math.cos(map_yaw) * left

        return StableSample(
            x=map_x + delta_map_x,
            y=map_y + delta_map_y,
            z=0.0,
            confidence=confidence,
            depth_m=float(obj.depth_m),
            center_u=int(obj.center_u),
            center_v=int(obj.center_v),
            wall_time=now,
            camera_x_right_m=x_right,
            camera_y_down_m=y_down,
            camera_z_forward_m=z_forward,
            east_offset_m=east_offset,
            north_offset_m=north_offset,
            vehicle_latitude=float(fix.latitude),
            vehicle_longitude=float(fix.longitude),
            vehicle_altitude=vehicle_alt,
            heading_deg=heading_deg,
            target_latitude=target_lat,
            target_longitude=target_lon,
            target_altitude=target_alt,
        )

    def _select_candidate(
        self,
        candidates: Sequence[StableSample],
        current_samples: Sequence[StableSample],
        latest_odom: Optional[Odometry],
    ) -> Optional[StableSample]:
        if not candidates:
            return None

        if current_samples:
            ref_x = statistics.median(sample.x for sample in current_samples)
            ref_y = statistics.median(sample.y for sample in current_samples)
            return min(candidates, key=lambda item: math.hypot(item.x - ref_x, item.y - ref_y))

        if self.candidate_policy == "highest_confidence":
            return max(candidates, key=lambda item: item.confidence)

        pose = self._current_pose_in_map(timeout_s=0.02)
        if pose is not None:
            px, py, _ = pose
            return min(candidates, key=lambda item: math.hypot(item.x - px, item.y - py))

        if latest_odom is not None:
            px = float(latest_odom.pose.pose.position.x)
            py = float(latest_odom.pose.pose.position.y)
            return min(candidates, key=lambda item: math.hypot(item.x - px, item.y - py))

        return min(candidates, key=lambda item: item.depth_m)

    def _freeze_samples_locked(self) -> Optional[StableSample]:
        if len(self.samples) < self.stable_required_frames:
            return None
        median_x = statistics.median(sample.x for sample in self.samples)
        median_y = statistics.median(sample.y for sample in self.samples)
        median_z = statistics.median(sample.z for sample in self.samples)
        spread = max(
            math.hypot(sample.x - median_x, sample.y - median_y)
            for sample in self.samples
        )
        if spread > self.max_target_spread_m:
            rospy.logwarn(
                "10-frame empty spread %.3fm > %.3fm; restarting",
                spread,
                self.max_target_spread_m,
            )
            self.samples = []
            self.stable_count_pub.publish(Int32(data=0))
            return None

        median_conf = statistics.median(sample.confidence for sample in self.samples)
        if median_conf < self.min_empty_confidence:
            self.samples = []
            self.stable_count_pub.publish(Int32(data=0))
            return None

        return StableSample(
            x=median_x,
            y=median_y,
            z=median_z,
            confidence=median_conf,
            depth_m=statistics.median(sample.depth_m for sample in self.samples),
            center_u=int(round(statistics.median(sample.center_u for sample in self.samples))),
            center_v=int(round(statistics.median(sample.center_v for sample in self.samples))),
            wall_time=time.monotonic(),
            camera_x_right_m=statistics.median(sample.camera_x_right_m for sample in self.samples),
            camera_y_down_m=statistics.median(sample.camera_y_down_m for sample in self.samples),
            camera_z_forward_m=statistics.median(sample.camera_z_forward_m for sample in self.samples),
            east_offset_m=statistics.median(sample.east_offset_m for sample in self.samples),
            north_offset_m=statistics.median(sample.north_offset_m for sample in self.samples),
            vehicle_latitude=statistics.median(sample.vehicle_latitude for sample in self.samples),
            vehicle_longitude=statistics.median(sample.vehicle_longitude for sample in self.samples),
            vehicle_altitude=statistics.median(sample.vehicle_altitude for sample in self.samples),
            heading_deg=circular_mean_deg([sample.heading_deg for sample in self.samples]),
            target_latitude=statistics.median(sample.target_latitude for sample in self.samples),
            target_longitude=statistics.median(sample.target_longitude for sample in self.samples),
            target_altitude=statistics.median(sample.target_altitude for sample in self.samples),
        )

    def _reset_samples(self, reason: str) -> None:
        with self.lock:
            if self.samples:
                rospy.loginfo_throttle(1.0, "reset empty stability count: %s", reason)
            self.samples = []
            self.last_valid_sample_wall = 0.0
            self.stable_count_pub.publish(Int32(data=0))

    # ------------------------------------------------------------------
    # services / state machine
    # ------------------------------------------------------------------
    def _prepare_service(self, _req) -> TriggerResponse:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return TriggerResponse(
                    success=False,
                    message="autonomous parking worker is already running",
                )
            self.cancel_event.clear()
            self.worker = threading.Thread(
                target=self._run_prepare,
                name="r300_autonomous_parking_prepare",
                daemon=True,
            )
            self.worker.start()
        return TriggerResponse(
            success=True,
            message="camera/detector preparation accepted; no vehicle motion will be commanded",
        )

    def _run_prepare(self) -> None:
        self.active_pub.publish(Bool(data=True))
        try:
            self._set_state(self.PREPARING, "checking camera and parking detector")
            self._ensure_camera()
            self._ensure_detector()
            self._set_state(self.PREPARED, "camera and isolated parking detector are ready")
        except InterruptedError:
            self._set_state(self.CANCELLED, "preparation cancelled")
        except Exception as exc:
            rospy.logerr("Parking perception preparation failed: %s", exc)
            self._set_state(self.ERROR, str(exc))
        finally:
            self.active_pub.publish(Bool(data=False))

    def _start_service(self, _req) -> TriggerResponse:
        started, message = self._launch_worker("manual_service", skip_park=False)
        return TriggerResponse(success=started, message=message)

    def _start_empty_search_service(self, _req) -> TriggerResponse:
        """Test/competition shortcut: skip park gating and search parking_empty directly."""
        started, message = self._launch_worker(
            "manual_empty_search", skip_park=True
        )
        return TriggerResponse(success=started, message=message)

    def _launch_worker(
        self, reason: str, skip_park: bool = False
    ) -> Tuple[bool, str]:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return False, "autonomous parking is already running"
            self.cancel_event.clear()
            self.stable_target_event.clear()
            self.park_confirmed_event.clear()
            self.samples = []
            self.detection_mode = None
            self.park_count = 0
            self.frozen_target = None
            self.accept_samples = False
            self.last_detection_stamp = None
            self.paused_waypoints_by_us = False
            self.worker = threading.Thread(
                target=self._run_parking,
                args=(reason, skip_park),
                name="r300_autonomous_parking_worker",
                daemon=True,
            )
            self.worker.start()
        mode = "empty-search" if skip_park else "full"
        return True, "autonomous parking start accepted (mode=%s)" % mode

    def _reset_service(self, _req) -> TriggerResponse:
        self.cancel_event.set()
        self.stable_target_event.set()
        self.move_base.cancel_all_goals()
        with self.lock:
            self.accept_samples = False
            self.detection_mode = None
            self.samples = []
            self.park_count = 0
            self.frozen_target = None
        self._restore_dwa()
        if self.resume_waypoints_after_cancel and self.paused_waypoints_by_us:
            self._call_trigger_service(self.resume_waypoints_service, timeout_s=0.5)
        self._set_state(self.CANCELLED, "operator reset/cancel")
        self.active_pub.publish(Bool(data=False))
        return TriggerResponse(success=True, message="parking cancelled and move_base goal cleared")

    def _run_parking(self, reason: str, skip_park: bool = False) -> None:
        self.active_pub.publish(Bool(data=True))
        try:
            self._set_state(self.PREPARING, "start reason=%s" % reason)
            self._take_navigation_control()
            self._ensure_camera()
            self._ensure_detector()
            self._wait_for_navigation_prerequisites()
            self._save_dwa_configuration()
            if (
                self.search_enabled
                and self.require_dwa_reconfigure_for_search
                and self.dwa_client is None
            ):
                raise RuntimeError(
                    "DWA dynamic_reconfigure is required for safe yaw search, "
                    "but /move_base/DWAPlannerROS is unavailable"
                )

            if self.require_park_before_empty and not skip_park:
                if not self._search_for_confirmed_park():
                    raise RuntimeError("park sign was not confirmed")
            elif skip_park:
                rospy.logwarn(
                    "Skipping park confirmation by explicit start_empty_search request"
                )
            target = self._search_for_stable_empty()
            if target is None:
                raise RuntimeError("no stable empty target obtained")
            self._check_cancelled()

            target = self._apply_goal_extension(target)
            self._publish_target(target)
            self._apply_dwa_profile(self.parking_dwa, "parking")

            self._set_state(
                self.NAVIGATING,
                "moving to stable empty map=(%.3f, %.3f)" % (target.x, target.y),
            )
            if not self._navigate_to_target(target):
                raise RuntimeError("move_base failed to reach parking target")

            self._set_state(self.VERIFYING, "move_base succeeded; verifying stop")
            if not self._verify_final_parking(target):
                raise RuntimeError("final parking verification failed")

            self._set_state(self.FINISHED, "parking completed")
            rospy.logwarn("Autonomous parking FINISHED")
        except InterruptedError:
            self._set_state(self.CANCELLED, "cancelled")
        except Exception as exc:
            rospy.logerr("Autonomous parking failed: %s", exc)
            self.move_base.cancel_all_goals()
            self._set_state(self.ERROR, str(exc))
        finally:
            self.accept_samples = False
            self.detection_mode = None
            self.active_pub.publish(Bool(data=False))
            self._restore_dwa()
            if self.stop_owned_camera_on_finish:
                self._stop_owned_process("camera")

    def _take_navigation_control(self) -> None:
        self._check_cancelled()
        with self.lock:
            status = self.waypoint_status
        if "state=RUNNING" in status:
            if not self.pause_running_waypoints:
                raise RuntimeError(
                    "waypoint executor is RUNNING; pause it before autonomous parking"
                )
            success = self._call_trigger_service(self.pause_waypoints_service, timeout_s=1.0)
            if not success:
                raise RuntimeError("failed to pause running waypoint executor")
            self.paused_waypoints_by_us = True
            rospy.sleep(0.3)

        self.move_base.cancel_all_goals()
        rospy.sleep(0.25)

    # ------------------------------------------------------------------
    # camera / detector process ownership
    # ------------------------------------------------------------------
    def _camera_stream_flags(self) -> Tuple[bool, bool, bool]:
        now = time.monotonic()
        return (
            now - self.last_rgb_wall <= self.camera_freshness_s,
            now - self.last_depth_wall <= self.camera_freshness_s,
            now - self.last_camera_info_wall <= self.camera_freshness_s,
        )

    def _camera_ready(self) -> bool:
        return all(self._camera_stream_flags())

    def _ensure_camera(self) -> None:
        deadline = time.monotonic() + self.camera_probe_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            if self._camera_ready():
                rospy.loginfo("RealSense already running; no duplicate camera launch")
                return
            rospy.sleep(0.10)

        flags = self._camera_stream_flags()
        if any(flags):
            # A RealSense process may still be finishing startup. Wait longer for
            # all three streams, but never launch a second process on the same USB device.
            partial_deadline = time.monotonic() + self.camera_start_timeout_s
            while time.monotonic() < partial_deadline and not rospy.is_shutdown():
                self._check_cancelled()
                if self._camera_ready():
                    rospy.loginfo("Existing RealSense completed startup; reusing it")
                    return
                rospy.sleep(0.10)
            flags = self._camera_stream_flags()
            raise RuntimeError(
                "RealSense is partially active (rgb=%s depth=%s info=%s). "
                "Refusing to launch a second process on the same device; restart the camera "
                "with align_depth=true and enable_sync=true."
                % flags
            )
        if not self.auto_start_camera:
            raise RuntimeError("camera is not running and auto_start_camera=false")

        command = [
            "roslaunch",
            "realsense2_camera",
            "rs_camera.launch",
            "align_depth:=true",
            "enable_sync:=true",
            "enable_color:=true",
            "enable_depth:=true",
            "enable_infra1:=false",
            "enable_infra2:=false",
            "enable_fisheye:=false",
            "enable_gyro:=false",
            "enable_accel:=false",
            "enable_pointcloud:=false",
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
        self._set_state(self.PREPARING, "starting RealSense camera")
        self._wait_until(
            self._camera_ready,
            self.camera_start_timeout_s,
            "RealSense RGB/aligned-depth/camera-info",
            owned_process="camera",
        )

    def _detector_ready(self) -> bool:
        return time.monotonic() - self.last_detections_wall <= self.detections_freshness_s

    def _ensure_detector(self) -> None:
        if self._detector_ready():
            rospy.loginfo("Dedicated parking detector already publishing")
            return
        if not self.auto_start_parking_detector:
            raise RuntimeError("parking detector is not running and auto-start is disabled")

        for label, path in (
            ("model1_path", self.model1_path),
            ("model2_path", self.model2_path),
            ("parking_detector_config", self.parking_detector_config),
        ):
            if not path or not Path(path).is_file():
                raise RuntimeError("%s does not exist: %s" % (label, path))

        command = [
            "roslaunch",
            "r300_autonomous_parking",
            "parking_perception_isolated.launch",
            "model1_path:=%s" % self.model1_path,
            "model2_path:=%s" % self.model2_path,
            "config_file:=%s" % self.parking_detector_config,
            "rgb_topic:=%s" % self.rgb_topic,
            "depth_topic:=%s" % self.depth_topic,
            "camera_info_topic:=%s" % self.camera_info_topic,
            "detections_topic:=%s" % self.detections_topic,
            "annotated_topic:=%s" % self.annotated_topic,
            "target_point_topic:=%s" % self.target_point_topic,
        ]
        self._start_owned_process("detector", command)
        self._set_state(self.PREPARING, "starting isolated parking detector")
        self._wait_until(
            self._detector_ready,
            self.detector_start_timeout_s,
            "parking detections",
            owned_process="detector",
        )

    def _start_owned_process(self, name: str, command: Sequence[str]) -> None:
        with self.process_lock:
            existing = self.camera_process if name == "camera" else self.detector_process
            if existing is not None and existing.poll() is None:
                return

            log_dir = Path.home() / ".ros" / "r300_autonomous_parking"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / ("%s.log" % name)
            log_handle = open(str(log_path), "ab", buffering=0)
            rospy.logwarn("Starting owned %s process: %s", name, " ".join(command))
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

    def _owned_process(self, name: str) -> Optional[subprocess.Popen]:
        return self.camera_process if name == "camera" else self.detector_process

    def _stop_owned_process(self, name: str) -> None:
        with self.process_lock:
            process = self._owned_process(name)
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

    def _wait_until(
        self,
        predicate,
        timeout_s: float,
        description: str,
        owned_process: Optional[str] = None,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            if predicate():
                return
            if owned_process:
                process = self._owned_process(owned_process)
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        "%s process exited with code %s; see ~/.ros/r300_autonomous_parking/%s.log"
                        % (owned_process, process.returncode, owned_process)
                    )
            rospy.sleep(0.10)
        raise RuntimeError("timeout waiting for %s" % description)

    # ------------------------------------------------------------------
    # search / motion
    # ------------------------------------------------------------------
    def _wait_for_navigation_prerequisites(self) -> None:
        self._set_state(
            self.PREPARING,
            "waiting for /one_x/fix, /one_x/heading_deg, TF, odom and move_base",
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            with self.lock:
                odom_ok = self.latest_odom is not None
            navigation_ok = self._navigation_snapshot() is not None
            server_ok = self.move_base.wait_for_server(rospy.Duration(0.05))
            if navigation_ok and odom_ok and server_ok:
                return
            rospy.sleep(0.10)
        raise RuntimeError(
            "navigation prerequisites not ready "
            "(/one_x/fix, /one_x/heading_deg, odom, map TF or move_base)"
        )

    def _search_for_confirmed_park(self) -> bool:
        self._apply_dwa_profile(self.search_dwa, "search_park")
        started = time.monotonic()
        rotation = 0.0

        if self._observe_for_park(self.search_observe_s):
            return True
        if not self.search_enabled:
            return False

        self._set_state(self.SEARCH_PARK, "rotating through move_base to find park")
        while rotation + 1.0e-6 < self.search_max_rotation:
            self._check_cancelled()
            if time.monotonic() - started > self.search_total_timeout_s:
                raise RuntimeError("park search exceeded total timeout")

            pose = self._current_pose_in_map(timeout_s=0.20)
            if pose is None:
                raise RuntimeError("cannot get map->base_link pose for park search")
            x, y, yaw = pose
            next_yaw = normalize_angle(
                yaw + self.search_direction * self.search_yaw_step
            )
            self.detection_mode = None
            self._reset_park_count("starting_park_search_rotation")
            self._send_pose_goal(x, y, next_yaw)
            self._wait_action_terminal(self.search_step_timeout_s, allow_timeout=True)
            self.move_base.cancel_all_goals()
            self._wait_for_stationary(self.search_settle_s, timeout_s=3.0)
            rotation += self.search_yaw_step

            if self._observe_for_park(self.search_observe_s):
                return True

        raise RuntimeError(
            "no park found after %.0f deg rotation" % math.degrees(rotation)
        )

    def _observe_for_park(self, duration_s: float) -> bool:
        self._reset_park_count("new_park_observation_window")
        self.park_confirmed_event.clear()
        self.detection_mode = "park"
        self._set_state(
            self.SEARCH_PARK,
            "waiting for %d consecutive park frames" % self.park_required_frames,
        )
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            if self.park_confirmed_event.wait(timeout=0.05):
                self.detection_mode = None
                return True
        self.detection_mode = None
        self._reset_park_count("park_observation_window_expired")
        return False

    def _search_for_stable_empty(self) -> Optional[StableSample]:
        self._apply_dwa_profile(self.search_dwa, "search")
        started = time.monotonic()
        rotation = 0.0

        # Observe the current heading before rotating.
        if self._observe_for_stable_target(self.search_observe_s):
            return self.frozen_target
        if not self.search_enabled:
            return None

        self._set_state(
            self.SEARCH_EMPTY,
            "rotating through move_base to find parking_empty",
        )
        while rotation + 1.0e-6 < self.search_max_rotation:
            self._check_cancelled()
            if time.monotonic() - started > self.search_total_timeout_s:
                raise RuntimeError("empty search exceeded total timeout")

            pose = self._current_pose_in_map(timeout_s=0.20)
            if pose is None:
                raise RuntimeError("cannot get map->base_link pose for search")
            x, y, yaw = pose
            next_yaw = normalize_angle(yaw + self.search_direction * self.search_yaw_step)
            self.accept_samples = False
            self.detection_mode = None
            self._reset_samples("starting_empty_search_rotation")
            self._send_pose_goal(x, y, next_yaw)
            self._wait_action_terminal(self.search_step_timeout_s, allow_timeout=True)
            self.move_base.cancel_all_goals()
            self._wait_for_stationary(self.search_settle_s, timeout_s=3.0)
            rotation += self.search_yaw_step

            if self._observe_for_stable_target(self.search_observe_s):
                return self.frozen_target

        raise RuntimeError("no stable empty found after %.0f deg rotation" % math.degrees(rotation))

    def _observe_for_stable_target(self, duration_s: float) -> bool:
        self._reset_samples("new_observation_window")
        self.stable_target_event.clear()
        self.accept_samples = True
        self.detection_mode = "empty"
        self._set_state(
            self.STABILIZING,
            "waiting for %d consecutive high-confidence parking_empty frames"
            % self.stable_required_frames,
        )
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            if self.stable_target_event.wait(timeout=0.05):
                self.accept_samples = False
                self.detection_mode = None
                return self.frozen_target is not None
        self.accept_samples = False
        self.detection_mode = None
        self._reset_samples("observation_window_expired")
        return False

    def _send_pose_goal(self, x: float, y: float, yaw: float) -> None:
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)
        self.move_base.send_goal(goal)

    def _navigate_to_target(self, target: StableSample) -> bool:
        pose = self._current_pose_in_map(timeout_s=0.20)
        if pose is None:
            raise RuntimeError("map->base_link unavailable before parking goal")
        x, y, _ = pose
        distance = math.hypot(target.x - x, target.y - y)
        if not self.min_goal_distance_m <= distance <= self.max_goal_distance_m:
            raise RuntimeError(
                "parking goal distance %.3fm outside [%.3f, %.3f]"
                % (distance, self.min_goal_distance_m, self.max_goal_distance_m)
            )

        yaw = math.atan2(target.y - y, target.x - x)
        attempts = self.parking_retry_count + 1
        for attempt in range(attempts):
            self._check_cancelled()
            self._send_pose_goal(target.x, target.y, yaw)
            state = self._wait_action_terminal(self.parking_goal_timeout_s, allow_timeout=False)
            if state == GoalStatus.SUCCEEDED:
                return True
            rospy.logwarn(
                "parking move_base attempt %d/%d ended state=%d",
                attempt + 1,
                attempts,
                state,
            )
            self.move_base.cancel_all_goals()
            if attempt + 1 < attempts:
                self._call_empty_service(self.clear_costmaps_service, timeout_s=0.5)
                rospy.sleep(self.retry_delay_s)
        return False

    def _wait_action_terminal(self, timeout_s: float, allow_timeout: bool) -> int:
        deadline = time.monotonic() + timeout_s
        terminal = {
            GoalStatus.PREEMPTED,
            GoalStatus.SUCCEEDED,
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        }
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            state = self.move_base.get_state()
            if state in terminal:
                return state
            rospy.sleep(0.05)
        self.move_base.cancel_all_goals()
        if allow_timeout:
            return GoalStatus.PREEMPTED
        return GoalStatus.LOST

    def _verify_final_parking(self, target: StableSample) -> bool:
        deadline = time.monotonic() + self.final_verify_timeout_s
        stable_since: Optional[float] = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            pose = self._current_pose_in_map(timeout_s=0.05)
            with self.lock:
                odom = self.latest_odom
            if pose is None or odom is None:
                stable_since = None
                rospy.sleep(0.05)
                continue

            x, y, _ = pose
            distance = math.hypot(target.x - x, target.y - y)
            vx = math.hypot(
                float(odom.twist.twist.linear.x),
                float(odom.twist.twist.linear.y),
            )
            wz = abs(float(odom.twist.twist.angular.z))
            good = (
                distance <= self.final_position_tolerance_m
                and vx <= self.final_linear_speed_tolerance_mps
                and wz <= self.final_angular_speed_tolerance_radps
            )
            if good:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= self.final_stable_time_s:
                    return True
            else:
                stable_since = None
            rospy.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    # target geodetic publication
    # ------------------------------------------------------------------
    def _apply_goal_extension(self, target: StableSample) -> StableSample:
        if abs(self.goal_extension_m) < 1.0e-6:
            return target
        pose = self._current_pose_in_map(timeout_s=0.20)
        if pose is None:
            return target
        x, y, _ = pose
        dx = target.x - x
        dy = target.y - y
        distance = math.hypot(dx, dy)
        if distance < 1.0e-6:
            return target
        return StableSample(
            x=target.x + self.goal_extension_m * dx / distance,
            y=target.y + self.goal_extension_m * dy / distance,
            z=target.z,
            confidence=target.confidence,
            depth_m=target.depth_m,
            center_u=target.center_u,
            center_v=target.center_v,
            wall_time=target.wall_time,
            camera_x_right_m=target.camera_x_right_m,
            camera_y_down_m=target.camera_y_down_m,
            camera_z_forward_m=target.camera_z_forward_m,
            east_offset_m=target.east_offset_m,
            north_offset_m=target.north_offset_m,
            vehicle_latitude=target.vehicle_latitude,
            vehicle_longitude=target.vehicle_longitude,
            vehicle_altitude=target.vehicle_altitude,
            heading_deg=target.heading_deg,
            target_latitude=target.target_latitude,
            target_longitude=target.target_longitude,
            target_altitude=target.target_altitude,
        )

    def _publish_target(self, target: StableSample) -> None:
        lat = target.target_latitude
        lon = target.target_longitude
        alt = target.target_altitude

        pose = self._current_pose_in_map(timeout_s=0.20)
        yaw = 0.0
        if pose is not None:
            yaw = math.atan2(target.y - pose[1], target.x - pose[0])

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.goal_frame
        pose_msg.pose.position.x = target.x
        pose_msg.pose.position.y = target.y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = yaw_to_quaternion(yaw)
        self.target_pose_pub.publish(pose_msg)

        fix = NavSatFix()
        fix.header.stamp = pose_msg.header.stamp
        fix.header.frame_id = "wgs84"
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt
        self.target_fix_pub.publish(fix)

        parking_target = ParkingTarget()
        parking_target.header.stamp = pose_msg.header.stamp
        parking_target.header.frame_id = "wgs84"
        parking_target.park = 1
        parking_target.latitude = lat
        parking_target.longitude = lon
        parking_target.altitude = alt
        parking_target.confidence = target.confidence
        parking_target.stable_frames = self.stable_required_frames
        parking_target.camera_x_right_m = target.camera_x_right_m
        parking_target.camera_y_down_m = target.camera_y_down_m
        parking_target.camera_z_forward_m = target.camera_z_forward_m
        parking_target.east_offset_m = target.east_offset_m
        parking_target.north_offset_m = target.north_offset_m
        self.parking_target_pub.publish(parking_target)

        rospy.logwarn(
            "Stable parking_empty target: camera=(right=%.3f,down=%.3f,forward=%.3f), "
            "ENU=(east=%.3f,north=%.3f), WGS84=(%.10f, %.10f), park=1 conf=%.3f",
            target.camera_x_right_m,
            target.camera_y_down_m,
            target.camera_z_forward_m,
            target.east_offset_m,
            target.north_offset_m,
            lat,
            lon,
            target.confidence,
        )

    # ------------------------------------------------------------------
    # DWA helpers
    # ------------------------------------------------------------------
    def _save_dwa_configuration(self) -> None:
        if not self.use_dynamic_reconfigure:
            return
        try:
            self.dwa_client = DynamicReconfigureClient(self.dwa_namespace, timeout=2.0)
            current = self.dwa_client.get_configuration(timeout=2.0)
            changed_keys = set(self.search_dwa) | set(self.parking_dwa)
            self.original_dwa_config = {
                key: current[key] for key in changed_keys if key in current
            }
            rospy.loginfo("Saved %d DWA parameters for later restore", len(self.original_dwa_config))
        except Exception as exc:
            self.dwa_client = None
            self.original_dwa_config = None
            rospy.logwarn("DWA dynamic_reconfigure unavailable: %s", exc)

    def _apply_dwa_profile(self, profile: Dict[str, object], name: str) -> None:
        if not self.use_dynamic_reconfigure or not profile:
            return
        if self.dwa_client is None:
            self._save_dwa_configuration()
        if self.dwa_client is None:
            return
        try:
            current = self.dwa_client.get_configuration(timeout=1.0)
            payload = {key: value for key, value in profile.items() if key in current}
            self.dwa_client.update_configuration(payload)
            rospy.logwarn("Applied DWA %s profile: %s", name, payload)
        except Exception as exc:
            rospy.logwarn("Failed to apply DWA %s profile: %s", name, exc)

    def _restore_dwa(self) -> None:
        if self.dwa_client is None or not self.original_dwa_config:
            return
        try:
            self.dwa_client.update_configuration(self.original_dwa_config)
            rospy.loginfo("Restored original DWA parameters")
        except Exception as exc:
            rospy.logwarn("Failed to restore DWA parameters: %s", exc)
        finally:
            self.original_dwa_config = None
            self.dwa_client = None

    # ------------------------------------------------------------------
    # utility / status
    # ------------------------------------------------------------------
    def _current_pose_in_map(self, timeout_s: float) -> Optional[Tuple[float, float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.goal_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(timeout_s),
            )
            translation = transform.transform.translation
            yaw = quaternion_to_yaw(transform.transform.rotation)
            return float(translation.x), float(translation.y), float(yaw)
        except Exception:
            return None

    def _vehicle_stationary(self, odom: Optional[Odometry]) -> bool:
        if odom is None:
            return False
        vx = math.hypot(
            float(odom.twist.twist.linear.x),
            float(odom.twist.twist.linear.y),
        )
        wz = abs(float(odom.twist.twist.angular.z))
        return vx <= self.stationary_linear_mps and wz <= self.stationary_angular_radps

    def _wait_for_stationary(self, stable_s: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        stable_since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self._check_cancelled()
            with self.lock:
                odom = self.latest_odom
            if self._vehicle_stationary(odom):
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_s:
                    return True
            else:
                stable_since = None
            rospy.sleep(0.05)
        return False

    def _call_trigger_service(self, name: str, timeout_s: float) -> bool:
        try:
            rospy.wait_for_service(name, timeout=timeout_s)
            proxy = rospy.ServiceProxy(name, Trigger)
            response = proxy()
            return bool(response.success)
        except Exception as exc:
            rospy.logwarn("Trigger service %s failed: %s", name, exc)
            return False

    def _call_empty_service(self, name: str, timeout_s: float) -> bool:
        try:
            rospy.wait_for_service(name, timeout=timeout_s)
            proxy = rospy.ServiceProxy(name, Empty)
            proxy()
            return True
        except Exception as exc:
            rospy.logwarn("Empty service %s failed: %s", name, exc)
            return False

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set() or rospy.is_shutdown():
            raise InterruptedError("parking cancelled")

    def _set_state(self, state: str, detail: str) -> None:
        with self.lock:
            self.state = state
            self.detail = detail
        rospy.loginfo("parking state=%s detail=%s", state, detail)
        self._publish_status()

    def _publish_status(self) -> None:
        with self.lock:
            count = len(self.samples)
            target = self.frozen_target
            text = "state=%s stable=%d/%d detail=%s" % (
                self.state,
                count,
                self.stable_required_frames,
                self.detail,
            )
            if target is not None:
                text += " target_map_x=%.3f target_map_y=%.3f confidence=%.3f" % (
                    target.x,
                    target.y,
                    target.confidence,
                )
        self.state_pub.publish(String(data=text))

    def _status_timer_cb(self, _event) -> None:
        self._publish_status()

    def _shutdown(self) -> None:
        self.cancel_event.set()
        try:
            self.move_base.cancel_all_goals()
        except Exception:
            pass
        self._restore_dwa()
        self._stop_owned_process("detector")
        if self.stop_owned_camera_on_shutdown:
            self._stop_owned_process("camera")


def main() -> None:
    rospy.init_node("r300_autonomous_parking", anonymous=False)
    AutonomousParkingNode()
    rospy.spin()


if __name__ == "__main__":
    main()
