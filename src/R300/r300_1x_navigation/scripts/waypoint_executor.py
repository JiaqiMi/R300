#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential GPS waypoint sender for move_base.

This node deliberately does NOT publish cmd_vel and does NOT cancel an active
move_base goal to perform a separate yaw pre-alignment.  For intermediate
waypoint pass-through, it sends the next move_base goal directly so the action
server preempts the previous goal without a cancel-only gap.  The only
autonomous motion authority is therefore:

    waypoint_executor -> move_base / DWA -> scout_base -> base

The target pose orientation is still set to the bearing from the current pose
to the waypoint so that move_base receives a meaningful full pose.  The DWA
configuration may ignore final yaw through yaw_goal_tolerance; while driving,
DWA is solely responsible for turning toward a waypoint that starts beside or
behind the vehicle.
"""

import math
import threading

import actionlib
import rospy
import tf.transformations as tft

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def llh_to_ecef(lat_deg, lon_deg, alt_m):
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


def llh_to_enu(lat_deg, lon_deg, alt_m,
               origin_lat, origin_lon, origin_alt):
    x, y, z = llh_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = llh_to_ecef(origin_lat, origin_lon, origin_alt)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)

    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    east = -sin_lon * dx + cos_lon * dy
    north = (
        -sin_lat * cos_lon * dx
        -sin_lat * sin_lon * dy
        + cos_lat * dz
    )
    up = (
        cos_lat * cos_lon * dx
        + cos_lat * sin_lon * dy
        + sin_lat * dz
    )

    return east, north, up


def yaw_to_quat(yaw):
    return tft.quaternion_from_euler(0.0, 0.0, yaw)


class WaypointExecutor(object):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def __init__(self):
        self.lock = threading.RLock()

        self.waypoints_param = rospy.get_param(
            "~waypoints_param", "/subject1_waypoints/waypoints")
        self.origin_topic = rospy.get_param("~origin_topic", "/one_x/origin")
        self.odom_topic = rospy.get_param(
            "~odom_topic", "/subject1/dwa_odom")
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.goal_frame = rospy.get_param("~goal_frame", "map")
        self.auto_start = rospy.get_param("~auto_start", False)
        self.max_goal_distance_from_origin_m = float(rospy.get_param(
            "~max_goal_distance_from_origin_m", 180.0))

        # Intermediate waypoints may be handed over to the next goal before
        # move_base reports SUCCEEDED.  The last waypoint still uses the normal
        # move_base completion condition.
        self.pass_through_enabled = bool(rospy.get_param(
            "~pass_through_enabled", True))
        self.waypoint_switch_distance_m = max(0.0, float(rospy.get_param(
            "~waypoint_switch_distance_m", 2.0)))
        # A distance-only handover can cut a sharp corner at high speed.  Use
        # the actual odometry feedback as a second gate.  Set <= 0 to disable
        # the speed gate, although a positive limit is strongly recommended.
        self.waypoint_switch_max_speed_mps = float(rospy.get_param(
            "~waypoint_switch_max_speed_mps", 0.60))

        # Optional turn-angle-aware handover.  The turn angle is computed at
        # the current waypoint from the incoming and outgoing route segments:
        # 0 deg means straight, 90 deg means a right-angle turn, and 180 deg
        # means a U-turn.  Each class has its own switch radius and speed gate.
        self.angle_adaptive_switch_enabled = bool(rospy.get_param(
            "~angle_adaptive_switch_enabled", True))
        self.turn_straight_max_deg = max(0.0, float(rospy.get_param(
            "~turn_straight_max_deg", 15.0)))
        self.turn_gentle_max_deg = max(
            self.turn_straight_max_deg, float(rospy.get_param(
                "~turn_gentle_max_deg", 35.0)))
        self.turn_medium_max_deg = max(
            self.turn_gentle_max_deg, float(rospy.get_param(
                "~turn_medium_max_deg", 70.0)))
        self.turn_sharp_max_deg = max(
            self.turn_medium_max_deg, float(rospy.get_param(
                "~turn_sharp_max_deg", 110.0)))
        self.turn_min_segment_length_m = max(0.01, float(rospy.get_param(
            "~turn_min_segment_length_m", 1.0)))
        self.turn_switch_distance_cap_ratio = min(0.49, max(0.05, float(
            rospy.get_param("~turn_switch_distance_cap_ratio", 0.45))))

        self.turn_profiles = {
            "STRAIGHT": {
                "distance_m": max(0.0, float(rospy.get_param(
                    "~straight_switch_distance_m", 4.0))),
                "speed_mps": float(rospy.get_param(
                    "~straight_switch_max_speed_mps", 2.5)),
            },
            "GENTLE": {
                "distance_m": max(0.0, float(rospy.get_param(
                    "~gentle_switch_distance_m", 3.5))),
                "speed_mps": float(rospy.get_param(
                    "~gentle_switch_max_speed_mps", 1.8)),
            },
            "MEDIUM": {
                "distance_m": max(0.0, float(rospy.get_param(
                    "~medium_switch_distance_m", 3.0))),
                "speed_mps": float(rospy.get_param(
                    "~medium_switch_max_speed_mps", 1.2)),
            },
            "SHARP": {
                "distance_m": max(0.0, float(rospy.get_param(
                    "~sharp_switch_distance_m", 2.0))),
                "speed_mps": float(rospy.get_param(
                    "~sharp_switch_max_speed_mps", 0.7)),
            },
        }

        # —— 子目标钳制器(2026-07-30, 室外滚动全局窗配套, 对标 Nav2 GPS 教程+carrot 语义) ——
        # 室外全局代价地图为 40m 滚动窗(launch 室外档), navfn 对窗外目标必报
        # "目标在图外"失败。真航点距车 > clamp_radius 时, 发"车→航点"连线上
        # clamp_radius 处的窗内子目标(胡萝卜), 控制环滚动前移; 真航点入窗即切真目标。
        self.clamp_enabled = bool(rospy.get_param("~goal_clamp_enabled", True))
        self.clamp_radius_m = float(rospy.get_param(
            "~goal_clamp_radius_m", 18.0))   # 必须 < 全局滚动窗半宽(20m), 留 TF/漂移余量
        self.clamp_resend_dist_m = float(rospy.get_param(
            "~goal_clamp_resend_dist_m", 5.0))   # 车每前进 5m 重新钳制一次
        self.clamp_resend_period_s = float(rospy.get_param(
            "~goal_clamp_resend_period_s", 4.0))  # 兜底周期(需配合下面的进展门)
        # 审查P0: 周期重发必须带"有进展"门——卡死时若无条件每 4s 重发, 每个新目标
        # 都会重置 move_base 的 8s patience → 永不 ABORTED → 失败策略与自动脱困
        # 反射双双饿死 = 无限静默停摆。无进展时不重发, 让 patience 正常走完。
        self.clamp_resend_min_progress_m = float(rospy.get_param(
            "~clamp_resend_min_progress_m", 0.5))
        # 目标点障碍投影(carrot 语义): 目标格在致命/内切区时沿连线向车回退到最近可行格
        self.goal_project_enabled = bool(rospy.get_param(
            "~goal_project_enabled", True))
        self.goal_project_max_m = float(rospy.get_param(
            "~goal_project_max_pullback_m", 3.0))
        self.goal_project_cost_thresh = int(rospy.get_param(
            "~goal_project_cost_thresh", 99))  # OccupancyGrid 值: 99=内切 100=致命; 软膨胀(1~98)可作目标
        # 失败策略(对齐 Nav2 WaypointFollower stop_on_failure=false 语义):
        #   retry_then_skip = 重试 retry_limit 次(延迟 retry_delay 给自动脱困倒车留窗) → 跳下一航点
        #   stop            = 原行为, 全任务 FAILED 停摆
        self.on_failure = rospy.get_param("~on_failure", "retry_then_skip")
        self.failure_retry_limit = int(rospy.get_param("~failure_retry_limit", 1))
        self.failure_retry_delay_s = float(rospy.get_param(
            "~failure_retry_delay_s", 8.0))  # 脱困反射: 静止判定~1.2s+倒车3.2s+复核, 8s 够整个反射走完
        self.skip_fuse_limit = int(rospy.get_param(
            "~consecutive_skip_fuse", 99))   # 2026-07-30 3→99(比赛档等效禁用): FAILED=全任务永久
                                             # 停摆, 三个连续航点恰好被障碍群罩住就自杀——比赛哲学:
                                             # 跳过再多点车还在跑, 都比原地宣布失败强。系统性故障
                                             # (TF断)另有 stop_on_stale 硬停兜底, 不靠这根保险丝

        self.origin = None
        self.latest_odom = None
        self.state = self.RUNNING if self.auto_start else self.IDLE
        self.current_index = 0
        self.goal_active = False
        self.last_error = ""
        # Human-readable operator command and state-machine transition.  These
        # are published with waypoint_status so both navigation modes expose
        # the same progress and recent action information.
        self.last_command = "NONE"
        self.last_transition = "INITIALIZED"

        # A goal generation ID makes delayed PREEMPTED callbacks harmless after
        # pause/cancel/skip.  Only the currently active generation may advance
        # or fail the waypoint state machine.
        self.goal_generation = 0
        self.active_goal_generation = None

        # 钳制器/失败策略运行态
        self.goal_is_clamped = False      # 当前活动目标是否为子目标(而非真航点)
        self.last_sent_target = None      # 最近发出的目标位置(rviz 胡萝卜标记用)
        self.clamp_sent_pos = None        # 发出子目标时的车位置(前移判据)
        self.clamp_sent_time = 0.0
        self.fail_count = 0               # 当前真航点累计失败次数
        self.retry_after = 0.0            # 失败重试的最早时刻(给脱困倒车留窗)
        self.consecutive_skips = 0        # 连续跳过熔断(审查P2: 防系统性故障被 COMPLETED 掩盖)
        self.skipped_indices = set()      # 已跳过航点(rviz 红色标记)
        self.latest_costmap = None        # 全局代价地图(目标投影用)

        self.waypoints = self.load_waypoints()

        self.client = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction)

        self.status_pub = rospy.Publisher(
            "/subject1/waypoint_status", String, queue_size=5, latch=True)
        self.current_pose_pub = rospy.Publisher(
            "/subject1/current_waypoint_pose", PoseStamped,
            queue_size=5, latch=True)
        # 航点可视化(rviz MarkerArray): 绿=已到 蓝=当前 灰=待走 红=跳过,
        # 橙球=当前子目标(胡萝卜), 深绿折线=路线
        self.marker_pub = rospy.Publisher(
            "/subject1/waypoint_markers", MarkerArray, queue_size=1, latch=True)

        # 全局代价地图(室外档 always_send_full_costmap=true, 1Hz 全幅), 目标投影用
        rospy.Subscriber(
            "/move_base/global_costmap/costmap", OccupancyGrid,
            self.costmap_cb, queue_size=1)
        rospy.Subscriber(
            self.origin_topic, NavSatFix, self.origin_cb, queue_size=1)
        rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_cb, queue_size=5)

        rospy.Service("/subject1/start_waypoints", Trigger, self.start_cb)
        rospy.Service("/subject1/cancel_waypoints", Trigger, self.cancel_cb)
        rospy.Service("/subject1/pause_waypoints", Trigger, self.pause_cb)
        rospy.Service("/subject1/resume_waypoints", Trigger, self.resume_cb)
        rospy.Service("/subject1/skip_waypoint", Trigger, self.skip_cb)

        # Only sends a goal if none is active; it never publishes cmd_vel.
        rospy.Timer(rospy.Duration(0.05), self.control_timer_cb)
        rospy.Timer(rospy.Duration(1.0), self.status_timer_cb)

        rospy.on_shutdown(self.shutdown_cb)

        rospy.logwarn(
            "waypoint_executor loaded %d waypoint(s), auto_start=%s, "
            "direct_move_base_only=true, pass_through=%s, "
            "angle_adaptive=%s, fixed_switch=(%.2fm, %.2fm/s)",
            len(self.waypoints), str(self.auto_start),
            str(self.pass_through_enabled),
            str(self.angle_adaptive_switch_enabled),
            self.waypoint_switch_distance_m,
            self.waypoint_switch_max_speed_mps)

        if not self.auto_start:
            rospy.logwarn(
                "航点已加载，但不会自动开始。启动命令："
                "rosservice call /subject1/start_waypoints")

    def load_waypoints(self):
        raw = rospy.get_param(self.waypoints_param, None)
        if raw is None:
            raise RuntimeError(
                "找不到航点参数：%s，请检查 subject1_waypoints.yaml 是否加载"
                % self.waypoints_param)
        if not isinstance(raw, list) or len(raw) == 0:
            raise RuntimeError("航点参数不是非空列表：%s" % self.waypoints_param)

        waypoints = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise RuntimeError("第 %d 个航点不是字典格式" % i)

            name = item.get("name", "wp_%02d" % (i + 1))
            lat = item.get("latitude_deg", item.get("latitude", None))
            lon = item.get("longitude_deg", item.get("longitude", None))
            alt = item.get("altitude_m", item.get("altitude", 0.0))

            if lat is None or lon is None:
                raise RuntimeError(
                    "航点 %s 缺少 latitude_deg / longitude_deg" % name)

            lat = float(lat)
            lon = float(lon)
            alt = float(alt)
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                raise RuntimeError(
                    "航点 %s 经纬度非法：lat=%s lon=%s" % (name, lat, lon))

            waypoints.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "enu_ready": False,
                "east": 0.0,
                "north": 0.0,
                "up": 0.0,
            })
        return waypoints

    def origin_cb(self, msg):
        if abs(msg.latitude) > 90.0 or abs(msg.longitude) > 180.0:
            rospy.logwarn_throttle(1.0, "收到非法 /one_x/origin，忽略")
            return

        with self.lock:
            self.origin = (msg.latitude, msg.longitude, msg.altitude)
            for wp in self.waypoints:
                east, north, up = llh_to_enu(
                    wp["lat"], wp["lon"], wp["alt"],
                    self.origin[0], self.origin[1], self.origin[2])
                wp["east"] = east
                wp["north"] = north
                wp["up"] = up
                wp["enu_ready"] = True

    def odom_cb(self, msg):
        with self.lock:
            self.latest_odom = msg

    def costmap_cb(self, msg):
        # 单引用原子替换, 读方拿到的永远是完整一幅
        self.latest_costmap = msg

    def costmap_cost_at(self, x, y):
        """查全局代价地图格值(OccupancyGrid: -1 未知, 0~100)。图外/无图返回 None。
        坐标系说明: goal_frame(map) 与代价地图 frame(odom) 由静态恒等 TF 绑定,
        坐标数值可直接互用(subject1_map_to_odom)。"""
        g = self.latest_costmap
        if g is None:
            return None
        res = g.info.resolution
        mx = int(math.floor((x - g.info.origin.position.x) / res))
        my = int(math.floor((y - g.info.origin.position.y) / res))
        if not (0 <= mx < g.info.width and 0 <= my < g.info.height):
            return None
        return g.data[my * g.info.width + mx]

    def project_goal_off_obstacle(self, tx, ty, rx, ry):
        """目标格落在致命/内切区时, 沿"目标→车"连线回退搜索最近可行格(carrot 语义,
        上限 goal_project_max_m)。返回 None = 无需调整或搜索失败(失败留给
        xy_goal_tolerance 1.5 + latch + 失败策略兜底)。"""
        c = self.costmap_cost_at(tx, ty)
        if c is None or c < self.goal_project_cost_thresh:
            return None
        d = math.hypot(tx - rx, ty - ry)
        if d < 1e-6:
            return None
        ux, uy = (rx - tx) / d, (ry - ty) / d
        # 审查P2: 回退量封顶于"车-目标"距离减容差带——否则近距时投影点会越过车体
        # 落到车后(贴障原地调头), 或落进 1.5m 容差圈被 latch 即刻误判"到达"
        max_pull = min(self.goal_project_max_m, d - 1.7)
        if max_pull <= 0.0:
            return None
        step = 0.1
        for i in range(1, int(max_pull / step) + 1):
            px, py = tx + ux * step * i, ty + uy * step * i
            c = self.costmap_cost_at(px, py)
            if c is not None and 0 <= c < self.goal_project_cost_thresh:
                return (px, py)
        return None

    def invalidate_active_goal(self):
        self.goal_active = False
        self.active_goal_generation = None

    def start_cb(self, req):
        with self.lock:
            if self.state == self.RUNNING:
                return TriggerResponse(False, "航点执行器已经在运行")

            if self.state in [self.COMPLETED, self.FAILED]:
                self.current_index = 0
                self.last_error = ""
                self.skipped_indices.clear()

            self.invalidate_active_goal()
            self.fail_count = 0
            self.retry_after = 0.0
            self.consecutive_skips = 0
            self.goal_is_clamped = False
            self.state = self.RUNNING
            self.last_command = "START"
            self.last_transition = "STARTED"
            rospy.logwarn("收到 /subject1/start_waypoints，开始执行航点")
            self.publish_status()
            return TriggerResponse(True, "开始执行航点")

    def cancel_cb(self, req):
        with self.lock:
            self.invalidate_active_goal()
            self.client.cancel_all_goals()
            self.state = self.IDLE
            self.current_index = 0
            self.last_error = ""
            self.skipped_indices.clear()
            self.goal_is_clamped = False
            self.fail_count = 0
            self.retry_after = 0.0
            self.consecutive_skips = 0
            self.last_command = "CANCEL"
            self.last_transition = "CANCELLED"
            self.publish_status()
            rospy.logwarn("已取消航点任务，并重置到第 1 个航点")
            return TriggerResponse(True, "已取消航点任务")

    def pause_cb(self, req):
        with self.lock:
            if self.state != self.RUNNING:
                return TriggerResponse(False, "当前不是运行状态，无法暂停")

            self.invalidate_active_goal()
            self.client.cancel_all_goals()
            self.goal_is_clamped = False
            self.state = self.PAUSED
            self.last_command = "PAUSE"
            self.last_transition = "PAUSED"
            self.publish_status()
            rospy.logwarn("航点任务已暂停")
            return TriggerResponse(True, "航点任务已暂停")

    def resume_cb(self, req):
        with self.lock:
            if self.state != self.PAUSED:
                return TriggerResponse(False, "当前不是 PAUSED 状态，无法恢复")

            self.invalidate_active_goal()
            self.state = self.RUNNING
            self.last_command = "RESUME"
            self.last_transition = "RESUMED"
            rospy.logwarn("航点任务已恢复")
            self.publish_status()
            return TriggerResponse(True, "航点任务已恢复")

    def skip_cb(self, req):
        with self.lock:
            if self.current_index + 1 >= len(self.waypoints):
                return TriggerResponse(False, "没有下一个航点")

            self.invalidate_active_goal()
            self.client.cancel_all_goals()
            self.current_index += 1
            self.fail_count = 0
            self.retry_after = 0.0
            self.goal_is_clamped = False
            self.state = self.RUNNING
            self.last_command = "SKIP"
            self.last_transition = "SKIPPED_TO_NEXT"
            rospy.logwarn("跳过到航点 %d/%d", self.current_index + 1,
                          len(self.waypoints))
            self.publish_status()
            return TriggerResponse(True, "已跳到下一个航点")

    def current_horizontal_speed(self):
        """Return measured horizontal speed from the latest odometry."""
        if self.latest_odom is None:
            return None

        # R300 is a differential/non-holonomic platform.  The DWA odometry
        # adapter forces lateral velocity to zero, so use the absolute
        # longitudinal speed as the pass-through gate.
        vx = float(self.latest_odom.twist.twist.linear.x)
        if not math.isfinite(vx):
            return None
        return abs(vx)

    def current_turn_profile(self):
        """Return the angle class and active pass-through limits.

        The incoming segment uses the previous route waypoint when available.
        For the first waypoint it uses the current vehicle position.  The
        outgoing segment always points from the current waypoint to the next
        waypoint.  A short/degenerate segment disables pass-through so an
        ambiguous waypoint cannot be skipped at speed.
        """
        if self.current_index + 1 >= len(self.waypoints):
            return None

        current_wp = self.waypoints[self.current_index]
        next_wp = self.waypoints[self.current_index + 1]
        if not current_wp["enu_ready"] or not next_wp["enu_ready"]:
            return None

        if self.current_index > 0:
            previous_wp = self.waypoints[self.current_index - 1]
            if not previous_wp["enu_ready"]:
                return None
            incoming_x = current_wp["east"] - previous_wp["east"]
            incoming_y = current_wp["north"] - previous_wp["north"]
        else:
            if self.latest_odom is None:
                return None
            incoming_x = (
                current_wp["east"] - self.latest_odom.pose.pose.position.x)
            incoming_y = (
                current_wp["north"] - self.latest_odom.pose.pose.position.y)

        outgoing_x = next_wp["east"] - current_wp["east"]
        outgoing_y = next_wp["north"] - current_wp["north"]
        incoming_length = math.hypot(incoming_x, incoming_y)
        outgoing_length = math.hypot(outgoing_x, outgoing_y)

        if (incoming_length < self.turn_min_segment_length_m or
                outgoing_length < self.turn_min_segment_length_m):
            return {
                "class": "SHORT_SEGMENT",
                "angle_deg": float("nan"),
                "signed_angle_deg": float("nan"),
                "distance_m": 0.0,
                "speed_mps": 0.0,
                "allow_pass": False,
                "incoming_length_m": incoming_length,
                "outgoing_length_m": outgoing_length,
            }

        dot = incoming_x * outgoing_x + incoming_y * outgoing_y
        cross = incoming_x * outgoing_y - incoming_y * outgoing_x
        signed_angle_deg = math.degrees(math.atan2(cross, dot))
        angle_deg = abs(signed_angle_deg)

        if not self.angle_adaptive_switch_enabled:
            turn_class = "FIXED"
            distance_m = self.waypoint_switch_distance_m
            speed_mps = self.waypoint_switch_max_speed_mps
            allow_pass = True
        elif angle_deg <= self.turn_straight_max_deg:
            turn_class = "STRAIGHT"
            distance_m = self.turn_profiles[turn_class]["distance_m"]
            speed_mps = self.turn_profiles[turn_class]["speed_mps"]
            allow_pass = True
        elif angle_deg <= self.turn_gentle_max_deg:
            turn_class = "GENTLE"
            distance_m = self.turn_profiles[turn_class]["distance_m"]
            speed_mps = self.turn_profiles[turn_class]["speed_mps"]
            allow_pass = True
        elif angle_deg <= self.turn_medium_max_deg:
            turn_class = "MEDIUM"
            distance_m = self.turn_profiles[turn_class]["distance_m"]
            speed_mps = self.turn_profiles[turn_class]["speed_mps"]
            allow_pass = True
        elif angle_deg <= self.turn_sharp_max_deg:
            turn_class = "SHARP"
            distance_m = self.turn_profiles[turn_class]["distance_m"]
            speed_mps = self.turn_profiles[turn_class]["speed_mps"]
            allow_pass = True
        else:
            # A near reversal is intentionally treated as a normal stop goal.
            # Waiting for SUCCEEDED is safer than handing a U-turn to DWA while
            # the vehicle is still moving through the waypoint.
            turn_class = "UTURN"
            distance_m = 0.0
            speed_mps = 0.0
            allow_pass = False

        if allow_pass:
            distance_cap = (
                self.turn_switch_distance_cap_ratio *
                min(incoming_length, outgoing_length))
            distance_m = min(distance_m, distance_cap)

        return {
            "class": turn_class,
            "angle_deg": angle_deg,
            "signed_angle_deg": signed_angle_deg,
            "distance_m": distance_m,
            "speed_mps": speed_mps,
            "allow_pass": allow_pass,
            "incoming_length_m": incoming_length,
            "outgoing_length_m": outgoing_length,
        }

    def current_target_geometry(self):
        if self.latest_odom is None or self.current_index >= len(self.waypoints):
            return None

        wp = self.waypoints[self.current_index]
        if not wp["enu_ready"]:
            return None

        cur_x = self.latest_odom.pose.pose.position.x
        cur_y = self.latest_odom.pose.pose.position.y
        dx = wp["east"] - cur_x
        dy = wp["north"] - cur_y
        distance = math.hypot(dx, dy)

        q = self.latest_odom.pose.pose.orientation
        current_yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        # At a nearly coincident target keep the current orientation instead of
        # sending an arbitrary atan2(0, 0) orientation.
        target_yaw = current_yaw if distance < 1.0e-3 else math.atan2(dy, dx)
        return target_yaw, current_yaw, distance

    def send_current_goal(self):
        if not self.client.wait_for_server(rospy.Duration(0.05)):
            rospy.loginfo_throttle(
                2.0, "等待 move_base action server：%s", self.move_base_action)
            return

        if self.current_index >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_index]
        if not wp["enu_ready"]:
            self.state = self.FAILED
            self.last_transition = "FAILED"
            self.last_error = "航点 ENU 未就绪，缺少 /one_x/origin"
            rospy.logerr(self.last_error)
            self.publish_status()
            return

        dist_origin = math.hypot(wp["east"], wp["north"])
        if dist_origin > self.max_goal_distance_from_origin_m:
            self.state = self.FAILED
            self.last_transition = "FAILED"
            self.last_error = (
                "拒绝航点 %s：距离原点 %.2f m 超过限制 %.2f m" % (
                    wp["name"], dist_origin,
                    self.max_goal_distance_from_origin_m))
            rospy.logerr(self.last_error)
            self.publish_status()
            return

        geometry = self.current_target_geometry()
        if geometry is None:
            rospy.loginfo_throttle(2.0, "等待当前 odom 后发送航点")
            return

        target_yaw, _, distance = geometry

        # —— 子目标钳制: 真航点在滚动全局窗外时发连线上的窗内子目标(胡萝卜) ——
        rx = self.latest_odom.pose.pose.position.x
        ry = self.latest_odom.pose.pose.position.y
        tx, ty = wp["east"], wp["north"]
        clamped = False
        if self.clamp_enabled and distance > self.clamp_radius_m:
            s = self.clamp_radius_m / distance
            tx = rx + (wp["east"] - rx) * s
            ty = ry + (wp["north"] - ry) * s
            clamped = True

        # —— 目标点障碍投影: 目标格在致命/内切区时沿连线向车回退到最近可行格 ——
        if self.goal_project_enabled:
            adj = self.project_goal_off_obstacle(tx, ty, rx, ry)
            if adj is not None:
                rospy.logwarn(
                    "航点 %s 目标点落在致命/内切区, 沿线回退 %.2fm 投影到可行格",
                    wp["name"], math.hypot(tx - adj[0], ty - adj[1]))
                tx, ty = adj

        q = yaw_to_quat(target_yaw)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = tx
        goal.target_pose.pose.position.y = ty
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.current_pose_pub.publish(goal.target_pose)

        self.goal_generation += 1
        generation = self.goal_generation
        self.active_goal_generation = generation
        self.goal_active = True
        self.goal_is_clamped = clamped
        self.last_sent_target = (tx, ty)
        self.clamp_sent_pos = (rx, ry)
        self.clamp_sent_time = rospy.get_time()
        self.last_transition = "SUBGOAL_SENT" if clamped else "GOAL_SENT"

        self.client.send_goal(
            goal,
            done_cb=lambda status, result, seq=generation:
                self.done_cb(status, result, seq))

        rospy.logwarn(
            "%s交给 move_base：航点 %d/%d [%s] 目标(%.3f, %.3f) "
            "真航点距离=%.3f target_yaw=%.1fdeg",
            ("子目标(钳制%.0fm)" % self.clamp_radius_m) if clamped else "直接",
            self.current_index + 1, len(self.waypoints), wp["name"],
            tx, ty, distance, math.degrees(target_yaw))
        self.publish_status()

    def control_timer_cb(self, event):
        with self.lock:
            if self.state != self.RUNNING:
                return

            if self.origin is None or self.latest_odom is None:
                rospy.loginfo_throttle(
                    2.0, "等待 /one_x/origin 和 /one_x/odom")
                return

            if self.current_index >= len(self.waypoints):
                self.state = self.COMPLETED
                self.invalidate_active_goal()
                self.last_transition = "COMPLETED"
                rospy.logwarn("全部航点已完成")
                self.publish_status()
                return

            # —— 子目标滚动前移: 车前进≥阈值或超时 → 重新钳制(复用"发新目标抢占
            # 旧目标"的直通机制); 真航点入窗后 send_current_goal 自动切回真目标 ——
            if (self.goal_active and self.goal_is_clamped
                    and self.latest_odom is not None
                    and self.clamp_sent_pos is not None):
                rx = self.latest_odom.pose.pose.position.x
                ry = self.latest_odom.pose.pose.position.y
                moved = math.hypot(rx - self.clamp_sent_pos[0],
                                   ry - self.clamp_sent_pos[1])
                elapsed = rospy.get_time() - self.clamp_sent_time
                if (moved >= self.clamp_resend_dist_m or
                        (elapsed >= self.clamp_resend_period_s
                         and moved >= self.clamp_resend_min_progress_m)):
                    self.invalidate_active_goal()
                    self.send_current_goal()
                    return

            # Intermediate waypoints use a route-angle-aware handover profile.
            # Straight points may switch early at high speed; sharper points use
            # a smaller radius and lower speed.  U-turn-like points wait for the
            # normal move_base SUCCEEDED result.
            if (self.pass_through_enabled and self.goal_active and
                    self.current_index + 1 < len(self.waypoints)):
                geometry = self.current_target_geometry()
                speed_mps = self.current_horizontal_speed()
                turn_profile = self.current_turn_profile()
                if (geometry is not None and speed_mps is not None and
                        turn_profile is not None):
                    if not turn_profile["allow_pass"]:
                        rospy.loginfo_throttle(
                            1.0,
                            "当前航点不允许提前切换：class=%s angle=%s，"
                            "等待 move_base SUCCEEDED",
                            turn_profile["class"],
                            ("nan" if not math.isfinite(
                                turn_profile["angle_deg"]) else
                             "%.1fdeg" % turn_profile["angle_deg"]))
                    else:
                        switch_distance_m = turn_profile["distance_m"]
                        switch_speed_mps = turn_profile["speed_mps"]
                        distance_ok = geometry[2] <= switch_distance_m
                        speed_gate_disabled = switch_speed_mps <= 0.0
                        speed_ok = (
                            speed_gate_disabled or
                            speed_mps <= switch_speed_mps)

                        if distance_ok and speed_ok:
                            old_index = self.current_index
                            old_name = self.waypoints[old_index]["name"]

                            # Invalidate the old callback generation first,
                            # but do NOT explicitly cancel the active goal.
                            # Sending the next goal on the same action client
                            # lets move_base preempt the old goal and accept the
                            # new one without a cancel-only interval.
                            self.invalidate_active_goal()
                            self.current_index += 1
                            self.fail_count = 0   # 审查P1: 失败计数按航点复位, 不跨航点泄漏
                            self.retry_after = 0.0
                            self.last_command = "AUTO_PASS"
                            self.last_transition = "PASSED_THROUGH"

                            rospy.logwarn(
                                "按转角提前切换航点 %d/%d [%s]："
                                "class=%s angle=%.1fdeg，"
                                "distance=%.3fm <= %.3fm，"
                                "speed=%.3fm/s <= %.3fm/s，"
                                "下一目标为 %d/%d [%s]",
                                old_index + 1, len(self.waypoints), old_name,
                                turn_profile["class"],
                                turn_profile["angle_deg"],
                                geometry[2], switch_distance_m,
                                speed_mps, switch_speed_mps,
                                self.current_index + 1, len(self.waypoints),
                                self.waypoints[self.current_index]["name"])
                            # Publish the pass-through event, then send the
                            # next waypoint in this same control cycle.  Do not
                            # wait for the next 50 ms timer tick.
                            self.publish_status()
                            self.send_current_goal()
                            return

                        if distance_ok and not speed_ok:
                            rospy.loginfo_throttle(
                                0.5,
                                "已进入自适应切换范围但速度仍高："
                                "class=%s angle=%.1fdeg，"
                                "distance=%.3fm <= %.3fm，"
                                "speed=%.3fm/s > %.3fm/s",
                                turn_profile["class"],
                                turn_profile["angle_deg"],
                                geometry[2], switch_distance_m,
                                speed_mps, switch_speed_mps)

            if not self.goal_active:
                # 失败重试延迟窗: 给自动脱困反射(静止判定+倒车0.4m)留完整时间
                if rospy.get_time() < self.retry_after:
                    return
                self.send_current_goal()

    def done_cb(self, status, result, generation):
        with self.lock:
            if generation != self.active_goal_generation:
                rospy.logdebug("忽略已过期 move_base 回调，generation=%d", generation)
                return

            self.invalidate_active_goal()

            if self.state != self.RUNNING:
                return

            wp_name = (
                self.waypoints[self.current_index]["name"]
                if self.current_index < len(self.waypoints) else "unknown")

            if status == GoalStatus.SUCCEEDED:
                if self.goal_is_clamped:
                    # 子目标到达 ≠ 真航点到达: 索引不前进, 控制环 50ms 内重新钳制续跑
                    # (不在 action 回调线程里直接 send_goal, 避免重入)
                    self.goal_is_clamped = False
                    rospy.loginfo(
                        "子目标到达, 将重新钳制续向真航点 %d/%d [%s]",
                        self.current_index + 1, len(self.waypoints), wp_name)
                    return
                rospy.logwarn("到达航点 %d/%d [%s]", self.current_index + 1,
                              len(self.waypoints), wp_name)
                self.current_index += 1
                self.fail_count = 0
                self.consecutive_skips = 0
                self.last_transition = "WAYPOINT_REACHED"
                if self.current_index >= len(self.waypoints):
                    self.state = self.COMPLETED
                    self.last_transition = "COMPLETED"
                    rospy.logwarn("全部航点完成(其中跳过 %d 个)",
                                  len(self.skipped_indices))
                self.publish_status()
                return

            if status == GoalStatus.PREEMPTED:
                # 有效 generation 下的 PREEMPTED = 外部客户端抢占(如 rviz 手发目标;
                # 本节点自己的抢占已被 generation 守卫过滤)。不计失败, 自动暂停任务
                # 把控制权交给操作员(审查P2), 恢复: rosservice call /subject1/resume_waypoints
                self.goal_is_clamped = False
                self.state = self.PAUSED
                self.last_transition = "PREEMPTED_EXTERNAL_PAUSED"
                rospy.logwarn(
                    "航点目标被外部客户端抢占(rviz 手发目标?), 任务自动暂停; "
                    "恢复: rosservice call /subject1/resume_waypoints")
                self.publish_status()
                return

            # —— 失败策略(2026-07-30): 单个航点失败不再拖死全任务 ——
            self.goal_is_clamped = False
            if self.on_failure == "retry_then_skip":
                self.fail_count += 1
                if self.fail_count <= self.failure_retry_limit:
                    self.retry_after = (
                        rospy.get_time() + self.failure_retry_delay_s)
                    self.last_transition = "FAILED_RETRY"
                    rospy.logwarn(
                        "move_base 在航点 %d/%d [%s] 失败(status=%d), "
                        "%.0fs 后重试 %d/%d(给自动脱困倒车留时间窗)",
                        self.current_index + 1, len(self.waypoints), wp_name,
                        status, self.failure_retry_delay_s,
                        self.fail_count, self.failure_retry_limit)
                    self.publish_status()
                    return
                self.skipped_indices.add(self.current_index)
                rospy.logerr(
                    "航点 %d/%d [%s] 连续失败 %d 次 → 跳过, 继续下一航点"
                    "(比赛语义: 任务连续性优先; on_failure:=stop 可回旧行为)",
                    self.current_index + 1, len(self.waypoints), wp_name,
                    self.fail_count)
                self.current_index += 1
                self.fail_count = 0
                # 审查P1: 跳过后与重试同宽的延迟窗——车多半仍被困, 立即发下一目标
                # 会掐灭本次 ABORTED 的脱困资格并与倒车抢 cmd_vel
                self.retry_after = (
                    rospy.get_time() + self.failure_retry_delay_s)
                self.consecutive_skips += 1
                self.last_transition = "FAILED_SKIPPED"
                if self.consecutive_skips >= self.skip_fuse_limit:
                    self.state = self.FAILED
                    self.last_transition = "FAILED"
                    self.last_error = (
                        "连续跳过 %d 个航点, 疑似系统性故障(TF断/定位失效/被围死), "
                        "停止任务待人工处理" % self.consecutive_skips)
                    rospy.logerr(self.last_error)
                    self.publish_status()
                    return
                if self.current_index >= len(self.waypoints):
                    self.state = self.COMPLETED
                    self.last_transition = "COMPLETED"
                    rospy.logwarn("全部航点完成(其中跳过 %d 个)",
                                  len(self.skipped_indices))
                self.publish_status()
                return

            self.state = self.FAILED
            self.last_transition = "FAILED"
            self.last_error = (
                "move_base 在航点 %d/%d [%s] 失败，status=%d" % (
                    self.current_index + 1, len(self.waypoints), wp_name,
                    status))
            rospy.logerr(self.last_error)
            self.publish_status()

    def status_timer_cb(self, event):
        with self.lock:
            self.publish_status()
            self.publish_waypoint_markers()

    def publish_waypoint_markers(self):
        """rviz 航点可视化: 绿=已到 蓝=当前 灰=待走 红=跳过, 橙球=当前子目标
        (胡萝卜), 深绿折线=航点路线, 文字=序号:名称。frame=goal_frame(map,
        与 odom 静态恒等 TF 绑定, subject1_map_to_odom)。"""
        now = rospy.Time.now()
        arr = MarkerArray()

        line = Marker()
        line.header.frame_id = self.goal_frame
        line.header.stamp = now
        line.ns = "route"
        line.id = 9998
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        line.color.r, line.color.g = 0.1, 0.5
        line.color.b, line.color.a = 0.1, 0.6
        line.pose.orientation.w = 1.0

        for i, wp in enumerate(self.waypoints):
            if not wp["enu_ready"]:
                continue
            line.points.append(Point(x=wp["east"], y=wp["north"], z=0.05))

            m = Marker()
            m.header.frame_id = self.goal_frame
            m.header.stamp = now
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wp["east"]
            m.pose.position.y = wp["north"]
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.5
            if i in self.skipped_indices:
                rgba = (1.0, 0.15, 0.15, 0.9)   # 红=已跳过
            elif i < self.current_index:
                rgba = (0.15, 0.9, 0.15, 0.9)   # 绿=已到达
            elif i == self.current_index:
                rgba = (0.2, 0.4, 1.0, 0.95)    # 蓝=当前目标
            else:
                rgba = (0.7, 0.7, 0.7, 0.7)     # 灰=待执行
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            arr.markers.append(m)

            t = Marker()
            t.header.frame_id = self.goal_frame
            t.header.stamp = now
            t.ns = "labels"
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = wp["east"]
            t.pose.position.y = wp["north"]
            t.pose.position.z = 0.9
            t.scale.z = 0.45
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 0.9
            t.text = "%d:%s" % (i + 1, wp["name"])
            arr.markers.append(t)

        if line.points:
            arr.markers.append(line)

        carrot = Marker()
        carrot.header.frame_id = self.goal_frame
        carrot.header.stamp = now
        carrot.ns = "carrot"
        carrot.id = 9999
        if (self.state == self.RUNNING and self.goal_is_clamped
                and self.last_sent_target is not None):
            carrot.type = Marker.SPHERE
            carrot.action = Marker.ADD
            carrot.pose.position.x = self.last_sent_target[0]
            carrot.pose.position.y = self.last_sent_target[1]
            carrot.pose.position.z = 0.3
            carrot.pose.orientation.w = 1.0
            carrot.scale.x = carrot.scale.y = carrot.scale.z = 0.7
            carrot.color.r, carrot.color.g = 1.0, 0.55
            carrot.color.b, carrot.color.a = 0.0, 0.95
        else:
            carrot.action = Marker.DELETE
        arr.markers.append(carrot)

        self.marker_pub.publish(arr)

    def publish_status(self):
        total = len(self.waypoints)
        human_index = min(self.current_index + 1, total) if total > 0 else 0
        progress = "%d/%d" % (human_index, total)

        speed_mps = self.current_horizontal_speed()
        speed_text = "nan" if speed_mps is None else "%.3f" % speed_mps
        turn_profile = self.current_turn_profile()
        if turn_profile is None:
            turn_class = "NONE"
            turn_angle_text = "nan"
            active_switch_distance_m = self.waypoint_switch_distance_m
            active_switch_speed_mps = self.waypoint_switch_max_speed_mps
        else:
            turn_class = turn_profile["class"]
            turn_angle_text = (
                "nan" if not math.isfinite(turn_profile["angle_deg"]) else
                "%.1f" % turn_profile["angle_deg"])
            active_switch_distance_m = turn_profile["distance_m"]
            active_switch_speed_mps = turn_profile["speed_mps"]

        if self.current_index < total:
            wp = self.waypoints[self.current_index]
            text = (
                "state=%s progress=%s index=%d total=%d current=%s "
                "lat=%.10f lon=%.10f east=%.3f north=%.3f "
                "speed_mps=%s turn_class=%s turn_angle_deg=%s "
                "switch_distance_m=%.3f switch_max_speed_mps=%.3f "
                "goal_active=%s last_command=%s transition=%s "
                "mode=direct_move_base_only error=%s" % (
                    self.state, progress, self.current_index, total,
                    wp["name"], wp["lat"], wp["lon"], wp["east"],
                    wp["north"], speed_text, turn_class, turn_angle_text,
                    active_switch_distance_m, active_switch_speed_mps,
                    str(self.goal_active), self.last_command,
                    self.last_transition, self.last_error))
        else:
            text = (
                "state=%s progress=%s index=%d total=%d goal_active=%s "
                "last_command=%s transition=%s "
                "mode=direct_move_base_only error=%s" % (
                    self.state, progress, self.current_index, total,
                    str(self.goal_active), self.last_command,
                    self.last_transition, self.last_error))
        self.status_pub.publish(String(data=text))

    def shutdown_cb(self):
        try:
            self.invalidate_active_goal()
            self.client.cancel_all_goals()
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("waypoint_executor")
    try:
        node = WaypointExecutor()
        rospy.spin()
    except Exception as e:
        rospy.logfatal("waypoint_executor 启动失败：%s", str(e))
        raise