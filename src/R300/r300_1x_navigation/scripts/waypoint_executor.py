#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential GPS waypoint sender for move_base.

This node deliberately does NOT publish cmd_vel and does NOT cancel an active
move_base goal to perform a separate yaw pre-alignment.  The only autonomous
motion authority is therefore:

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
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


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
        self.odom_topic = rospy.get_param("~odom_topic", "/one_x/odom")
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.goal_frame = rospy.get_param("~goal_frame", "map")
        self.auto_start = rospy.get_param("~auto_start", False)
        self.max_goal_distance_from_origin_m = float(rospy.get_param(
            "~max_goal_distance_from_origin_m", 180.0))

        self.origin = None
        self.latest_odom = None
        self.state = self.RUNNING if self.auto_start else self.IDLE
        self.current_index = 0
        self.goal_active = False
        self.last_error = ""

        # A goal generation ID makes delayed PREEMPTED callbacks harmless after
        # pause/cancel/skip.  Only the currently active generation may advance
        # or fail the waypoint state machine.
        self.goal_generation = 0
        self.active_goal_generation = None

        self.waypoints = self.load_waypoints()

        self.client = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction)

        self.status_pub = rospy.Publisher(
            "/subject1/waypoint_status", String, queue_size=5, latch=True)
        self.current_pose_pub = rospy.Publisher(
            "/subject1/current_waypoint_pose", PoseStamped,
            queue_size=5, latch=True)

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
            "direct_move_base_only=true",
            len(self.waypoints), str(self.auto_start))

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

            self.invalidate_active_goal()
            self.state = self.RUNNING
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
            self.publish_status()
            rospy.logwarn("已取消航点任务，并重置到第 1 个航点")
            return TriggerResponse(True, "已取消航点任务")

    def pause_cb(self, req):
        with self.lock:
            if self.state != self.RUNNING:
                return TriggerResponse(False, "当前不是运行状态，无法暂停")

            self.invalidate_active_goal()
            self.client.cancel_all_goals()
            self.state = self.PAUSED
            self.publish_status()
            rospy.logwarn("航点任务已暂停")
            return TriggerResponse(True, "航点任务已暂停")

    def resume_cb(self, req):
        with self.lock:
            if self.state != self.PAUSED:
                return TriggerResponse(False, "当前不是 PAUSED 状态，无法恢复")

            self.invalidate_active_goal()
            self.state = self.RUNNING
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
            self.state = self.RUNNING
            rospy.logwarn("跳过到航点 %d/%d", self.current_index + 1,
                          len(self.waypoints))
            self.publish_status()
            return TriggerResponse(True, "已跳到下一个航点")

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
            self.last_error = "航点 ENU 未就绪，缺少 /one_x/origin"
            rospy.logerr(self.last_error)
            self.publish_status()
            return

        dist_origin = math.hypot(wp["east"], wp["north"])
        if dist_origin > self.max_goal_distance_from_origin_m:
            self.state = self.FAILED
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
        q = yaw_to_quat(target_yaw)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = wp["east"]
        goal.target_pose.pose.position.y = wp["north"]
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

        self.client.send_goal(
            goal,
            done_cb=lambda status, result, seq=generation:
                self.done_cb(status, result, seq))

        rospy.logwarn(
            "直接交给 move_base：航点 %d/%d [%s] east=%.3f north=%.3f "
            "distance=%.3f target_yaw=%.1fdeg",
            self.current_index + 1, len(self.waypoints), wp["name"],
            wp["east"], wp["north"], distance, math.degrees(target_yaw))
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
                rospy.logwarn("全部航点已完成")
                self.publish_status()
                return

            # No pre-alignment or mid-course goal cancellation is allowed here.
            # DWA keeps uninterrupted ownership of cmd_vel until move_base reports
            # this waypoint as succeeded or failed.
            if not self.goal_active:
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
                rospy.logwarn("到达航点 %d/%d [%s]", self.current_index + 1,
                              len(self.waypoints), wp_name)
                self.current_index += 1
                if self.current_index >= len(self.waypoints):
                    self.state = self.COMPLETED
                    rospy.logwarn("全部航点完成")
                self.publish_status()
                return

            self.state = self.FAILED
            self.last_error = (
                "move_base 在航点 %d/%d [%s] 失败，status=%d" % (
                    self.current_index + 1, len(self.waypoints), wp_name,
                    status))
            rospy.logerr(self.last_error)
            self.publish_status()

    def status_timer_cb(self, event):
        with self.lock:
            self.publish_status()

    def publish_status(self):
        if self.current_index < len(self.waypoints):
            wp = self.waypoints[self.current_index]
            text = (
                "state=%s index=%d total=%d current=%s "
                "lat=%.10f lon=%.10f east=%.3f north=%.3f "
                "goal_active=%s mode=direct_move_base_only error=%s" % (
                    self.state, self.current_index, len(self.waypoints),
                    wp["name"], wp["lat"], wp["lon"], wp["east"],
                    wp["north"], str(self.goal_active), self.last_error))
        else:
            text = (
                "state=%s index=%d total=%d goal_active=%s "
                "mode=direct_move_base_only error=%s" % (
                    self.state, self.current_index, len(self.waypoints),
                    str(self.goal_active), self.last_error))
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
