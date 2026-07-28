#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual road-sign guidance for lidar navigation.

The node never publishes cmd_vel.  After a stable LEFT/RIGHT detection it:
1. pauses waypoint_executor;
2. sends one temporary move_base goal in base_link;
3. resumes the original unfinished GPS waypoint after the temporary goal.
"""

import math
import threading

import actionlib
import rospy
import tf.transformations as tft
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from r300_vision_msgs.msg import DetectedObjectArray
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


class DirectionSignLocalGoal:
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    def __init__(self):
        self.lock = threading.RLock()

        self.detections_topic = rospy.get_param(
            "~detections_topic", "/r300_vision/detections")
        self.goal_frame = rospy.get_param("~goal_frame", "base_link")
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.pause_service = rospy.get_param(
            "~pause_service", "/subject1/pause_waypoints")
        self.resume_service = rospy.get_param(
            "~resume_service", "/subject1/resume_waypoints")
        self.waypoint_status_topic = rospy.get_param(
            "~waypoint_status_topic", "/subject1/waypoint_status")

        self.left_classes = self._class_set(rospy.get_param(
            "~left_classes", ["chevro_left", "chevron_left"]))
        self.right_classes = self._class_set(rospy.get_param(
            "~right_classes", ["chevron_right", "chevro_right"]))
        if self.left_classes & self.right_classes:
            raise ValueError("left_classes 与 right_classes 不能重叠")

        self.min_confidence = float(rospy.get_param("~min_confidence", 0.75))
        self.confirm_frames = max(1, int(rospy.get_param("~confirm_frames", 3)))
        self.confirm_max_gap_s = max(
            0.05, float(rospy.get_param("~confirm_max_gap_s", 0.60)))
        self.max_detection_age_s = max(
            0.05, float(rospy.get_param("~max_detection_age_s", 0.80)))
        self.min_bbox_width_px = max(
            1, int(rospy.get_param("~min_bbox_width_px", 30)))
        self.min_bbox_height_px = max(
            1, int(rospy.get_param("~min_bbox_height_px", 20)))
        self.require_valid_depth = bool(
            rospy.get_param("~require_valid_depth", False))
        self.min_distance_m = max(
            0.0, float(rospy.get_param("~min_trigger_distance_m", 0.80)))
        self.max_distance_m = max(
            self.min_distance_m,
            float(rospy.get_param("~max_trigger_distance_m", 8.00)))

        self.goal_distance_m = max(
            0.5, float(rospy.get_param("~turn_goal_distance_m", 4.00)))
        self.goal_angle_deg = min(
            85.0, max(5.0, float(rospy.get_param("~turn_angle_deg", 60.0))))
        self.goal_timeout_s = max(
            1.0, float(rospy.get_param("~local_goal_timeout_s", 25.0)))
        self.pause_settle_s = max(
            0.0, float(rospy.get_param("~pause_settle_s", 0.20)))
        self.service_timeout_s = max(
            0.5, float(rospy.get_param("~service_timeout_s", 5.0)))
        self.action_timeout_s = max(
            0.5, float(rospy.get_param("~action_server_timeout_s", 8.0)))
        self.single_use = bool(rospy.get_param("~single_use", True))
        self.only_when_running = bool(
            rospy.get_param("~trigger_only_when_waypoints_running", True))

        self.waypoint_state = "UNKNOWN"
        self.state = "WAITING"
        self.candidate_direction = None
        self.candidate_count = 0
        self.last_candidate_time = rospy.Time(0)
        self.selected_direction = "NONE"
        self.selected_class = "NONE"
        self.selected_confidence = 0.0
        self.goal_x = float("nan")
        self.goal_y = float("nan")
        self.executing = False
        self.used = False
        self.error = ""

        self.move_base = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction)
        self.pause_proxy = rospy.ServiceProxy(self.pause_service, Trigger)
        self.resume_proxy = rospy.ServiceProxy(self.resume_service, Trigger)

        self.state_pub = rospy.Publisher(
            "/subject1/direction_sign/state", String, queue_size=5, latch=True)
        self.direction_pub = rospy.Publisher(
            "/subject1/direction_sign/selected_direction",
            String, queue_size=2, latch=True)
        self.goal_pub = rospy.Publisher(
            "/subject1/direction_sign/local_goal",
            PoseStamped, queue_size=2, latch=True)

        rospy.Subscriber(
            self.waypoint_status_topic, String, self.waypoint_status_cb,
            queue_size=5)
        rospy.Subscriber(
            self.detections_topic, DetectedObjectArray, self.detections_cb,
            queue_size=5)
        rospy.Service(
            "/subject1/direction_sign/reset", Trigger, self.reset_cb)
        rospy.Timer(rospy.Duration(1.0), lambda _event: self.publish_state())
        rospy.on_shutdown(self.shutdown)

        self.publish_state()
        rospy.logwarn(
            "Direction sign guidance ready: %s, confirm=%d, goal=%.1fm@%.1fdeg",
            self.detections_topic, self.confirm_frames,
            self.goal_distance_m, self.goal_angle_deg)

    @staticmethod
    def _class_set(values):
        if not isinstance(values, list):
            raise ValueError("left_classes/right_classes 必须是 YAML 列表")
        return {str(value).strip().lower() for value in values if str(value).strip()}

    def waypoint_status_cb(self, msg):
        for token in msg.data.split():
            if token.startswith("state="):
                with self.lock:
                    self.waypoint_state = token.split("=", 1)[1]
                break

    def _direction(self, class_name):
        name = str(class_name).strip().lower()
        if name in self.left_classes:
            return self.LEFT
        if name in self.right_classes:
            return self.RIGHT
        return None

    def _valid_candidate(self, obj):
        direction = self._direction(obj.class_name)
        if direction is None or float(obj.confidence) < self.min_confidence:
            return None
        if obj.x_max - obj.x_min < self.min_bbox_width_px:
            return None
        if obj.y_max - obj.y_min < self.min_bbox_height_px:
            return None

        depth_ok = bool(obj.depth_valid) and math.isfinite(float(obj.depth_m))
        if self.require_valid_depth and not depth_ok:
            return None
        if depth_ok and not self.min_distance_m <= float(obj.depth_m) <= self.max_distance_m:
            return None
        return direction

    def _best_candidate(self, msg):
        best = None
        for obj in msg.objects:
            direction = self._valid_candidate(obj)
            if direction is None:
                continue
            if best is None or float(obj.confidence) > float(best[1].confidence):
                best = (direction, obj)
        return best

    def detections_cb(self, msg):
        if msg.header.stamp != rospy.Time(0):
            age = (rospy.Time.now() - msg.header.stamp).to_sec()
            if age < -0.2 or age > self.max_detection_age_s:
                return

        now = rospy.Time.now()
        with self.lock:
            if self.executing or (self.single_use and self.used):
                return
            if self.only_when_running and self.waypoint_state != "RUNNING":
                self._clear_confirmation()
                return

            candidate = self._best_candidate(msg)
            if candidate is None:
                if (self.last_candidate_time != rospy.Time(0) and
                        (now - self.last_candidate_time).to_sec() >
                        self.confirm_max_gap_s):
                    self._clear_confirmation()
                    self.publish_state()
                return

            direction, obj = candidate
            gap = (float("inf") if self.last_candidate_time == rospy.Time(0)
                   else (now - self.last_candidate_time).to_sec())
            if direction != self.candidate_direction or gap > self.confirm_max_gap_s:
                self.candidate_direction = direction
                self.candidate_count = 1
            else:
                self.candidate_count += 1
            self.last_candidate_time = now
            self.state = "CONFIRMING"
            self.selected_class = str(obj.class_name)
            self.selected_confidence = float(obj.confidence)
            self.publish_state()

            if self.candidate_count < self.confirm_frames:
                return

            self.executing = True
            self.selected_direction = direction
            threading.Thread(
                target=self.execute_turn, args=(direction,), daemon=True).start()

    def _clear_confirmation(self):
        self.candidate_direction = None
        self.candidate_count = 0
        self.last_candidate_time = rospy.Time(0)
        if not self.executing and not self.used:
            self.state = "WAITING"

    def make_goal(self, direction):
        angle = math.radians(self.goal_angle_deg)
        if direction == self.RIGHT:
            angle = -angle
        self.goal_x = self.goal_distance_m * math.cos(angle)
        self.goal_y = self.goal_distance_m * math.sin(angle)
        q = tft.quaternion_from_euler(0.0, 0.0, angle)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        self.goal_pub.publish(goal.target_pose)
        return goal

    def execute_turn(self, direction):
        try:
            self.set_state("PAUSING_WAYPOINTS")
            rospy.wait_for_service(self.pause_service, self.service_timeout_s)
            response = self.pause_proxy()
            if not response.success:
                self.fail("暂停航点失败：%s" % response.message, used=False)
                return

            rospy.sleep(self.pause_settle_s)
            if not self.move_base.wait_for_server(rospy.Duration(self.action_timeout_s)):
                self.fail("等待 move_base 超时；原航点保持 PAUSED", used=True)
                return

            goal = self.make_goal(direction)
            self.direction_pub.publish(String(data=direction))
            self.set_state("EXECUTING_LOCAL_TURN")
            rospy.logwarn(
                "指示牌=%s class=%s confidence=%.3f，临时目标 x=%.2f y=%.2f",
                direction, self.selected_class, self.selected_confidence,
                self.goal_x, self.goal_y)

            self.move_base.send_goal(goal)
            if not self.move_base.wait_for_result(rospy.Duration(self.goal_timeout_s)):
                self.move_base.cancel_goal()
                self.fail("临时目标超时；原航点保持 PAUSED", used=True)
                return
            if self.move_base.get_state() != GoalStatus.SUCCEEDED:
                self.fail(
                    "临时目标失败 status=%d；原航点保持 PAUSED" %
                    self.move_base.get_state(), used=True)
                return

            rospy.wait_for_service(self.resume_service, self.service_timeout_s)
            response = self.resume_proxy()
            if not response.success:
                self.fail("临时目标完成，但恢复航点失败：%s" % response.message,
                          used=True)
                return

            with self.lock:
                self.state = "COMPLETED"
                self.used = True
                self.executing = False
                self.error = ""
                self.publish_state()
            rospy.logwarn(
                "临时%s转向完成，已恢复当前未完成 GPS 航点并重新规划。",
                direction)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self.fail("指示牌临时目标异常：%s" % exc, used=True)
        except Exception as exc:
            self.fail("指示牌节点未处理异常：%s" % exc, used=True)

    def set_state(self, state):
        with self.lock:
            self.state = state
            self.publish_state()

    def fail(self, message, used):
        with self.lock:
            self.state = "ERROR"
            self.error = str(message)
            self.executing = False
            self.used = bool(used)
            self.publish_state()
        rospy.logerr("%s。排查后可手动调用 /subject1/resume_waypoints", message)

    def reset_cb(self, _request):
        with self.lock:
            if self.executing:
                return TriggerResponse(False, "临时目标正在执行")
            self.used = False
            self.selected_direction = "NONE"
            self.selected_class = "NONE"
            self.selected_confidence = 0.0
            self.goal_x = float("nan")
            self.goal_y = float("nan")
            self.error = ""
            self._clear_confirmation()
            self.publish_state()
        return TriggerResponse(True, "指示牌状态已重置")

    def publish_state(self):
        with self.lock:
            text = (
                "state=%s waypoint_state=%s used=%s executing=%s "
                "candidate=%s count=%d/%d selected=%s class=%s "
                "confidence=%.3f goal_x=%s goal_y=%s error=%s" % (
                    self.state, self.waypoint_state, self.used, self.executing,
                    self.candidate_direction or "NONE", self.candidate_count,
                    self.confirm_frames, self.selected_direction,
                    self.selected_class, self.selected_confidence,
                    "nan" if not math.isfinite(self.goal_x) else "%.3f" % self.goal_x,
                    "nan" if not math.isfinite(self.goal_y) else "%.3f" % self.goal_y,
                    self.error.replace(" ", "_") if self.error else "NONE"))
            self.state_pub.publish(String(data=text))

    def shutdown(self):
        try:
            self.move_base.cancel_goal()
        except Exception:
            pass


def main():
    rospy.init_node("direction_sign_local_goal")
    DirectionSignLocalGoal()
    rospy.spin()


if __name__ == "__main__":
    main()
