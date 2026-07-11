#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


def zero_twist():
    return Twist()


class UltrasonicEStopNode:
    def __init__(self):
        self.input_cmd_vel = rospy.get_param("~input_cmd_vel", "/subject1/cmd_vel_guarded")
        self.output_cmd_vel = rospy.get_param("~output_cmd_vel", "/subject1/cmd_vel_safe")

        self.danger_topic = rospy.get_param("~danger_topic", "/ultrasonic/danger")
        self.front_min_topic = rospy.get_param("~front_min_topic", "/ultrasonic/front_min")
        self.side_min_topic = rospy.get_param("~side_min_topic", "/ultrasonic/side_min")

        self.require_ultrasonic = bool(rospy.get_param("~require_ultrasonic", True))
        self.ultrasonic_timeout_s = float(rospy.get_param("~ultrasonic_timeout_s", 0.80))
        self.cmd_timeout_s = float(rospy.get_param("~cmd_timeout_s", 0.50))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))

        # false：danger 时不管前进/后退/转向全部置零，更安全
        # true：danger 时只拦截前进，允许倒车脱困
        self.only_stop_forward = bool(rospy.get_param("~only_stop_forward", False))

        self.last_cmd = Twist()
        self.last_cmd_time = rospy.Time(0)

        self.last_danger = False
        self.last_danger_time = rospy.Time(0)

        self.front_min = float("nan")
        self.side_min = float("nan")

        rospy.Subscriber(self.input_cmd_vel, Twist, self.cmd_cb, queue_size=20)
        rospy.Subscriber(self.danger_topic, Bool, self.danger_cb, queue_size=20)
        rospy.Subscriber(self.front_min_topic, Float32, self.front_cb, queue_size=20)
        rospy.Subscriber(self.side_min_topic, Float32, self.side_cb, queue_size=20)

        self.cmd_pub = rospy.Publisher(self.output_cmd_vel, Twist, queue_size=20)
        self.estop_pub = rospy.Publisher("/ultrasonic/estop_active", Bool, queue_size=10)
        self.state_pub = rospy.Publisher("/ultrasonic/estop_state", String, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self.timer_cb)

        rospy.loginfo(
            "ultrasonic_estop_node started: input=%s output=%s danger=%s",
            self.input_cmd_vel,
            self.output_cmd_vel,
            self.danger_topic,
        )

    def cmd_cb(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = rospy.Time.now()

    def danger_cb(self, msg):
        self.last_danger = bool(msg.data)
        self.last_danger_time = rospy.Time.now()

    def front_cb(self, msg):
        self.front_min = msg.data

    def side_cb(self, msg):
        self.side_min = msg.data

    @staticmethod
    def fmt(x):
        if math.isfinite(x):
            return "%.3f" % x
        return "nan"

    def timer_cb(self, _event):
        now = rospy.Time.now()

        out = self.last_cmd
        force_stop = False
        reasons = []

        cmd_age = (now - self.last_cmd_time).to_sec()
        ultra_age = (now - self.last_danger_time).to_sec()

        if cmd_age > self.cmd_timeout_s:
            force_stop = True
            reasons.append("cmd_timeout %.2fs" % cmd_age)

        if self.require_ultrasonic and ultra_age > self.ultrasonic_timeout_s:
            force_stop = True
            reasons.append("ultrasonic_timeout %.2fs" % ultra_age)

        if self.last_danger:
            if self.only_stop_forward and out.linear.x <= 0.0:
                pass
            else:
                force_stop = True
                reasons.append("ultrasonic_danger")

        if force_stop:
            out = zero_twist()

        self.cmd_pub.publish(out)
        self.estop_pub.publish(Bool(data=force_stop))

        state = (
            "estop=%s danger=%s reason=%s cmd_age=%.2f ultra_age=%.2f "
            "front_min=%s side_min=%s out_vx=%.3f out_wz=%.3f"
            % (
                force_stop,
                self.last_danger,
                ";".join(reasons) if reasons else "pass",
                cmd_age,
                ultra_age,
                self.fmt(self.front_min),
                self.fmt(self.side_min),
                out.linear.x,
                out.angular.z,
            )
        )

        self.state_pub.publish(state)


if __name__ == "__main__":
    rospy.init_node("ultrasonic_estop_node")
    UltrasonicEStopNode()
    rospy.spin()