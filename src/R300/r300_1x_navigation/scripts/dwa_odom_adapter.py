#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare 1X odometry velocity feedback for a differential-drive DWA planner.

The pose and timestamp are copied unchanged from /one_x/odom.  Only the twist
fed to DWA is conditioned:

* force lateral velocity to zero for the non-holonomic R300;
* low-pass filter forward speed and yaw rate;
* apply small deadbands and an angular-rate clamp.

The costmaps still obtain the robot pose from the original odom->base_link TF.
This node is therefore deliberately a velocity-feedback adapter, not a second
TF or localization source.
"""

import math
import threading

import rospy
from nav_msgs.msg import Odometry


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class DwaOdomAdapter(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_odom_topic", "/one_x/odom")
        self.output_topic = rospy.get_param(
            "~output_odom_topic", "/subject1/dwa_odom"
        )

        self.forward_deadband_mps = abs(float(rospy.get_param(
            "~forward_deadband_mps", 0.02
        )))
        self.yaw_rate_deadband_radps = abs(float(rospy.get_param(
            "~yaw_rate_deadband_radps", 0.015
        )))
        self.max_yaw_rate_radps = abs(float(rospy.get_param(
            "~max_yaw_rate_radps", 0.45
        )))
        self.forward_filter_tau_s = max(0.0, float(rospy.get_param(
            "~forward_filter_tau_s", 0.10
        )))
        self.yaw_rate_filter_tau_s = max(0.0, float(rospy.get_param(
            "~yaw_rate_filter_tau_s", 0.15
        )))
        self.max_dt_s = max(0.02, float(rospy.get_param("~max_dt_s", 0.30)))

        self.lock = threading.RLock()
        self.initialized = False
        self.last_stamp = rospy.Time(0)
        self.filtered_vx = 0.0
        self.filtered_wz = 0.0

        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)
        self.sub = rospy.Subscriber(
            self.input_topic, Odometry, self.odom_cb, queue_size=50
        )

        rospy.logwarn(
            "dwa_odom_adapter: %s -> %s; force vy=0, tau_v=%.3fs, "
            "tau_w=%.3fs, wz_deadband=%.3frad/s",
            self.input_topic,
            self.output_topic,
            self.forward_filter_tau_s,
            self.yaw_rate_filter_tau_s,
            self.yaw_rate_deadband_radps,
        )

    @staticmethod
    def low_pass(previous, current, dt, tau):
        if tau <= 1.0e-6 or dt <= 0.0:
            return current
        alpha = dt / (tau + dt)
        return previous + alpha * (current - previous)

    def odom_cb(self, msg):
        stamp = msg.header.stamp
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()

        raw_vx = float(msg.twist.twist.linear.x)
        raw_wz = float(msg.twist.twist.angular.z)

        if not math.isfinite(raw_vx) or not math.isfinite(raw_wz):
            rospy.logwarn_throttle(
                1.0,
                "dwa_odom_adapter discarded non-finite twist: vx=%r wz=%r",
                raw_vx,
                raw_wz,
            )
            return

        with self.lock:
            if not self.initialized:
                self.filtered_vx = raw_vx
                self.filtered_wz = raw_wz
                self.initialized = True
                dt = 0.0
            else:
                dt = (stamp - self.last_stamp).to_sec()
                if dt <= 0.0 or dt > self.max_dt_s:
                    dt = 0.0

                self.filtered_vx = self.low_pass(
                    self.filtered_vx,
                    raw_vx,
                    dt,
                    self.forward_filter_tau_s,
                )
                self.filtered_wz = self.low_pass(
                    self.filtered_wz,
                    raw_wz,
                    dt,
                    self.yaw_rate_filter_tau_s,
                )

            self.last_stamp = stamp
            vx = self.filtered_vx
            wz = self.filtered_wz

        if abs(vx) < self.forward_deadband_mps:
            vx = 0.0
        if abs(wz) < self.yaw_rate_deadband_radps:
            wz = 0.0
        wz = clamp(wz, -self.max_yaw_rate_radps, self.max_yaw_rate_radps)

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist

        out.twist.twist.linear.x = vx
        out.twist.twist.linear.y = 0.0
        out.twist.twist.linear.z = 0.0
        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = wz

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("dwa_odom_adapter")
    DwaOdomAdapter()
    rospy.spin()
