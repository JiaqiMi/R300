#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt 1X odometry for the non-holonomic DWA local planner.

The 1X driver already publishes body-frame linear.x and angular.z.  This node
only removes lateral motion from the twist and applies small deadbands/clamps.
It intentionally does *not* differentiate yaw again: differentiating the same
heading twice in two nodes can inject high-frequency angular-rate noise into
DWA and make it keep selecting curved trajectories.
"""

import rospy

from nav_msgs.msg import Odometry


class DwaOdomAdapter(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_odom_topic", "/one_x/odom")
        self.output_topic = rospy.get_param(
            "~output_odom_topic", "/subject1/dwa_odom")

        self.forward_deadband_mps = float(rospy.get_param(
            "~forward_deadband_mps", 0.03))
        self.yaw_rate_deadband_radps = float(rospy.get_param(
            "~yaw_rate_deadband_radps", 0.02))
        self.max_yaw_rate_radps = abs(float(rospy.get_param(
            "~max_yaw_rate_radps", 0.45)))

        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)
        self.sub = rospy.Subscriber(
            self.input_topic, Odometry, self.odom_cb, queue_size=20)

        rospy.logwarn(
            "dwa_odom_adapter started: %s -> %s "
            "(copy pose/vx/wz from 1X, force vy=0; no yaw differentiation)",
            self.input_topic, self.output_topic)

    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def odom_cb(self, msg):
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist

        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z

        if abs(vx) < self.forward_deadband_mps:
            vx = 0.0
        if abs(wz) < self.yaw_rate_deadband_radps:
            wz = 0.0

        out.twist.twist.linear.x = vx
        out.twist.twist.linear.y = 0.0
        out.twist.twist.linear.z = 0.0
        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = self.clamp(
            wz, -self.max_yaw_rate_radps, self.max_yaw_rate_radps)

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("dwa_odom_adapter")
    DwaOdomAdapter()
    rospy.spin()
