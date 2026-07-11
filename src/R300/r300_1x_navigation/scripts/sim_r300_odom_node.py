#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight closed-loop R300 simulator for DWA tuning.

It subscribes to DWA cmd_vel and publishes:
  /one_x/odom
  odom -> base_link TF
  /sim/truth_odom

DWA only sees /one_x/odom. /sim/truth_odom is for comparison.

This node is used to test whether DWA parameters themselves cause S-shaped
motion under ideal or drifted odometry.
"""

import math
import random
import rospy
import tf.transformations as tft
import tf2_ros

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class SimR300OdomNode(object):
    def __init__(self):
        self.cmd_topic = rospy.get_param("~cmd_topic", "/subject1/cmd_vel_raw")
        self.nav_odom_topic = rospy.get_param("~nav_odom_topic", "/one_x/odom")
        self.truth_odom_topic = rospy.get_param("~truth_odom_topic", "/sim/truth_odom")

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))

        # Vehicle physical limits used by the simulator.
        self.max_v = float(rospy.get_param("~max_v", 1.5))
        self.max_w = float(rospy.get_param("~max_w", 0.6))
        self.acc_lim_v = float(rospy.get_param("~acc_lim_v", 0.8))
        self.acc_lim_w = float(rospy.get_param("~acc_lim_w", 1.2))

        # First-order response time constants. Larger value = slower response.
        self.tau_v = float(rospy.get_param("~tau_v", 0.20))
        self.tau_w = float(rospy.get_param("~tau_w", 0.15))

        # Optional odometry error model. DWA sees nav odom, not truth odom.
        self.pos_noise_std = float(rospy.get_param("~pos_noise_std", 0.0))
        self.yaw_noise_std_deg = float(rospy.get_param("~yaw_noise_std_deg", 0.0))

        # Constant drift in nav odom, expressed in odom/map frame.
        self.drift_x_mps = float(rospy.get_param("~drift_x_mps", 0.0))
        self.drift_y_mps = float(rospy.get_param("~drift_y_mps", 0.0))
        self.yaw_drift_degps = float(rospy.get_param("~yaw_drift_degps", 0.0))

        # Optional periodic position jump, useful to imitate bad GPS/INS reset.
        self.jump_period_s = float(rospy.get_param("~jump_period_s", 0.0))
        self.jump_std_m = float(rospy.get_param("~jump_std_m", 0.0))

        # Real vehicle state.
        self.x = float(rospy.get_param("~initial_x", 0.0))
        self.y = float(rospy.get_param("~initial_y", 0.0))
        self.yaw = math.radians(float(rospy.get_param("~initial_yaw_deg", 0.0)))

        # Real executed velocity.
        self.v = 0.0
        self.w = 0.0

        # Commanded velocity.
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.last_cmd_time = rospy.Time.now()

        # Drift state.
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_yaw = 0.0
        self.jump_x = 0.0
        self.jump_y = 0.0
        self.last_jump_time = rospy.Time.now()

        self.tf_br = tf2_ros.TransformBroadcaster()
        self.nav_odom_pub = rospy.Publisher(self.nav_odom_topic, Odometry, queue_size=20)
        self.truth_odom_pub = rospy.Publisher(self.truth_odom_topic, Odometry, queue_size=20)

        rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=10)

        self.last_time = rospy.Time.now()

        rospy.logwarn(
            "sim_r300_odom_node started: cmd=%s nav_odom=%s truth_odom=%s "
            "rate=%.1fHz max_v=%.2f max_w=%.2f",
            self.cmd_topic, self.nav_odom_topic, self.truth_odom_topic,
            self.rate_hz, self.max_v, self.max_w)

    def cmd_cb(self, msg):
        self.cmd_v = clamp(msg.linear.x, -self.max_v, self.max_v)
        self.cmd_w = clamp(msg.angular.z, -self.max_w, self.max_w)
        self.last_cmd_time = rospy.Time.now()

    def update_vehicle(self, dt):
        # Stop if command is stale.
        if (rospy.Time.now() - self.last_cmd_time).to_sec() > 0.5:
            target_v = 0.0
            target_w = 0.0
        else:
            target_v = self.cmd_v
            target_w = self.cmd_w

        # First-order response.
        if self.tau_v > 1.0e-4:
            desired_dv = (target_v - self.v) * dt / self.tau_v
        else:
            desired_dv = target_v - self.v

        if self.tau_w > 1.0e-4:
            desired_dw = (target_w - self.w) * dt / self.tau_w
        else:
            desired_dw = target_w - self.w

        # Acceleration limits.
        max_dv = self.acc_lim_v * dt
        max_dw = self.acc_lim_w * dt
        self.v += clamp(desired_dv, -max_dv, max_dv)
        self.w += clamp(desired_dw, -max_dw, max_dw)

        self.v = clamp(self.v, -self.max_v, self.max_v)
        self.w = clamp(self.w, -self.max_w, self.max_w)

        # Unicycle / differential-drive kinematics.
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = wrap_pi(self.yaw + self.w * dt)

    def update_drift(self, dt):
        self.drift_x += self.drift_x_mps * dt
        self.drift_y += self.drift_y_mps * dt
        self.drift_yaw = wrap_pi(
            self.drift_yaw + math.radians(self.yaw_drift_degps) * dt)

        if self.jump_period_s > 0.0 and self.jump_std_m > 0.0:
            now = rospy.Time.now()
            if (now - self.last_jump_time).to_sec() >= self.jump_period_s:
                self.last_jump_time = now
                self.jump_x += random.gauss(0.0, self.jump_std_m)
                self.jump_y += random.gauss(0.0, self.jump_std_m)
                rospy.logwarn(
                    "sim odom jump injected: jump_x=%.3f jump_y=%.3f",
                    self.jump_x, self.jump_y)

    def make_odom(self, stamp, x, y, yaw, v, w, frame_id, child_frame_id):
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.twist.twist.linear.x = v
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = w

        return msg

    def publish_tf(self, stamp, x, y, yaw):
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]

        self.tf_br.sendTransform(tf_msg)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            self.last_time = now

            if dt <= 0.0 or dt > 0.2:
                dt = 1.0 / self.rate_hz

            self.update_vehicle(dt)
            self.update_drift(dt)

            yaw_noise = math.radians(self.yaw_noise_std_deg) * random.gauss(0.0, 1.0)
            nav_x = (
                self.x
                + self.drift_x
                + self.jump_x
                + random.gauss(0.0, self.pos_noise_std)
            )
            nav_y = (
                self.y
                + self.drift_y
                + self.jump_y
                + random.gauss(0.0, self.pos_noise_std)
            )
            nav_yaw = wrap_pi(self.yaw + self.drift_yaw + yaw_noise)

            # DWA sees this odom.
            nav_odom = self.make_odom(
                now, nav_x, nav_y, nav_yaw,
                self.v, self.w,
                self.odom_frame, self.base_frame)
            self.nav_odom_pub.publish(nav_odom)
            self.publish_tf(now, nav_x, nav_y, nav_yaw)

            # Truth odom is only for RViz / analysis.
            truth_odom = self.make_odom(
                now, self.x, self.y, self.yaw,
                self.v, self.w,
                self.odom_frame, "truth_base_link")
            self.truth_odom_pub.publish(truth_odom)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("sim_r300_odom_node")
    node = SimR300OdomNode()
    node.spin()
