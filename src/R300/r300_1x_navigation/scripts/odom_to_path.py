#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class OdomToPath:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/one_x/odom")
        self.path_topic = rospy.get_param("~path_topic", "/one_x/path")
        self.max_points = int(rospy.get_param("~max_points", 3000))

        self.path = Path()
        self.pub = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=50)

        rospy.logwarn("odom_to_path: %s -> %s", self.odom_topic, self.path_topic)

    def odom_cb(self, msg):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose

        self.path.header = msg.header
        self.path.poses.append(ps)

        if len(self.path.poses) > self.max_points:
            self.path.poses = self.path.poses[-self.max_points:]

        self.pub.publish(self.path)


if __name__ == "__main__":
    rospy.init_node("odom_to_path")
    node = OdomToPath()
    rospy.spin()
