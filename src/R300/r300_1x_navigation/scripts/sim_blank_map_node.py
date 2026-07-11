#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import OccupancyGrid


class SimBlankMapNode(object):
    def __init__(self):
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.resolution = float(rospy.get_param("~resolution", 0.10))
        self.width_m = float(rospy.get_param("~width_m", 80.0))
        self.height_m = float(rospy.get_param("~height_m", 80.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))

        self.width = int(round(self.width_m / self.resolution))
        self.height = int(round(self.height_m / self.resolution))

        self.msg = OccupancyGrid()
        self.msg.header.frame_id = self.map_frame
        self.msg.info.resolution = self.resolution
        self.msg.info.width = self.width
        self.msg.info.height = self.height

        # 让车初始位于地图中心附近。
        self.msg.info.origin.position.x = -0.5 * self.width_m
        self.msg.info.origin.position.y = -0.5 * self.height_m
        self.msg.info.origin.position.z = 0.0
        self.msg.info.origin.orientation.w = 1.0

        # 0 表示完全空旷，-1 表示未知，100 表示障碍。
        self.msg.data = [0] * (self.width * self.height)

        self.pub = rospy.Publisher(
            self.map_topic, OccupancyGrid, queue_size=1, latch=True)

        rospy.logwarn(
            "sim_blank_map_node started: topic=%s frame=%s size=%.1fm x %.1fm "
            "resolution=%.2f cells=%d x %d",
            self.map_topic, self.map_frame, self.width_m, self.height_m,
            self.resolution, self.width, self.height)

    def spin(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.msg.header.stamp = rospy.Time.now()
            self.pub.publish(self.msg)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("sim_blank_map_node")
    node = SimBlankMapNode()
    node.spin()
