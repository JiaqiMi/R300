#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Publish a 360-degree clearing-only LaserScan for the visual costmap layer.

Why:
The normal visual obstacle scan only covers the camera FOV (for example,
-27 deg to +27 deg). Historical obstacle cells that move outside the current
camera FOV cannot be ray-traced by a front-only clear scan and may remain as
black blobs in the local costmap.

This node listens to the visual obstacle scan only as a heartbeat and source
of frame/range metadata. For every received input scan, it publishes a finite
360-degree clearing scan. The visual ObstacleLayer must use:
  - vision_clear_scan: marking=false, clearing=true
  - vision_mark_scan:  marking=true,  clearing=false

costmap_2d clears the visual layer first, then re-marks obstacles from the
current visual obstacle scan. Therefore, the visual layer represents only the
currently detected obstacles, while stale visual cells in any direction are
removed.

Safety boundary:
Use this clear scan ONLY inside the dedicated vision_obstacle_layer. Do not
feed it into lidar, static-map, or other sensor layers.
"""

import math

import rospy
from sensor_msgs.msg import LaserScan


class VisionFullCircleClearScanNode:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param(
            "~input_topic", "/r300_vision/obstacle_scan"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/r300_vision/clear_scan"
        )

        self.clear_margin_m = float(
            rospy.get_param("~clear_margin_m", 0.05)
        )
        self.max_clear_range_m = float(
            rospy.get_param("~max_clear_range_m", 0.0)
        )
        self.clear_angle_increment_deg = float(
            rospy.get_param("~clear_angle_increment_deg", 0.5)
        )

        if self.clear_margin_m <= 0.0:
            raise ValueError("~clear_margin_m must be > 0")
        if self.max_clear_range_m < 0.0:
            raise ValueError("~max_clear_range_m must be >= 0")
        if not 0.05 <= self.clear_angle_increment_deg <= 10.0:
            raise ValueError(
                "~clear_angle_increment_deg must be within [0.05, 10.0]"
            )

        self.angle_increment = math.radians(
            self.clear_angle_increment_deg
        )

        # Cover [-pi, pi) without duplicating the -pi/pi endpoint.
        self.beam_count = int(math.ceil((2.0 * math.pi) / self.angle_increment))
        self.angle_min = -math.pi
        self.angle_max = (
            self.angle_min + (self.beam_count - 1) * self.angle_increment
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            LaserScan,
            queue_size=2,
        )
        self.subscriber = rospy.Subscriber(
            self.input_topic,
            LaserScan,
            self.scan_callback,
            queue_size=2,
        )

        rospy.loginfo(
            "Vision 360-deg clear-scan node ready: %s -> %s, "
            "beams=%d, increment=%.3f deg",
            self.input_topic,
            self.output_topic,
            self.beam_count,
            self.clear_angle_increment_deg,
        )

    def scan_callback(self, message: LaserScan) -> None:
        if message.range_max <= message.range_min:
            rospy.logwarn_throttle(
                2.0,
                "Invalid input LaserScan limits: range_min=%.3f, range_max=%.3f",
                message.range_min,
                message.range_max,
            )
            return

        clear_range = message.range_max - self.clear_margin_m
        if self.max_clear_range_m > 0.0:
            clear_range = min(clear_range, self.max_clear_range_m)
        clear_range = max(clear_range, message.range_min + 0.01)

        output = LaserScan()
        output.header = message.header
        output.angle_min = self.angle_min
        output.angle_max = self.angle_max
        output.angle_increment = self.angle_increment
        output.time_increment = 0.0
        output.scan_time = message.scan_time
        output.range_min = message.range_min
        output.range_max = message.range_max
        output.ranges = [clear_range] * self.beam_count
        output.intensities = []

        self.publisher.publish(output)


def main() -> None:
    rospy.init_node("vision_clear_scan_node")
    VisionFullCircleClearScanNode()
    rospy.spin()


if __name__ == "__main__":
    main()
