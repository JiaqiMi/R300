#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在SSH终端中实时显示 ROS1 local_costmap 的统计信息和ASCII局部地图。"""

import math
import threading
import time
from typing import Optional, Tuple

import rospy
import tf2_ros
from nav_msgs.msg import OccupancyGrid


class CostmapTerminalViewer:
    def __init__(self) -> None:
        self.topic = rospy.get_param(
            "~topic", "/move_base/local_costmap/costmap"
        )
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.radius_m = float(rospy.get_param("~radius_m", 5.0))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 2.0))
        self.downsample = max(1, int(rospy.get_param("~downsample", 2)))
        self.clear_terminal = bool(rospy.get_param("~clear_terminal", True))

        self._lock = threading.Lock()
        self._latest: Optional[OccupancyGrid] = None
        self._last_receive_wall = 0.0
        self._receive_count = 0
        self._first_receive_wall = 0.0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.subscriber = rospy.Subscriber(
            self.topic,
            OccupancyGrid,
            self._callback,
            queue_size=1,
        )

    def _callback(self, message: OccupancyGrid) -> None:
        now = time.time()
        with self._lock:
            self._latest = message
            self._last_receive_wall = now
            self._receive_count += 1
            if self._first_receive_wall == 0.0:
                self._first_receive_wall = now

    def _robot_cell(
        self, message: OccupancyGrid
    ) -> Tuple[int, int, str]:
        frame = message.header.frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            mx = int(
                math.floor(
                    (x - message.info.origin.position.x)
                    / message.info.resolution
                )
            )
            my = int(
                math.floor(
                    (y - message.info.origin.position.y)
                    / message.info.resolution
                )
            )
            return mx, my, "TF"
        except Exception as exc:
            return (
                message.info.width // 2,
                message.info.height // 2,
                "地图中心回退: {}".format(str(exc).splitlines()[0]),
            )

    @staticmethod
    def _symbol(value: int) -> str:
        if value < 0:
            return "?"
        if value == 0:
            return " "
        if value >= 90:
            return "#"
        if value >= 60:
            return "O"
        if value >= 30:
            return "+"
        return "."

    def _block_symbol(
        self,
        data,
        width: int,
        height: int,
        x0: int,
        y0: int,
        step: int,
    ) -> str:
        values = []
        for yy in range(y0, min(y0 + step, height)):
            row = yy * width
            for xx in range(x0, min(x0 + step, width)):
                values.append(data[row + xx])

        if not values:
            return " "

        known = [v for v in values if v >= 0]
        if not known:
            return "?"
        return self._symbol(max(known))

    def _render(self, message: OccupancyGrid) -> str:
        width = int(message.info.width)
        height = int(message.info.height)
        resolution = float(message.info.resolution)
        data = message.data

        robot_x, robot_y, robot_source = self._robot_cell(message)
        radius_cells = max(1, int(math.ceil(self.radius_m / resolution)))

        min_x = max(0, robot_x - radius_cells)
        max_x = min(width - 1, robot_x + radius_cells)
        min_y = max(0, robot_y - radius_cells)
        max_y = min(height - 1, robot_y + radius_cells)

        free = unknown = low = inflated = lethal = 0
        for value in data:
            if value < 0:
                unknown += 1
            elif value == 0:
                free += 1
            elif value >= 90:
                lethal += 1
            elif value >= 50:
                inflated += 1
            else:
                low += 1

        now = time.time()
        with self._lock:
            age = now - self._last_receive_wall
            elapsed = max(1e-6, now - self._first_receive_wall)
            average_rate = self._receive_count / elapsed

        lines = [
            "Topic: {}".format(self.topic),
            "Frame: {} | size={}x{} | resolution={:.3f} m/cell".format(
                message.header.frame_id, width, height, resolution
            ),
            "Origin: ({:.2f}, {:.2f}) | robot_cell=({}, {}) [{}]".format(
                message.info.origin.position.x,
                message.info.origin.position.y,
                robot_x,
                robot_y,
                robot_source,
            ),
            "消息年龄={:.3f}s | 平均接收频率≈{:.2f} Hz".format(
                age, average_rate
            ),
            "free={} low={} inflated={} lethal={} unknown={}".format(
                free, low, inflated, lethal, unknown
            ),
            "显示范围: 机器人周围 ±{:.1f}m，降采样={}格/字符".format(
                self.radius_m, self.downsample
            ),
            "图例: R=机器人  #=致命障碍  O=高代价膨胀  +=中代价  .=低代价  ?=未知",
            "",
        ]

        for y in range(max_y, min_y - 1, -self.downsample):
            chars = []
            for x in range(min_x, max_x + 1, self.downsample):
                if (
                    x <= robot_x < x + self.downsample
                    and y <= robot_y < y + self.downsample
                ):
                    chars.append("R")
                else:
                    chars.append(
                        self._block_symbol(
                            data,
                            width,
                            height,
                            x,
                            y,
                            self.downsample,
                        )
                    )
            lines.append("".join(chars))

        return "\n".join(lines)

    def run(self) -> None:
        rate = rospy.Rate(max(0.2, self.refresh_hz))
        while not rospy.is_shutdown():
            with self._lock:
                message = self._latest

            if self.clear_terminal:
                print("\033[2J\033[H", end="")

            if message is None:
                print("等待 costmap: {}".format(self.topic))
            else:
                print(self._render(message))

            rate.sleep()


def main() -> None:
    rospy.init_node("costmap_terminal_viewer")
    viewer = CostmapTerminalViewer()
    viewer.run()


if __name__ == "__main__":
    main()
