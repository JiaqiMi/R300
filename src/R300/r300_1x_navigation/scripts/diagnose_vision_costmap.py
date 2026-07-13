#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""诊断 VisionSnapshotLayer 的实时状态。"""

import math
import os
import sys
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple


def add_ros_paths() -> None:
    distro = os.environ.get("ROS_DISTRO", "noetic")
    for path in (
        f"/opt/ros/{distro}/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


add_ros_paths()

import rosgraph
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan


class Diagnostic:
    def __init__(self) -> None:
        self.duration_s = float(rospy.get_param("~duration_s", 20.0))
        self.raw_topic = str(rospy.get_param(
            "~raw_topic", "/r300_vision/obstacle_scan"
        ))
        self.active_topic = str(rospy.get_param(
            "~active_topic", "/r300_vision/active_obstacle_scan"
        ))
        self.costmap_topic = str(rospy.get_param(
            "~costmap_topic", "/move_base/local_costmap/costmap"
        ))
        self.max_range_m = float(rospy.get_param("~max_range_m", 10.0))

        self.lock = threading.RLock()
        self.raw_scan: Optional[LaserScan] = None
        self.active_scan: Optional[LaserScan] = None
        self.map_data: Optional[List[int]] = None
        self.map_rx_times: Deque[float] = deque(maxlen=100)

        rospy.Subscriber(self.raw_topic, LaserScan, self.raw_callback, queue_size=3)
        rospy.Subscriber(
            self.active_topic, LaserScan, self.active_callback, queue_size=3
        )
        rospy.Subscriber(
            self.costmap_topic, OccupancyGrid, self.map_callback, queue_size=1
        )

    def raw_callback(self, msg: LaserScan) -> None:
        with self.lock:
            self.raw_scan = msg

    def active_callback(self, msg: LaserScan) -> None:
        with self.lock:
            self.active_scan = msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        expected = int(msg.info.width) * int(msg.info.height)
        if len(msg.data) != expected:
            return
        with self.lock:
            self.map_data = list(msg.data)
            self.map_rx_times.append(time.monotonic())

    def scan_stats(
        self, msg: Optional[LaserScan]
    ) -> Tuple[int, Optional[float], float]:
        if msg is None:
            return 0, None, 999.0
        values = [
            float(value)
            for value in msg.ranges
            if math.isfinite(value)
            and msg.range_min <= value <= min(msg.range_max, self.max_range_m)
        ]
        age = (
            999.0
            if msg.header.stamp == rospy.Time(0)
            else (rospy.Time.now() - msg.header.stamp).to_sec()
        )
        return len(values), min(values) if values else None, age

    @staticmethod
    def rate(times: List[float]) -> float:
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        return 0.0 if duration <= 0.0 else (len(times) - 1) / duration

    @staticmethod
    def param(path: str):
        try:
            return rospy.get_param(path)
        except KeyError:
            return "<missing>"

    def print_static(self) -> None:
        print("\n========== VisionSnapshotLayer 参数 ==========")
        ns = "/move_base/local_costmap/vision_snapshot_layer"
        for path in (
            "/move_base/local_costmap/plugins",
            f"{ns}/topic",
            f"{ns}/hold_time_s",
            f"{ns}/expected_update_rate_s",
            f"{ns}/max_message_age_s",
            f"{ns}/stop_on_stale",
            f"{ns}/active_scan_topic",
            "/vision_obstacle_layer_node/hold_time_s",
            "/move_base/local_costmap/always_send_full_costmap",
        ):
            print(f"{path} = {self.param(path)}")

        master = rosgraph.Master(rospy.get_name())
        publishers, subscribers, _services = master.getSystemState()
        raw_subscribers = []
        active_publishers = []
        for topic, nodes in subscribers:
            if topic == self.raw_topic:
                raw_subscribers = nodes
        for topic, nodes in publishers:
            if topic == self.active_topic:
                active_publishers = nodes
        print(f"{self.raw_topic} subscribers = {raw_subscribers}")
        print(f"{self.active_topic} publishers = {active_publishers}")

    def run(self) -> None:
        self.print_static()
        print("\n========== 动态状态 ==========")
        print(
            "时间 | 原始扫描束/最近距 | 活动TTL束/最近距 | "
            "lethal/nonzero | 完整costmap频率"
        )

        start = time.monotonic()
        rate = rospy.Rate(2.0)
        saw_raw = False
        saw_active = False
        saw_lethal = False

        while not rospy.is_shutdown() and time.monotonic() - start < self.duration_s:
            with self.lock:
                raw = self.raw_scan
                active = self.active_scan
                map_data = None if self.map_data is None else list(self.map_data)
                map_times = list(self.map_rx_times)

            raw_count, raw_nearest, raw_age = self.scan_stats(raw)
            active_count, active_nearest, active_age = self.scan_stats(active)
            lethal = 0
            nonzero = 0
            if map_data is not None:
                lethal = sum(1 for value in map_data if value >= 90)
                nonzero = sum(1 for value in map_data if value > 0)

            saw_raw = saw_raw or raw_count > 0
            saw_active = saw_active or active_count > 0
            saw_lethal = saw_lethal or lethal > 0

            def value_text(value: Optional[float]) -> str:
                return "-" if value is None else f"{value:.2f}m"

            elapsed = time.monotonic() - start
            print(
                f"{elapsed:4.1f}s | "
                f"{raw_count:4d}/{value_text(raw_nearest):>6} age={raw_age:.3f} | "
                f"{active_count:4d}/{value_text(active_nearest):>6} age={active_age:.3f} | "
                f"{lethal:5d}/{nonzero:5d} | {self.rate(map_times):.2f} Hz"
            )
            rate.sleep()

        print("\n========== 判断 ==========")
        if not saw_raw:
            print("原始 obstacle_scan 没有有效障碍束，问题仍在视觉检测/适配节点。")
        elif not saw_active:
            print("原始扫描有障碍，但活动TTL扫描没有，VisionSnapshotLayer未加载或未接收。")
        elif not saw_lethal:
            print("活动TTL扫描有障碍，但完整costmap没有致命格，检查插件加载日志。")
        else:
            print("原始扫描 -> odom TTL障碍 -> costmap 标记链路已打通。")
            print("空场景持续1秒后，活动TTL束和致命格应同步降为0。")


def main() -> None:
    rospy.init_node("diagnose_vision_costmap", anonymous=True)
    Diagnostic().run()


if __name__ == "__main__":
    main()
