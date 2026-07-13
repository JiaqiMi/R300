#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R300 视觉 obstacle_scan -> 360° costmap_scan（V3：保持原始时间戳）

用于修复车辆转向时出现的：
- 旧方向黑团残留；
- 新方向只有红色扫描、没有黑色障碍；
- 障碍物随车体旋转产生拖影或错位。

关键原则
--------
输入 LaserScan 已经在 base_link 下，并带有正常的采集时间戳。
因此本节点：
1. 每收到一帧输入，只发布一帧输出；
2. 保留输入 header.stamp，不把旧扫描改成当前时刻；
3. 不缓存并重复播放上一帧；
4. 输入时间戳过旧时直接丢弃，而不是重新盖当前时间；
5. 同一帧中同时保留当前障碍束，并填充 360° 有限清除射线。

推荐 costmap 参数
------------------
obstacle_range: 10.0
raytrace_range: 10.5

vision_costmap_scan:
  topic: /r300_vision/costmap_scan
  data_type: LaserScan
  marking: true
  clearing: true
  inf_is_valid: false
  observation_persistence: 0.0
"""

import math
from typing import List

import rospy
from sensor_msgs.msg import LaserScan


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class VisionCostmapScanNode:
    def __init__(self) -> None:
        self.input_topic = str(rospy.get_param(
            "~input_topic",
            "/r300_vision/obstacle_scan",
        ))
        self.output_topic = str(rospy.get_param(
            "~output_topic",
            "/r300_vision/costmap_scan",
        ))

        # 留空表示严格继承输入 frame_id。当前正常应为 base_link。
        self.output_frame_id = str(rospy.get_param(
            "~output_frame_id",
            "",
        ))

        self.clear_margin_m = float(rospy.get_param(
            "~clear_margin_m",
            0.05,
        ))
        self.max_clear_range_m = float(rospy.get_param(
            "~max_clear_range_m",
            10.40,
        ))
        self.obstacle_max_range_m = float(rospy.get_param(
            "~obstacle_max_range_m",
            10.00,
        ))
        self.output_angle_increment_deg = float(rospy.get_param(
            "~output_angle_increment_deg",
            0.50,
        ))
        self.obstacle_spread_bins = max(
            0,
            int(rospy.get_param("~obstacle_spread_bins", 1)),
        )

        # 超过该年龄的输入直接丢弃，绝不重新盖当前时间戳。
        self.max_input_age_s = float(rospy.get_param(
            "~max_input_age_s",
            0.50,
        ))
        self.max_future_stamp_s = float(rospy.get_param(
            "~max_future_stamp_s",
            0.10,
        ))
        self.allow_zero_stamp = bool(rospy.get_param(
            "~allow_zero_stamp",
            False,
        ))

        self._validate_parameters()

        self.output_angle_min = -math.pi
        self.output_angle_increment = math.radians(
            self.output_angle_increment_deg
        )

        # 使用 360° / increment 个互不重复的方向：
        # [-pi, pi-increment]
        self.output_beam_count = int(round(
            (2.0 * math.pi) / self.output_angle_increment
        ))
        self.output_angle_max = (
            self.output_angle_min
            + (self.output_beam_count - 1)
            * self.output_angle_increment
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
            "Vision costmap-scan V3 ready: %s -> %s, "
            "preserve_stamp=true, beams=%d, increment=%.3fdeg",
            self.input_topic,
            self.output_topic,
            self.output_beam_count,
            self.output_angle_increment_deg,
        )

    def _validate_parameters(self) -> None:
        if self.clear_margin_m <= 0.0:
            raise ValueError("~clear_margin_m must be > 0")
        if self.max_clear_range_m <= 0.0:
            raise ValueError("~max_clear_range_m must be > 0")
        if self.obstacle_max_range_m <= 0.0:
            raise ValueError("~obstacle_max_range_m must be > 0")
        if self.max_input_age_s <= 0.0:
            raise ValueError("~max_input_age_s must be > 0")
        if self.max_future_stamp_s < 0.0:
            raise ValueError("~max_future_stamp_s must be >= 0")
        if not 0.05 <= self.output_angle_increment_deg <= 5.0:
            raise ValueError(
                "~output_angle_increment_deg must be within [0.05, 5.0]"
            )

    def _stamp_is_acceptable(self, message: LaserScan) -> bool:
        if message.header.stamp == rospy.Time(0):
            if self.allow_zero_stamp:
                rospy.logwarn_throttle(
                    2.0,
                    "Input scan has zero stamp; passing it through because "
                    "~allow_zero_stamp=true",
                )
                return True

            rospy.logwarn_throttle(
                2.0,
                "Dropping input scan with zero timestamp",
            )
            return False

        age = (rospy.Time.now() - message.header.stamp).to_sec()

        if age > self.max_input_age_s:
            rospy.logwarn_throttle(
                2.0,
                "Dropping stale input scan: age=%.3fs > %.3fs",
                age,
                self.max_input_age_s,
            )
            return False

        if age < -self.max_future_stamp_s:
            rospy.logwarn_throttle(
                2.0,
                "Dropping future-dated input scan: age=%.3fs",
                age,
            )
            return False

        return True

    def _angle_to_output_index(self, angle: float) -> int:
        wrapped = wrap_to_pi(angle)
        raw_index = int(round(
            (wrapped - self.output_angle_min)
            / self.output_angle_increment
        ))
        return raw_index % self.output_beam_count

    def scan_callback(self, message: LaserScan) -> None:
        if message.range_max <= message.range_min:
            rospy.logwarn_throttle(
                2.0,
                "Invalid input scan limits: range_min=%.3f range_max=%.3f",
                message.range_min,
                message.range_max,
            )
            return

        if not self._stamp_is_acceptable(message):
            return

        clear_range = min(
            float(message.range_max) - self.clear_margin_m,
            self.max_clear_range_m,
        )
        clear_range = max(
            clear_range,
            float(message.range_min) + 0.01,
        )

        output_range_max = max(
            float(message.range_max),
            clear_range + self.clear_margin_m,
        )

        output_ranges: List[float] = [
            clear_range
        ] * self.output_beam_count

        obstacle_beams = 0
        input_angle = float(message.angle_min)

        max_obstacle_range = min(
            float(message.range_max),
            self.obstacle_max_range_m,
        )

        for value in message.ranges:
            if (
                math.isfinite(value)
                and float(message.range_min) <= value <= max_obstacle_range
            ):
                center_index = self._angle_to_output_index(input_angle)

                for offset in range(
                    -self.obstacle_spread_bins,
                    self.obstacle_spread_bins + 1,
                ):
                    output_index = (
                        center_index + offset
                    ) % self.output_beam_count

                    output_ranges[output_index] = min(
                        output_ranges[output_index],
                        float(value),
                    )

                obstacle_beams += 1

            input_angle += float(message.angle_increment)

        output = LaserScan()

        # 核心修复：严格保留原始采集时间，不能在车辆转动时
        # 把旧的 base_link 扫描重新解释为当前姿态。
        output.header.stamp = message.header.stamp
        output.header.seq = message.header.seq
        output.header.frame_id = (
            self.output_frame_id.strip()
            if self.output_frame_id.strip()
            else message.header.frame_id
        )

        output.angle_min = self.output_angle_min
        output.angle_max = self.output_angle_max
        output.angle_increment = self.output_angle_increment
        output.time_increment = 0.0
        output.scan_time = float(message.scan_time)
        output.range_min = float(message.range_min)
        output.range_max = output_range_max
        output.ranges = output_ranges
        output.intensities = []

        self.publisher.publish(output)

        age = (rospy.Time.now() - message.header.stamp).to_sec()
        rospy.loginfo_throttle(
            1.0,
            "costmap_scan V3: frame=%s age=%.3fs "
            "obstacle_beams=%d clear_range=%.2f",
            output.header.frame_id,
            age,
            obstacle_beams,
            clear_range,
        )


def main() -> None:
    rospy.init_node("vision_costmap_scan_node")
    VisionCostmapScanNode()
    rospy.spin()


if __name__ == "__main__":
    main()