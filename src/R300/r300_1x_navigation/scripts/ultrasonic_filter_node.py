#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import math

from collections import deque



import rospy

from std_msgs.msg import Float32MultiArray, Float32, UInt8MultiArray, String, Bool





class UltrasonicFilterNode:

    def __init__(self):

        self.input_topic = rospy.get_param("~input_topic", "/ultrasonic/ranges")



        self.window_size = int(rospy.get_param("~window_size", 3))

        self.max_jump_m = float(rospy.get_param("~max_jump_m", 0.50))



        # 通道定义：相对车体

        self.idx_right_front = int(rospy.get_param("~idx_right_front", 0))

        self.idx_front_right = int(rospy.get_param("~idx_front_right", 1))

        self.idx_front_left = int(rospy.get_param("~idx_front_left", 2))

        self.idx_left_front = int(rospy.get_param("~idx_left_front", 3))



        # 阈值：这里只做状态判断，不控制车

        self.front_warn_m = float(rospy.get_param("~front_warn_m", 0.80))

        self.front_danger_m = float(rospy.get_param("~front_danger_m", 0.45))

        self.side_warn_m = float(rospy.get_param("~side_warn_m", 0.45))

        self.side_danger_m = float(rospy.get_param("~side_danger_m", 0.25))



        self.trigger_frames = int(rospy.get_param("~trigger_frames", 2))

        self.clear_frames = int(rospy.get_param("~clear_frames", 3))



        self.buffers = [deque(maxlen=self.window_size) for _ in range(4)]

        self.last_filtered = [float("nan")] * 4



        self.warn_count = 0

        self.danger_count = 0

        self.clear_count = 0



        self.warn_latched = False

        self.danger_latched = False



        self.pub_filtered = rospy.Publisher("/ultrasonic/filtered_ranges", Float32MultiArray, queue_size=10)

        self.pub_filtered_valid = rospy.Publisher("/ultrasonic/filtered_valid_mask", UInt8MultiArray, queue_size=10)



        self.pub_front_min = rospy.Publisher("/ultrasonic/front_min", Float32, queue_size=10)

        self.pub_side_min = rospy.Publisher("/ultrasonic/side_min", Float32, queue_size=10)

        self.pub_nearest = rospy.Publisher("/ultrasonic/nearest_range", Float32, queue_size=10)



        self.pub_warn = rospy.Publisher("/ultrasonic/warn", Bool, queue_size=10)

        self.pub_danger = rospy.Publisher("/ultrasonic/danger", Bool, queue_size=10)

        self.pub_state = rospy.Publisher("/ultrasonic/state", String, queue_size=10)



        rospy.Subscriber(self.input_topic, Float32MultiArray, self.ranges_cb, queue_size=10)



        rospy.loginfo("ultrasonic_filter_node started, input=%s", self.input_topic)

        rospy.loginfo(

            "channel map: data[0]=right_front, data[1]=front_right, data[2]=front_left, data[3]=left_front"

        )



    @staticmethod

    def valid(x):

        return math.isfinite(x) and x > 0.0



    @staticmethod

    def median(values):

        vals = sorted(values)

        n = len(vals)

        if n == 0:

            return float("nan")

        if n % 2 == 1:

            return vals[n // 2]

        return 0.5 * (vals[n // 2 - 1] + vals[n // 2])



    def update_one_channel(self, i, value):

        if not self.valid(value):

            return self.last_filtered[i]



        last = self.last_filtered[i]



        # 简单跳变抑制：如果单帧变化过大，先不立即相信

        if self.valid(last) and abs(value - last) > self.max_jump_m:

            self.buffers[i].append(last)

        else:

            self.buffers[i].append(value)



        finite_values = [x for x in self.buffers[i] if self.valid(x)]

        filtered = self.median(finite_values)



        self.last_filtered[i] = filtered

        return filtered



    def get_min_by_indices(self, ranges, indices):

        vals = []

        for idx in indices:

            if 0 <= idx < len(ranges):

                v = ranges[idx]

                if self.valid(v):

                    vals.append(v)

        return min(vals) if vals else float("nan")



    def ranges_cb(self, msg):

        raw = list(msg.data)



        while len(raw) < 4:

            raw.append(float("nan"))



        filtered = []

        valid_mask = []



        for i in range(4):

            f = self.update_one_channel(i, raw[i])

            filtered.append(f)

            valid_mask.append(1 if self.valid(f) else 0)



        front_min = self.get_min_by_indices(

            filtered,

            [self.idx_front_right, self.idx_front_left]

        )



        side_min = self.get_min_by_indices(

            filtered,

            [self.idx_right_front, self.idx_left_front]

        )



        nearest = self.get_min_by_indices(filtered, [0, 1, 2, 3])



        front_danger = self.valid(front_min) and front_min < self.front_danger_m

        side_danger = self.valid(side_min) and side_min < self.side_danger_m

        danger_now = front_danger or side_danger



        front_warn = self.valid(front_min) and front_min < self.front_warn_m

        side_warn = self.valid(side_min) and side_min < self.side_warn_m

        warn_now = front_warn or side_warn



        if danger_now:

            self.danger_count += 1

            self.clear_count = 0

            if self.danger_count >= self.trigger_frames:

                self.danger_latched = True

                self.warn_latched = True



        elif warn_now:

            self.warn_count += 1

            self.clear_count = 0

            self.danger_count = 0

            if self.warn_count >= self.trigger_frames:

                self.warn_latched = True



        else:

            self.clear_count += 1

            self.warn_count = 0

            self.danger_count = 0



            if self.clear_count >= self.clear_frames:

                self.warn_latched = False

                self.danger_latched = False



        msg_filtered = Float32MultiArray()

        msg_filtered.data = filtered

        self.pub_filtered.publish(msg_filtered)



        msg_valid = UInt8MultiArray()

        msg_valid.data = valid_mask

        self.pub_filtered_valid.publish(msg_valid)



        self.pub_front_min.publish(Float32(data=front_min))

        self.pub_side_min.publish(Float32(data=side_min))

        self.pub_nearest.publish(Float32(data=nearest))



        self.pub_warn.publish(Bool(data=self.warn_latched))

        self.pub_danger.publish(Bool(data=self.danger_latched))



        state = (

            "filtered=%s valid=%s front_min=%.3f side_min=%.3f nearest=%.3f "

            "warn=%s danger=%s"

            % (

                [round(x, 3) if self.valid(x) else None for x in filtered],

                valid_mask,

                front_min if self.valid(front_min) else float("nan"),

                side_min if self.valid(side_min) else float("nan"),

                nearest if self.valid(nearest) else float("nan"),

                self.warn_latched,

                self.danger_latched,

            )

        )



        self.pub_state.publish(state)





if __name__ == "__main__":

    rospy.init_node("ultrasonic_filter_node")

    UltrasonicFilterNode()

    rospy.spin()
