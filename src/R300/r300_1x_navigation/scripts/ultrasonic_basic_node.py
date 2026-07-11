#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import struct
import time
import serial
import rospy

from std_msgs.msg import Float32MultiArray, Float32, String, UInt8MultiArray


def crc16_modbus(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_cmd(slave, start, count):
    body = struct.pack(">BBHH", slave, 0x03, start, count)
    crc = crc16_modbus(body)
    return body + struct.pack("<H", crc)


def parse_frame(resp, slave, count):
    expected_len = 1 + 1 + 1 + 2 * count + 2

    if len(resp) < expected_len:
        raise RuntimeError("response too short: len=%d hex=%s" % (len(resp), resp.hex(" ")))

    for i in range(0, len(resp) - expected_len + 1):
        frame = resp[i:i + expected_len]

        if frame[0] != slave:
            continue
        if frame[1] != 0x03:
            continue
        if frame[2] != 2 * count:
            continue

        recv_crc = struct.unpack("<H", frame[-2:])[0]
        calc_crc = crc16_modbus(frame[:-2])
        if recv_crc != calc_crc:
            continue

        raw = []
        for k in range(count):
            v = struct.unpack(">H", frame[3 + 2 * k:5 + 2 * k])[0]
            raw.append(v)

        return raw, frame

    raise RuntimeError("no valid frame, hex=%s" % resp.hex(" "))


class UltrasonicBasicNode:
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyUSB1")
        self.baud = int(rospy.get_param("~baud", 9600))
        self.slave = int(rospy.get_param("~slave", 1))
        self.start = int(str(rospy.get_param("~start", "0x0106")), 0)
        self.count = int(rospy.get_param("~count", 4))
        self.rate_hz = float(rospy.get_param("~rate", 5.0))
        self.scale = float(rospy.get_param("~scale", 0.001))

        self.min_valid_m = float(rospy.get_param("~min_valid_m", 0.02))
        self.max_valid_m = float(rospy.get_param("~max_valid_m", 6.0))
        self.invalid_raw_min = int(rospy.get_param("~invalid_raw_min", 65000))

        self.pub_ranges = rospy.Publisher("/ultrasonic/ranges", Float32MultiArray, queue_size=10)
        self.pub_min = rospy.Publisher("/ultrasonic/min_range", Float32, queue_size=10)
        self.pub_valid = rospy.Publisher("/ultrasonic/valid_mask", UInt8MultiArray, queue_size=10)
        self.pub_debug = rospy.Publisher("/ultrasonic/debug", String, queue_size=10)

        self.cmd = build_cmd(self.slave, self.start, self.count)

        rospy.loginfo("ultrasonic config: port=%s baud=%d slave=%d start=0x%04X count=%d rate=%.1f",
                      self.port, self.baud, self.slave, self.start, self.count, self.rate_hz)
        rospy.loginfo("ultrasonic request cmd: %s", self.cmd.hex(" "))

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=0.15,
            write_timeout=0.15,
        )

        time.sleep(0.5)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def raw_to_meter(self, raw):
        if raw == 0 or raw >= self.invalid_raw_min:
            return float("nan"), 0

        dist = raw * self.scale

        if dist < self.min_valid_m or dist > self.max_valid_m:
            return float("nan"), 0

        return dist, 1

    def read_once(self):
        self.ser.reset_input_buffer()
        self.ser.write(self.cmd)
        self.ser.flush()

        resp = self.ser.read(64)
        raw, frame = parse_frame(resp, self.slave, self.count)

        ranges = []
        valid = []

        for v in raw:
            d, ok = self.raw_to_meter(v)
            ranges.append(d)
            valid.append(ok)

        return raw, ranges, valid, frame

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        fail_count = 0

        while not rospy.is_shutdown():
            try:
                raw, ranges, valid, frame = self.read_once()

                msg = Float32MultiArray()
                msg.data = ranges
                self.pub_ranges.publish(msg)

                valid_msg = UInt8MultiArray()
                valid_msg.data = valid
                self.pub_valid.publish(valid_msg)

                finite = [x for x in ranges if math.isfinite(x)]
                min_range = min(finite) if finite else float("nan")
                self.pub_min.publish(Float32(data=min_range))

                debug = "OK raw=%s ranges=%s valid=%s min=%.3f frame=%s" % (
                    raw,
                    [round(x, 3) if math.isfinite(x) else None for x in ranges],
                    valid,
                    min_range,
                    frame.hex(" "),
                )
                self.pub_debug.publish(debug)

                if fail_count > 0:
                    rospy.loginfo("ultrasonic recovered: %s", debug)

                fail_count = 0

            except Exception as e:
                fail_count += 1
                rospy.logwarn_throttle(1.0, "ultrasonic read failed: %s", str(e))
                self.pub_debug.publish("READ_FAIL x%d: %s" % (fail_count, str(e)))

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("ultrasonic_basic_node")
    node = UltrasonicBasicNode()
    node.spin()