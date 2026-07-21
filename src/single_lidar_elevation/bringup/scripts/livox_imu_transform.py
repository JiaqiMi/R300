#!/usr/bin/env python3
# 雷达内置 IMU 姿态倾角修正：/livox/imu_filtered -> /imu/data
# 倾角由 ~tilt_pitch_deg 参数给出（斜装 45° 的前雷达为 -45；水平安装填 0）。

import math
import numpy as np
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import tf.transformations as tft

rospy.init_node('livox_imu_transform')

tilt_pitch_deg = float(rospy.get_param('~tilt_pitch_deg', -45.0))

pub = rospy.Publisher('/imu/data', Imu, queue_size=10)

correction_q = tft.quaternion_from_euler(0, math.radians(tilt_pitch_deg), 0)
correction_q_inverse = tft.quaternion_from_euler(0, math.radians(-tilt_pitch_deg), 0)
R_corr_inverse = tft.quaternion_matrix(correction_q_inverse)[:3, :3]


def rotate_vector(v, R):
    vec = np.array([v.x, v.y, v.z])
    vec_rot = R.dot(vec)
    return Vector3(*vec_rot)


def callback(msg):
    q_orig = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    q_corr = tft.quaternion_multiply(q_orig, correction_q)
    msg.orientation = Quaternion(*q_corr)
    msg.angular_velocity = rotate_vector(msg.angular_velocity, R_corr_inverse)
    msg.linear_acceleration = rotate_vector(msg.linear_acceleration, R_corr_inverse)
    pub.publish(msg)


sub = rospy.Subscriber('/livox/imu_filtered', Imu, callback)
rospy.spin()
