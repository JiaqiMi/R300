#!/usr/bin/env python3
# 重力对齐：收到 IMU 3 秒后，用 madgwick 姿态发布一次静态 TF odom->camera_init，
# 使 FAST-LIO 的地图系 z 轴与重力对齐（高程图必需）。

import rospy
import tf2_ros
from sensor_msgs.msg import Imu


class StaticTransformPublisher:
    def __init__(self):
        rospy.init_node('livox_odom_transform')
        self.start_time = rospy.Time.now()
        self.imu_sub = rospy.Subscriber('/livox/imu_filtered', Imu, self.imu_callback)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.odom_frame = "odom"
        self.camera_init_frame = "camera_init"
        self.transform_broadcasted = False

    def imu_callback(self, msg):
        # 等 3 秒让 madgwick 收敛，再发布一次静态变换
        if rospy.Time.now() - self.start_time > rospy.Duration(3) and not self.transform_broadcasted:
            orientation_q = msg.orientation
            t = tf2_ros.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.camera_init_frame
            t.transform.translation.x = 0
            t.transform.translation.y = 0
            t.transform.translation.z = 0
            t.transform.rotation.x = orientation_q.x
            t.transform.rotation.y = orientation_q.y
            t.transform.rotation.z = orientation_q.z
            t.transform.rotation.w = orientation_q.w
            self.static_broadcaster.sendTransform(t)
            self.transform_broadcasted = True
            rospy.loginfo("Published static transform from /odom to /camera_init")


if __name__ == '__main__':
    try:
        StaticTransformPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
