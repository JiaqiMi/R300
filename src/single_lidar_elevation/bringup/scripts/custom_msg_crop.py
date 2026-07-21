#!/usr/bin/env python3
# 应急节点：FAST-LIO 输入前裁剪（CustomMsg -> CustomMsg）。
# 仅当推车测试发现里程计被"随体移动的物体"（铝腿/小车/推车人）污染时启用（launch 里 lio_input_crop:=true）。
import numpy as np
import rospy
from livox_ros_driver2.msg import CustomMsg


class CustomMsgCrop:
    def __init__(self):
        self.min_range = float(rospy.get_param('~min_range', 0.0))
        self.boxes = np.array(rospy.get_param('~boxes', []), dtype=np.float32).reshape(-1, 6)
        in_topic = rospy.get_param('~input_topic', 'livox/lidar_192_168_1_192')
        out_topic = rospy.get_param('~output_topic', 'livox/lidar_front_cropped')
        self.pub = rospy.Publisher(out_topic, CustomMsg, queue_size=5)
        rospy.Subscriber(in_topic, CustomMsg, self.cb, queue_size=5)
        rospy.loginfo('custom_msg_crop: %s -> %s (min_range=%.2f, boxes=%d)',
                      in_topic, out_topic, self.min_range, len(self.boxes))

    def keep(self, p):
        r2 = p.x * p.x + p.y * p.y + p.z * p.z
        if r2 < self.min_range * self.min_range:
            return False
        for b in self.boxes:
            if b[0] < p.x < b[1] and b[2] < p.y < b[3] and b[4] < p.z < b[5]:
                return False
        return True

    def cb(self, msg):
        msg.points = [p for p in msg.points if self.keep(p)]
        msg.point_num = len(msg.points)
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('custom_msg_crop')
    CustomMsgCrop()
    rospy.spin()
