#!/usr/bin/env python
# coding=utf-8
# 从车接收主车坐标以及主车速度，并通过话题 multfodom 发布出来

import rospy
import socket
import struct
from std_msgs.msg import Float32MultiArray

def FrameListener():
    rospy.init_node('listen_tfodom_improved')
    pub = rospy.Publisher('multfodom', Float32MultiArray, queue_size=10)

    # 创建 UDP 套接字
    soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # 增大接收缓冲区
    soc.settimeout(0.01)  # 设置非阻塞等待，避免阻塞主线程

    soc.bind(('', 10000))
    rospy.loginfo("UDP socket bound on port 10000")

    while not rospy.is_shutdown():
        try:
            data, address = soc.recvfrom(30)
            if len(data) != 24:
                rospy.logwarn("Invalid data length: %d", len(data))
                continue

            # 解包主车数据
            posx, posy, posaz, odomvx, odomvy, odomaz = struct.unpack("ffffff", data)
            rospy.loginfo_throttle(1, f"Recv: {posx:.2f}, {posy:.2f}, {posaz:.2f}, {odomvx:.2f}, {odomvy:.2f}, {odomaz:.2f}")

            # 封装并发布 ROS 消息
            msg = Float32MultiArray()
            msg.data = [posx, posy, posaz, odomvx, odomvy, odomaz]
            pub.publish(msg)

        except socket.timeout:
            continue  # 忽略无数据情况
        except Exception as e:
            rospy.logwarn("UDP Receive Error: %s", str(e))
            continue

if __name__ == '__main__':
    try:
        FrameListener()
    except rospy.ROSInterruptException:
        pass
