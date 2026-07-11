#!/usr/bin/env python
# coding=utf-8
# 主车发送主车坐标以及主车速度给从车

import math
import rospy
import socket
import sys
import struct
import tf
from numpy import array

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry  

x_m = 0
y_m = 0
yaw_m = 0

odom_vx = 0
odom_vy = 0
odom_az = 0

origin_x = 0
origin_y = 0
origin_yaw = 0
initialized = False

def odom_callback(msg):
	global odom_vx,odom_vy,odom_az,x_m,y_m,yaw_m
	x_m = msg.pose.pose.position.x
	y_m = msg.pose.pose.position.y
	q = msg.pose.pose.orientation
	siny_cosp = 2 * (q.w * q.z + q.x * q.y)
	cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
	yaw_m = math.atan2(siny_cosp, cosy_cosp)
	odom_vx = msg.twist.twist.linear.x
	odom_vy = msg.twist.twist.linear.y
	odom_az = msg.twist.twist.angular.z
	#print(odom_vx,odom_vy,odom_az)

def publishOdom():
	global odom_vx,odom_vy,odom_az
	global origin_x,origin_y,origin_yaw,initialized
	rospy.init_node('send_tfodom')
	rospy.Subscriber('odom',Odometry,odom_callback) #/odom_apm 
	listener = tf.TransformListener()
	soc=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	soc.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)#udp 广播
	network = '<broadcast>' #向所有机器人广播
	slave_ip = [
	'192.168.10.21',
	'192.168.10.25',
	'192.168.10.24',
	'192.168.10.26'
	]
	rate = rospy.Rate(25.0)
	while not rospy.is_shutdown():
		try:
			(trans,rot) = listener.lookupTransform("odom","base_link",rospy.Time(0)) #odom base_link base_link相对于odom坐标系的位置
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			#print("i cannt reciver")
			rospy.Duration(1.0)
			continue

		(roll,pitch,yaw) = tf.transformations.euler_from_quaternion(rot)
	    
		
		if not initialized:
			origin_x = x_m
			origin_y = y_m
			origin_yaw = yaw_m
			initialized = True
			rospy.loginfo("Initial position recorded: x=%.3f, y=%.3f, yaw=%.3f", origin_x, origin_y, origin_yaw)
        

		rel_x = x_m - origin_x
		rel_y = y_m - origin_y
		rel_yaw = yaw_m - origin_yaw

		send_data = struct.pack("ffffff",trans[0],trans[1],yaw,odom_vx,odom_vy,odom_az)
		#send_data = struct.pack("ffffff",rel_x,rel_y,rel_yaw,odom_vx,odom_vy,odom_az)
		
		# send_data = struct.pack("ffffff",x_m,y_m,yaw_m,odom_vx,odom_vy,odom_az)
		for ip in slave_ip:
			soc.sendto(send_data, (ip,10000))
		print(trans[0],trans[1],yaw,odom_vx,odom_vy,odom_az)
		#print(rel_x,rel_y,rel_yaw,odom_vx,odom_vy,odom_az)
		#print(x_m,y_m,yaw_m,odom_vx,odom_vy,odom_az)
		rate.sleep()

if __name__ == '__main__':
    try:
        publishOdom()
    except rospy.ROSInterruptException:
        soc.close()
        pass
