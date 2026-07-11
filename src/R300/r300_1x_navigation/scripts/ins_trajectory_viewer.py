#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import rospy

from nav_msgs.msg import Odometry, Path

from sensor_msgs.msg import NavSatFix

from std_msgs.msg import Float64

from geometry_msgs.msg import PoseStamped

from visualization_msgs.msg import Marker

from std_srvs.srv import Empty, EmptyResponse





class InsTrajectoryViewer:

    def __init__(self):

        self.path = Path()

        self.path.header.frame_id = "odom"



        self.latest_fix = None

        self.latest_heading = None

        self.last_sample_time = rospy.Time(0)



        self.sample_hz = rospy.get_param("~sample_hz", 5.0)

        self.max_points = rospy.get_param("~max_points", 3000)



        self.path_pub = rospy.Publisher(

            "/one_x/trajectory", Path, queue_size=1, latch=True

        )

        self.info_pub = rospy.Publisher(

            "/one_x/trajectory_info", Marker, queue_size=1

        )



        rospy.Subscriber("/one_x/odom", Odometry, self.odom_cb, queue_size=50)

        rospy.Subscriber("/one_x/fix", NavSatFix, self.fix_cb, queue_size=10)

        rospy.Subscriber("/one_x/heading_deg", Float64, self.heading_cb, queue_size=10)



        rospy.Service("/one_x/trajectory_reset", Empty, self.reset_cb)



        rospy.loginfo("INS trajectory viewer started.")

        rospy.loginfo("RViz topics: /one_x/trajectory and /one_x/trajectory_info")



    def fix_cb(self, msg):

        self.latest_fix = msg



    def heading_cb(self, msg):

        self.latest_heading = msg.data



    def reset_cb(self, _req):

        self.path = Path()

        self.path.header.frame_id = "odom"

        self.path_pub.publish(self.path)

        rospy.loginfo("INS trajectory cleared.")

        return EmptyResponse()



    def odom_cb(self, msg):

        now = msg.header.stamp

        if now.is_zero():

            now = rospy.Time.now()



        if (now - self.last_sample_time).to_sec() < 1.0 / self.sample_hz:

            self.publish_info(msg)

            return



        self.last_sample_time = now



        pose = PoseStamped()

        pose.header = msg.header

        pose.pose = msg.pose.pose



        self.path.header.stamp = now

        self.path.header.frame_id = msg.header.frame_id if msg.header.frame_id else "odom"

        self.path.poses.append(pose)



        if len(self.path.poses) > self.max_points:

            self.path.poses = self.path.poses[-self.max_points:]



        self.path_pub.publish(self.path)

        self.publish_info(msg)



    def publish_info(self, odom):

        marker = Marker()

        marker.header.stamp = rospy.Time.now()

        marker.header.frame_id = odom.header.frame_id if odom.header.frame_id else "odom"

        marker.ns = "one_x_info"

        marker.id = 0

        marker.type = Marker.TEXT_VIEW_FACING

        marker.action = Marker.ADD



        marker.pose.position.x = odom.pose.pose.position.x

        marker.pose.position.y = odom.pose.pose.position.y

        marker.pose.position.z = odom.pose.pose.position.z + 1.0



        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.45

        marker.color.r = 1.0

        marker.color.g = 1.0

        marker.color.b = 1.0

        marker.color.a = 1.0



        lat_text = "N/A"

        lon_text = "N/A"

        alt_text = "N/A"



        if self.latest_fix is not None:

            lat_text = "%.8f" % self.latest_fix.latitude

            lon_text = "%.8f" % self.latest_fix.longitude

            alt_text = "%.2f m" % self.latest_fix.altitude



        heading_text = "N/A"

        if self.latest_heading is not None:

            heading_text = "%.2f deg" % self.latest_heading



        marker.text = (

            "INS 1X\n"

            "Lat: %s\n"

            "Lon: %s\n"

            "Alt: %s\n"

            "Heading: %s\n"

            "E: %.2f m  N: %.2f m"

            % (

                lat_text,

                lon_text,

                alt_text,

                heading_text,

                odom.pose.pose.position.x,

                odom.pose.pose.position.y,

            )

        )



        self.info_pub.publish(marker)





if __name__ == "__main__":

    rospy.init_node("ins_trajectory_viewer")

    InsTrajectoryViewer()

    rospy.spin()

