#include "ros/ros.h"
#include "tf/transform_broadcaster.h"
#include "nav_msgs/Odometry.h"
#include "tf/tf.h"
#include "geometry_msgs/PoseStamped.h"

nav_msgs::Odometry odom_msgs;
ros::Publisher odom_pub;
geometry_msgs::TransformStamped odom_trans;

boost::array<double, 36> odom_pose_covariance = {
    {1e-9, 0, 0, 0, 0, 0, 
    0, 1e-3, 1e-9, 0, 0, 0,  
    0, 0, 1e6, 0, 0, 0,
    0, 0, 0, 1e6, 0, 0, 
    0, 0, 0, 0, 1e6, 0, 
    0, 0, 0, 0, 0, 1e-9}};

void odom_publish()
{
    odom_msgs.header.stamp = ros::Time::now();
    odom_msgs.header.frame_id = "odom";
    odom_msgs.child_frame_id = "base_link";
    odom_msgs.pose.covariance = odom_pose_covariance;

    odom_trans.header.stamp = ros::Time::now();
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_link";
    odom_trans.transform.translation.z = 0.0;

}

void posCallback(const geometry_msgs::PoseStampedConstPtr & pos_msgs)
{
    odom_msgs.pose.pose = pos_msgs->pose;
    odom_trans.transform.translation.x = pos_msgs->pose.position.x;
    odom_trans.transform.translation.y = pos_msgs->pose.position.y;
    odom_trans.transform.rotation = pos_msgs->pose.orientation;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "apm_navigation");
    ros::NodeHandle n;
    tf::TransformBroadcaster odom_br;
    ros::Subscriber pos_sub = n.subscribe("/mavros/local_position/pose", 1, posCallback);
    odom_pub = n.advertise<nav_msgs::Odometry>("/odom_apm",1);
    ros::Rate r(100);
	odom_trans.transform.rotation.z = -0.001;
	odom_trans.transform.rotation.w = 0.999;
	

    while(ros::ok())
    {
        ros::spinOnce();
	odom_publish();
        odom_pub.publish(odom_msgs);
        odom_br.sendTransform(odom_trans);
        r.sleep();
    }
    

    return 0;
}
