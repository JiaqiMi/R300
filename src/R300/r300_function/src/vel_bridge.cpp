#include <ros/ros.h>
#include <string>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/State.h>

geometry_msgs::Twist twist;
std::string state = "MANUAL";

//线速度峰值
double vel_x_max;
//角速度降幅调节参数
double ang_z_p;

void vel_callback(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
	twist.linear.x = msg->twist.linear.y;
  twist.angular.z = -msg->twist.linear.x;
  if(msg->twist.linear.y > vel_x_max)
  {
      twist.linear.x = vel_x_max;
  }
  if(msg->twist.linear.y < -vel_x_max)
  {
      twist.linear.x = -vel_x_max;
  }
	if(state == "GUIDED" || state == "AUTO")
	{
		twist.angular.z = twist.angular.z / ang_z_p;
	}

}

void state_callback(const mavros_msgs::State::ConstPtr& msg)
{
    state = msg->mode;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vel_bridge");
    ros::NodeHandle n;
		ros::param::param<double>("~VEL_MAX", vel_x_max, 1.0);
		ros::param::param<double>("~ANG_P", ang_z_p, 2);
    ros::Subscriber local_vel_sub = n.subscribe("mavros/local_position/velocity_local", 1, vel_callback);
    ros::Subscriber state_sub = n.subscribe("mavros/state", 100, state_callback);
    ros::Publisher body_vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel",1);
    ros::Rate r(10);
    while(ros::ok())
    {
        ros::spinOnce();
        std::cout << "Car mode is " << state << std::endl;
        std::cout << "Car linear is " << twist.linear.x << std::endl;
        std::cout << "Car angular is " << twist.angular.z << std::endl;
        if(state != "MANUAL")
        {
            body_vel_pub.publish(twist);
        }
        r.sleep();
    }
}
