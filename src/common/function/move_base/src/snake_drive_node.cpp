#include <move_base/move_base.h>
#include <move_base_msgs/RecoveryStatus.h>
#include <cmath>

#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>

#include <geometry_msgs/Twist.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
class SnakeDrive {
public:
    SnakeDrive() {
        // 初始化节点
        nh_ = ros::NodeHandle("~");
        
        // 获取参数
        nh_.param("linear_speed", linear_speed_, 0.5);
        nh_.param("max_steering_angle", max_steering_angle_, 0.5);
        nh_.param("frequency", frequency_, 0.5);
        nh_.param("publish_rate", publish_rate_, 20.0);
        
        // 设置发布者
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/ugv2/cmd_vel", 1);
        
        ROS_INFO("Snake Drive node started");
        ROS_INFO("Linear speed: %.2f m/s", linear_speed_);
        ROS_INFO("Max steering angle: %.2f rad", max_steering_angle_);
        ROS_INFO("Frequency: %.2f Hz", frequency_);
    }
    
    void run() {
        ros::Rate rate(publish_rate_);
        double time = 0.0;
        
        while (ros::ok()) {
            // 计算当前转向角度 (正弦波)
            double steering = max_steering_angle_ * sin(2 * M_PI * frequency_ * time);
            
            // 创建并发布速度消息
            geometry_msgs::Twist cmd_vel;
            cmd_vel.linear.x = linear_speed_;
            cmd_vel.angular.z = steering;
            
            cmd_vel_pub_.publish(cmd_vel);
            
            // 更新时间并休眠
            time += 1.0 / publish_rate_;
            rate.sleep();
        }
    }
    
private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_vel_pub_;
    
    double linear_speed_;
    double max_steering_angle_;
    double frequency_;
    double publish_rate_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "snake_drive_node");
    
    SnakeDrive snake_drive;
    snake_drive.run();
    
    return 0;
}