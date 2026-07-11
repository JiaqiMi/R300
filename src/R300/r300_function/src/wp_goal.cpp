#include "ros/ros.h"
#include <Eigen/Eigen>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <GeographicLib/Geocentric.hpp>
#include "move_base_msgs/MoveBaseAction.h"
#include <actionlib/client/simple_action_client.h>
#include "geometry_msgs/PoseStamped.h"
#include <mavros/frame_tf.h>
#include "mavros_msgs/WaypointList.h"
#include "sensor_msgs/NavSatFix.h"

std::vector<geometry_msgs::Pose> localpoint_list;
mavros_msgs::WaypointList wp_list;
Eigen::Vector3d current_gps;
geometry_msgs::Pose pose;
geometry_msgs::PoseStamped local_pose;
int point_num;


typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> mb_action;

void wp_callback(const mavros_msgs::WaypointList::ConstPtr& msg)
{
    wp_list = *msg;
    point_num = msg->waypoints.size();
}

void gp_callback(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    current_gps = {msg->latitude, msg->longitude, msg->altitude};
}

void lp_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    local_pose = *msg;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "waypoint_to_goal");
    ros::NodeHandle n;
    ros::Subscriber wp_sub = n.subscribe<mavros_msgs::WaypointList>("mavros/mission/waypoints", 10, wp_callback);
    ros::Subscriber gp_sub = n.subscribe<sensor_msgs::NavSatFix>("mavros/global_position/global", 10, gp_callback);
    ros::Subscriber lp_sub = n.subscribe<geometry_msgs::PoseStamped> ("mavros/local_position/pose", 10, lp_callback);
    ros::Rate r(10);
    mb_action mb_client("move_base", true);
    move_base_msgs::MoveBaseGoal goal;

    int out_time;
    ros::param::param<int>("~OutTime", out_time, 30);

    //多次读取,排除问题数据

    int wp_flag;
    std::cout << "Please input 1 get waypoints\n";
    std::cin >> wp_flag;
    if(wp_flag != 1)
    {
        ros::shutdown();
    }

    int data_flag;
    while(data_flag <= 20)
    {
        ros::spinOnce();
        data_flag++;
        r.sleep();
    }

    //将MP航点信息转换为本地坐标系的的xyz数据
    for(int i=1; i < point_num; i++)
    {
        GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(), GeographicLib::Constants::WGS84_f());
        Eigen::Vector3d goal_gps(wp_list.waypoints[i].x_lat, wp_list.waypoints[i].y_long, 0);
        Eigen::Vector3d goal_ecef;
        Eigen::Vector3d current_ecef;
        Eigen::Vector3d ecef_offset;
        Eigen::Vector3d enu_offset;

        //将当前位置的经纬度坐标系坐标转换为ecef坐标系的坐标
        earth.Forward(current_gps.x(), current_gps.y(), current_gps.z(), current_ecef.x(), current_ecef.y(), current_ecef.z());
        earth.Forward(goal_gps.x(), goal_gps.y(), goal_gps.z(), goal_ecef.x(), goal_ecef.y(), goal_ecef.z());
        ecef_offset = goal_ecef - current_ecef;
        enu_offset = mavros::ftf::transform_frame_ecef_enu(ecef_offset, current_gps);
        pose.position.x = local_pose.pose.position.x + enu_offset[0];
        pose.position.y = local_pose.pose.position.y + enu_offset[1];
        std::cout << "Point[" << i << "]" << "x is [" << pose.position.x << "]" << std::endl;
        std::cout << "Point[" << i << "]" << "y is [" << pose.position.y << "]" << std::endl;
        pose.orientation = local_pose.pose.orientation;
        localpoint_list.push_back(pose);
    }

    int start_flag;
    std::cout << "Input 1 to start" << std::endl;
    std::cin >> start_flag;
    
    if(start_flag != 1)
    {
        ROS_ERROR("Input error");
        ros::shutdown();
    }

    point_num -= 1;

    int  num = localpoint_list.size();
     ROS_INFO("list_size [%d]   point_num[%d]\n",num,point_num);
    for(int j=0; j < point_num; j++)
    {
        ros::spinOnce();
        goal.target_pose.pose = localpoint_list[j];
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        mb_client.sendGoal(goal);
        ROS_INFO("Go to goal [%d]\n",j+1);

        /*************************************************************
          *     由于MP地面站下发的航点没有航向，
          *     在航点读取时每个目标点航向初始化为local_pose的航向，
          *     每次到达目标点判断航向角误差范围时会存在旋转，
          *     因此提供两种判断是否到达目标点方式，可按需选择
        *************************************************************/
        /*  
        *   1、只判断与目标点的距离，其中
        *   distance要大于等于dwa_local_planner_params.yaml中 #目标点判定x,y轴误差范围 xy_goal_tolerance: 0.4
        *   否则不会下发下一个目标点指令
        *   在计算过程中对数据进行了放大
        */
        int distance;
        do
        {
            int dx = abs((int)(goal.target_pose.pose.position.x * 100 - local_pose.pose.position.x * 100));
            int dy = abs((int)(goal.target_pose.pose.position.y * 100 - local_pose.pose.position.y * 100));
            distance = pow(dx,2) + pow(dy,2);

            ros::spinOnce();
            r.sleep();
        }while(distance > 1600);//距离目标点<0.4m时,视为已到达该目标点
        ROS_INFO("Go to goal [%d] succeese\n",j+1);

        /* 
        *   2、距离和航向角都要判断
        *   采用move_base action判断机制  
         */
        // {
            // bool finish_state = false;
            // finish_state = mb_client.waitForResult(ros::Duration(out_time));
            // ROS_INFO("finish_state [%x]\n",finish_state);
            // if(!finish_state)
            // {
            //     mb_client.cancelGoal();
            //     ROS_WARN("Timed out achieving goal [%d]\n",j+1);
            // }
            // else
            // {
            //     if(mb_client.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
            //     {
            //         ROS_INFO("Go to goal [%d] succeese\n",j+1);
            //     }
            //     else
            //     {
            //         ROS_ERROR("Go to goal [%d] failed\n",j+1);
            //     }
            // }
        // }
        
    }
    return 0;
}

