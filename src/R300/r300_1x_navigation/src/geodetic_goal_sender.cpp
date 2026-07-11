#include <cmath>
#include <memory>
#include <stdexcept>
#include <limits>
#include <string>

#include <actionlib/client/simple_action_client.h>
#include <boost/bind.hpp>
#include <move_base_msgs/MoveBaseAction.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_srvs/Trigger.h>
#include <tf/transform_datatypes.h>

#include "r300_1x_navigation/geodesy.hpp"

class GeodeticGoalSender
{
public:
  using MoveBaseClient = actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction>;

  GeodeticGoalSender()
      : nh_(), pnh_("~"), action_client_(nullptr), have_origin_(false), have_odom_(false),
        start_requested_(false), goal_sent_(false)
  {
    pnh_.param<double>("target_latitude_deg", target_.lat_deg, std::numeric_limits<double>::quiet_NaN());
    pnh_.param<double>("target_longitude_deg", target_.lon_deg, std::numeric_limits<double>::quiet_NaN());
    pnh_.param<double>("target_altitude_m", target_.alt_m, 0.0);
    pnh_.param<double>("max_goal_distance_m", max_goal_distance_m_, 150.0);
    pnh_.param<bool>("auto_start", start_requested_, false);
    pnh_.param<std::string>("origin_topic", origin_topic_, std::string("/one_x/origin"));
    pnh_.param<std::string>("odom_topic", odom_topic_, std::string("/one_x/odom"));
    pnh_.param<std::string>("move_base_action", action_name_, std::string("/move_base"));
    pnh_.param<std::string>("goal_frame", goal_frame_, std::string("map"));

    if (!IsValidTarget())
    {
      throw std::runtime_error("target_latitude_deg and target_longitude_deg must be supplied as valid WGS-84 degrees");
    }
    if (max_goal_distance_m_ <= 0.0)
    {
      throw std::runtime_error("max_goal_distance_m must be positive");
    }

    origin_sub_ = nh_.subscribe(origin_topic_, 1, &GeodeticGoalSender::OriginCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 10, &GeodeticGoalSender::OdomCallback, this);
    start_service_ = nh_.advertiseService("/subject1/start_goal", &GeodeticGoalSender::StartService, this);
    cancel_service_ = nh_.advertiseService("/subject1/cancel_goal", &GeodeticGoalSender::CancelService, this);
    action_client_.reset(new MoveBaseClient(action_name_, true));
    timer_ = nh_.createTimer(ros::Duration(0.25), &GeodeticGoalSender::TimerCallback, this);

    ROS_INFO_STREAM("Subject-1 target accepted: lat=" << target_.lat_deg << ", lon=" << target_.lon_deg
                                                        << ", auto_start=" << (start_requested_ ? "true" : "false"));
    if (!start_requested_)
    {
      ROS_INFO("Goal is armed but will not be sent until: rosservice call /subject1/start_goal");
    }
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber origin_sub_;
  ros::Subscriber odom_sub_;
  ros::ServiceServer start_service_;
  ros::ServiceServer cancel_service_;
  ros::Timer timer_;
  std::unique_ptr<MoveBaseClient> action_client_;

  r300_1x_navigation::Geodetic target_;
  r300_1x_navigation::Geodetic origin_;
  nav_msgs::Odometry latest_odom_;
  bool have_origin_;
  bool have_odom_;
  bool start_requested_;
  bool goal_sent_;
  double max_goal_distance_m_;
  std::string origin_topic_;
  std::string odom_topic_;
  std::string action_name_;
  std::string goal_frame_;

  bool IsValidTarget() const
  {
    return std::isfinite(target_.lat_deg) && std::isfinite(target_.lon_deg) &&
           std::fabs(target_.lat_deg) <= 90.0 && std::fabs(target_.lon_deg) <= 180.0;
  }

  void OriginCallback(const sensor_msgs::NavSatFixConstPtr &msg)
  {
    if (!std::isfinite(msg->latitude) || !std::isfinite(msg->longitude) ||
        std::fabs(msg->latitude) > 90.0 || std::fabs(msg->longitude) > 180.0)
    {
      ROS_WARN_THROTTLE(1.0, "Ignoring invalid 1X origin message");
      return;
    }
    origin_.lat_deg = msg->latitude;
    origin_.lon_deg = msg->longitude;
    origin_.alt_m = msg->altitude;
    have_origin_ = true;
  }

  void OdomCallback(const nav_msgs::OdometryConstPtr &msg)
  {
    latest_odom_ = *msg;
    have_odom_ = true;
  }

  bool StartService(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    if (goal_sent_)
    {
      res.success = false;
      res.message = "A goal has already been sent; call /subject1/cancel_goal before sending another.";
      return true;
    }
    start_requested_ = true;
    res.success = true;
    res.message = "Goal start requested. It will be sent after 1X origin, odometry, and move_base are ready.";
    return true;
  }

  bool CancelService(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    action_client_->cancelAllGoals();
    start_requested_ = false;
    goal_sent_ = false;
    res.success = true;
    res.message = "move_base goal cancelled; start request cleared.";
    ROS_WARN("Subject-1 geographic goal cancelled");
    return true;
  }

  void TimerCallback(const ros::TimerEvent &)
  {
    if (!start_requested_ || goal_sent_)
    {
      return;
    }
    if (!have_origin_ || !have_odom_)
    {
      ROS_INFO_THROTTLE(2.0, "Waiting for 1X origin and /one_x/odom before sending geographic goal");
      return;
    }
    if (!action_client_->waitForServer(ros::Duration(0.05)))
    {
      ROS_INFO_THROTTLE(2.0, "Waiting for move_base action server: %s", action_name_.c_str());
      return;
    }

    const r300_1x_navigation::Enu enu = r300_1x_navigation::GeodeticToEnu(target_, origin_);
    const double distance_m = std::hypot(enu.east, enu.north);
    if (!std::isfinite(distance_m) || distance_m > max_goal_distance_m_)
    {
      ROS_ERROR("Refusing geographic goal: distance %.2f m exceeds configured max_goal_distance_m %.2f m",
                distance_m, max_goal_distance_m_);
      start_requested_ = false;
      return;
    }

    const double dx = enu.east - latest_odom_.pose.pose.position.x;
    const double dy = enu.north - latest_odom_.pose.pose.position.y;
    const double approach_yaw = std::atan2(dy, dx);

    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.header.frame_id = goal_frame_;
    goal.target_pose.pose.position.x = enu.east;
    goal.target_pose.pose.position.y = enu.north;
    goal.target_pose.pose.position.z = 0.0;
    goal.target_pose.pose.orientation = tf::createQuaternionMsgFromYaw(approach_yaw);

    action_client_->sendGoal(goal,
                             boost::bind(&GeodeticGoalSender::DoneCallback, this, _1, _2));
    goal_sent_ = true;
    ROS_INFO("Sent geographic target to move_base: ENU east=%.3f m north=%.3f m distance=%.3f m",
             enu.east, enu.north, distance_m);
  }

  void DoneCallback(const actionlib::SimpleClientGoalState &state,
                    const move_base_msgs::MoveBaseResultConstPtr &)
  {
    ROS_INFO_STREAM("Subject-1 geographic goal completed with move_base state: " << state.toString());
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "geodetic_goal_sender");
  try
  {
    GeodeticGoalSender sender;
    ros::spin();
  }
  catch (const std::exception &e)
  {
    ROS_FATAL("geodetic_goal_sender failed: %s", e.what());
    return 1;
  }
  return 0;
}
