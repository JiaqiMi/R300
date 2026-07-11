#include <algorithm>
#include <cmath>
#include <string>

#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <std_srvs/Trigger.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

// The guard has exactly one autonomous command source:
//
//     move_base / DWA -> cmd_vel_guard -> base
//
// It does not generate a second angular command and it never cancels move_base.
// The optional heading gate only suppresses FORWARD speed while the current
// waypoint is far outside the vehicle's heading.  DWA's own angular.z is still
// passed through, so the vehicle first pivots using the same DWA command stream
// and then resumes forward motion once the heading is sufficiently aligned.
class CmdVelGuard
{
public:
  CmdVelGuard()
      : nh_(), pnh_("~"), tf_listener_(ros::Duration(10.0)),
        have_raw_cmd_(false), have_odom_(false), have_scan_(false),
        have_goal_(false), enabled_(false), heading_gate_active_(false),
        heading_gate_completed_for_goal_(false),
        last_heading_error_deg_(0.0), last_goal_distance_m_(0.0),
        last_publish_time_(0.0)
  {
    pnh_.param<std::string>("input_topic", input_topic_,
                            std::string("/subject1/cmd_vel_raw"));
    pnh_.param<std::string>("output_topic", output_topic_,
                            std::string("/subject1/cmd_vel_safe"));
    pnh_.param<std::string>("odom_topic", odom_topic_,
                            std::string("/one_x/odom"));
    pnh_.param<std::string>("scan_topic", scan_topic_, std::string("/scan"));
    pnh_.param<std::string>("goal_topic", goal_topic_,
                            std::string("/subject1/current_waypoint_pose"));

    pnh_.param<bool>("require_scan", require_scan_, true);
    pnh_.param<bool>("enabled", enabled_, false);

    pnh_.param<double>("max_linear_mps", max_linear_mps_, 0.35);
    pnh_.param<double>("max_angular_radps", max_angular_radps_, 0.45);
    pnh_.param<double>("max_linear_acc_mps2", max_linear_acc_mps2_, 0.50);
    pnh_.param<double>("max_angular_acc_radps2", max_angular_acc_radps2_, 0.80);
    pnh_.param<double>("command_timeout_s", command_timeout_s_, 0.50);
    pnh_.param<double>("localization_timeout_s", localization_timeout_s_, 0.50);
    pnh_.param<double>("scan_timeout_s", scan_timeout_s_, 0.70);
    pnh_.param<double>("output_rate_hz", output_rate_hz_, 50.0);

    // Heading-gate parameters.  Two thresholds provide hysteresis so that
    // forward motion is not repeatedly enabled/disabled around one angle.
    pnh_.param<bool>("heading_gate_enabled", heading_gate_enabled_, true);
    pnh_.param<double>("heading_gate_enter_deg", heading_gate_enter_deg_, 35.0);
    pnh_.param<double>("heading_gate_exit_deg", heading_gate_exit_deg_, 12.0);
    pnh_.param<double>("heading_gate_min_goal_distance_m",
                       heading_gate_min_goal_distance_m_, 1.20);
    pnh_.param<double>("heading_gate_goal_timeout_s",
                       heading_gate_goal_timeout_s_, 0.0);
    pnh_.param<double>("heading_align_kp_radps_per_rad",
                       heading_align_kp_radps_per_rad_, 1.20);
    pnh_.param<double>("heading_align_min_angular_radps",
                       heading_align_min_angular_radps_, 0.10);
    pnh_.param<double>("heading_align_max_angular_radps",
                       heading_align_max_angular_radps_, 0.35);
    pnh_.param<double>("heading_align_release_hold_s",
                       heading_align_release_hold_s_, 0.40);

    heading_gate_enter_deg_ = std::max(0.0, heading_gate_enter_deg_);
    heading_gate_exit_deg_ = std::max(0.0, heading_gate_exit_deg_);
    if (heading_gate_exit_deg_ >= heading_gate_enter_deg_)
    {
      heading_gate_exit_deg_ = 0.5 * heading_gate_enter_deg_;
      ROS_WARN("heading_gate_exit_deg must be lower than heading_gate_enter_deg; "
               "using %.1f deg", heading_gate_exit_deg_);
    }

    heading_align_kp_radps_per_rad_ =
        std::max(0.0, heading_align_kp_radps_per_rad_);
    heading_align_min_angular_radps_ =
        std::max(0.0, heading_align_min_angular_radps_);
    heading_align_max_angular_radps_ =
        std::max(heading_align_min_angular_radps_,
                 std::min(max_angular_radps_, heading_align_max_angular_radps_));
    heading_align_release_hold_s_ =
        std::max(0.0, heading_align_release_hold_s_);

    raw_sub_ = nh_.subscribe(input_topic_, 10, &CmdVelGuard::RawCmdCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 10, &CmdVelGuard::OdomCallback, this);
    scan_sub_ = nh_.subscribe(scan_topic_, 10, &CmdVelGuard::ScanCallback, this);
    goal_sub_ = nh_.subscribe(goal_topic_, 2, &CmdVelGuard::GoalCallback, this);

    output_pub_ = nh_.advertise<geometry_msgs::Twist>(output_topic_, 10);
    diagnostics_pub_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>(
        "/subject1/cmd_vel_guard/diagnostics", 2);

    enable_service_ = nh_.advertiseService(
        "/subject1/enable_motion", &CmdVelGuard::EnableService, this);
    disable_service_ = nh_.advertiseService(
        "/subject1/disable_motion", &CmdVelGuard::DisableService, this);

    const double period = 1.0 / std::max(1.0, output_rate_hz_);
    timer_ = nh_.createTimer(
        ros::Duration(period), &CmdVelGuard::TimerCallback, this);
    diagnostics_timer_ = nh_.createTimer(
        ros::Duration(1.0), &CmdVelGuard::DiagnosticsTimerCallback, this);

    ROS_INFO_STREAM("cmd_vel_guard source=" << input_topic_
                    << " output=" << output_topic_
                    << " enabled=" << (enabled_ ? "true" : "false")
                    << " direct_move_base_only=true"
                    << " heading_gate=" << (heading_gate_enabled_ ? "true" : "false")
                    << " goal_topic=" << goal_topic_);
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  tf::TransformListener tf_listener_;
  ros::Subscriber raw_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber scan_sub_;
  ros::Subscriber goal_sub_;
  ros::Publisher output_pub_;
  ros::Publisher diagnostics_pub_;
  ros::ServiceServer enable_service_;
  ros::ServiceServer disable_service_;
  ros::Timer timer_;
  ros::Timer diagnostics_timer_;

  std::string input_topic_;
  std::string output_topic_;
  std::string odom_topic_;
  std::string scan_topic_;
  std::string goal_topic_;

  bool require_scan_;
  bool enabled_;
  bool have_raw_cmd_;
  bool have_odom_;
  bool have_scan_;
  bool have_goal_;
  bool heading_gate_enabled_;
  bool heading_gate_active_;
  bool heading_gate_completed_for_goal_;

  geometry_msgs::Twist last_raw_cmd_;
  geometry_msgs::Twist last_output_cmd_;
  nav_msgs::Odometry latest_odom_;
  geometry_msgs::PoseStamped latest_goal_;
  ros::Time raw_cmd_time_;
  ros::Time odom_time_;
  ros::Time scan_time_;
  ros::Time goal_time_;
  ros::Time heading_align_release_candidate_time_;
  ros::Time last_publish_time_;

  double max_linear_mps_;
  double max_angular_radps_;
  double max_linear_acc_mps2_;
  double max_angular_acc_radps2_;
  double command_timeout_s_;
  double localization_timeout_s_;
  double scan_timeout_s_;
  double output_rate_hz_;

  double heading_gate_enter_deg_;
  double heading_gate_exit_deg_;
  double heading_gate_min_goal_distance_m_;
  double heading_gate_goal_timeout_s_;
  double heading_align_kp_radps_per_rad_;
  double heading_align_min_angular_radps_;
  double heading_align_max_angular_radps_;
  double heading_align_release_hold_s_;
  double last_heading_error_deg_;
  double last_goal_distance_m_;

  std::string last_reason_;
  std::string last_source_;

  void RawCmdCallback(const geometry_msgs::TwistConstPtr &msg)
  {
    last_raw_cmd_ = *msg;
    raw_cmd_time_ = ros::Time::now();
    have_raw_cmd_ = true;
  }

  void OdomCallback(const nav_msgs::OdometryConstPtr &msg)
  {
    latest_odom_ = *msg;
    odom_time_ = ros::Time::now();
    have_odom_ = true;
  }

  void ScanCallback(const sensor_msgs::LaserScanConstPtr &)
  {
    scan_time_ = ros::Time::now();
    have_scan_ = true;
  }

  void GoalCallback(const geometry_msgs::PoseStampedConstPtr &msg)
  {
    // waypoint_executor publishes once per active waypoint. Re-arm the
    // one-shot initial alignment only when its coordinate/frame changes.
    bool new_goal = !have_goal_;
    if (!new_goal)
    {
      const double dx = msg->pose.position.x - latest_goal_.pose.position.x;
      const double dy = msg->pose.position.y - latest_goal_.pose.position.y;
      const bool stamp_changed =
          msg->header.stamp != latest_goal_.header.stamp;
      new_goal = stamp_changed ||
                 (msg->header.frame_id != latest_goal_.header.frame_id) ||
                 (std::hypot(dx, dy) > 0.05);
    }

    latest_goal_ = *msg;
    goal_time_ = ros::Time::now();
    have_goal_ = true;

    if (new_goal)
    {
      heading_gate_active_ = false;
      heading_gate_completed_for_goal_ = false;
      heading_align_release_candidate_time_ = ros::Time(0);
      ROS_INFO("heading alignment armed for new goal: frame=%s x=%.3f y=%.3f",
               latest_goal_.header.frame_id.c_str(),
               latest_goal_.pose.position.x,
               latest_goal_.pose.position.y);
    }
  }

  bool EnableService(std_srvs::Trigger::Request &,
                     std_srvs::Trigger::Response &res)
  {
    enabled_ = true;
    res.success = true;
    res.message =
        "Motion enabled. The guard will still stop the vehicle if "
        "command/localization/scan freshness checks fail.";
    ROS_WARN("Subject-1 motion enabled by service");
    return true;
  }

  bool DisableService(std_srvs::Trigger::Request &,
                      std_srvs::Trigger::Response &res)
  {
    enabled_ = false;
    heading_gate_active_ = false;
    heading_align_release_candidate_time_ = ros::Time(0);
    PublishImmediateStop();
    res.success = true;
    res.message = "Motion disabled and zero velocity published.";
    ROS_WARN("Subject-1 motion disabled by service");
    return true;
  }

  static double Clamp(double value, double lower, double upper)
  {
    return std::max(lower, std::min(upper, value));
  }

  static double WrapPi(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  bool IsFresh(const ros::Time &now, const ros::Time &stamp,
               bool have_message, double timeout_s) const
  {
    return have_message && !stamp.isZero() &&
           (now - stamp).toSec() <= timeout_s;
  }

  bool InputsHealthy(const ros::Time &now, std::string *reason) const
  {
    if (!enabled_)
    {
      *reason = "motion_disabled";
      return false;
    }
    if (!IsFresh(now, raw_cmd_time_, have_raw_cmd_, command_timeout_s_))
    {
      *reason = "cmd_vel_timeout";
      return false;
    }
    if (!IsFresh(now, odom_time_, have_odom_, localization_timeout_s_))
    {
      *reason = "one_x_odom_timeout";
      return false;
    }
    if (require_scan_ && !IsFresh(now, scan_time_, have_scan_, scan_timeout_s_))
    {
      *reason = "scan_timeout";
      return false;
    }

    *reason = "healthy";
    return true;
  }

  bool GetGoalInOdomFrame(double *goal_x, double *goal_y)
  {
    if (!have_goal_ || !have_odom_)
    {
      return false;
    }

    const ros::Time now = ros::Time::now();
    // waypoint_executor publishes the active goal as a latched message once.
    // A non-positive timeout therefore means "keep the latched goal valid
    // until it is replaced", which is the normal waypoint-navigation mode.
    if (heading_gate_goal_timeout_s_ > 0.0 &&
        !IsFresh(now, goal_time_, have_goal_, heading_gate_goal_timeout_s_))
    {
      return false;
    }

    const std::string odom_frame = latest_odom_.header.frame_id.empty()
                                       ? std::string("odom")
                                       : latest_odom_.header.frame_id;
    const std::string goal_frame = latest_goal_.header.frame_id;
    if (goal_frame.empty())
    {
      ROS_WARN_THROTTLE(1.0, "heading gate received a goal with an empty frame_id");
      return false;
    }

    try
    {
      tf::StampedTransform transform;
      tf_listener_.lookupTransform(odom_frame, goal_frame, ros::Time(0), transform);
      tf::Pose goal_pose;
      tf::poseMsgToTF(latest_goal_.pose, goal_pose);
      const tf::Pose goal_in_odom = transform * goal_pose;
      *goal_x = goal_in_odom.getOrigin().x();
      *goal_y = goal_in_odom.getOrigin().y();
      return true;
    }
    catch (const tf::TransformException &ex)
    {
      ROS_WARN_THROTTLE(1.0, "heading gate cannot transform goal %s -> %s: %s",
                        goal_frame.c_str(), odom_frame.c_str(), ex.what());
      return false;
    }
  }

  bool ApplyHeadingGate(geometry_msgs::Twist *desired)
  {
    if (!heading_gate_enabled_)
    {
      heading_gate_active_ = false;
      heading_gate_completed_for_goal_ = false;
      heading_align_release_candidate_time_ = ros::Time(0);
      return false;
    }

    double goal_x = 0.0;
    double goal_y = 0.0;
    if (!GetGoalInOdomFrame(&goal_x, &goal_y))
    {
      // A manual /move_base_simple/goal does not update goal_topic. In that
      // case, do not alter DWA output.
      heading_gate_active_ = false;
      heading_align_release_candidate_time_ = ros::Time(0);
      return false;
    }

    const double robot_x = latest_odom_.pose.pose.position.x;
    const double robot_y = latest_odom_.pose.pose.position.y;
    const double dx = goal_x - robot_x;
    const double dy = goal_y - robot_y;
    const double distance = std::hypot(dx, dy);
    last_goal_distance_m_ = distance;

    if (distance < heading_gate_min_goal_distance_m_)
    {
      heading_gate_active_ = false;
      heading_gate_completed_for_goal_ = true;
      heading_align_release_candidate_time_ = ros::Time(0);
      last_heading_error_deg_ = 0.0;
      return false;
    }

    const double current_yaw = tf::getYaw(latest_odom_.pose.pose.orientation);
    const double goal_bearing = std::atan2(dy, dx);
    const double heading_error = WrapPi(goal_bearing - current_yaw);
    const double deg_per_rad = 180.0 / 3.14159265358979323846;
    const double abs_error_deg = std::fabs(heading_error) * deg_per_rad;
    last_heading_error_deg_ = heading_error * deg_per_rad;

    // ONE-SHOT per waypoint. The previous implementation discarded DWA's
    // linear.x but kept its angular.z. DWA was therefore planning a forward
    // arc that the robot never executed, which makes raw angular.z alternate
    // and can keep the gate active indefinitely.
    if (!heading_gate_completed_for_goal_ && !heading_gate_active_ &&
        abs_error_deg >= heading_gate_enter_deg_)
    {
      heading_gate_active_ = true;
      heading_align_release_candidate_time_ = ros::Time(0);
      ROS_INFO("initial heading alignment entered: goal distance=%.2f m, "
               "heading error=%.1f deg",
               distance, last_heading_error_deg_);
    }

    if (!heading_gate_active_)
    {
      return false;
    }

    // Release after the attitude stays within tolerance briefly; then DWA
    // receives full and uninterrupted motion authority for this waypoint.
    if (abs_error_deg <= heading_gate_exit_deg_)
    {
      const ros::Time now = ros::Time::now();
      if (heading_align_release_candidate_time_.isZero())
      {
        heading_align_release_candidate_time_ = now;
      }
      if ((now - heading_align_release_candidate_time_).toSec() >=
          heading_align_release_hold_s_)
      {
        heading_gate_active_ = false;
        heading_gate_completed_for_goal_ = true;
        heading_align_release_candidate_time_ = ros::Time(0);
        ROS_INFO("initial heading alignment released: goal distance=%.2f m, "
                 "heading error=%.1f deg; DWA linear.x is passed through",
                 distance, last_heading_error_deg_);
        return false;
      }
    }
    else
    {
      heading_align_release_candidate_time_ = ros::Time(0);
    }

    // A direct yaw-only P controller is required while forward motion is
    // blocked. ROS +angular.z is CCW, matching positive bearing-minus-yaw.
    double turn_cmd = heading_align_kp_radps_per_rad_ * heading_error;
    turn_cmd = Clamp(turn_cmd, -heading_align_max_angular_radps_,
                     heading_align_max_angular_radps_);
    if (std::fabs(turn_cmd) < heading_align_min_angular_radps_)
    {
      turn_cmd = (heading_error >= 0.0 ? 1.0 : -1.0) *
                 heading_align_min_angular_radps_;
    }

    desired->linear.x = 0.0;
    desired->angular.z = turn_cmd;
    return true;
  }

  void TimerCallback(const ros::TimerEvent &)
  {
    const ros::Time now = ros::Time::now();
    std::string reason;
    geometry_msgs::Twist desired;

    if (InputsHealthy(now, &reason))
    {
      // Scout Mini/R300 is non-holonomic.  Do not pass lateral velocity on.
      desired.linear.x = Clamp(last_raw_cmd_.linear.x, 0.0, max_linear_mps_);
      desired.angular.z = Clamp(last_raw_cmd_.angular.z,
                                -max_angular_radps_, max_angular_radps_);

      const bool heading_gate_turning = ApplyHeadingGate(&desired);
      if (heading_gate_turning)
      {
        reason = "initial_heading_align";
        last_source_ = "heading_align_controller";
      }
      else
      {
        last_source_ = "move_base";
      }

      const double dt = last_publish_time_.isZero()
                            ? 0.0
                            : (now - last_publish_time_).toSec();
      if (dt > 0.0 && dt < 1.0)
      {
        const double max_dw = max_angular_acc_radps2_ * dt;
        // A safety-induced stop of forward motion must be immediate.  When the
        // gate is inactive, normal acceleration limiting is retained.
        if (heading_gate_turning)
        {
          desired.linear.x = 0.0;
        }
        else
        {
          const double max_dv = max_linear_acc_mps2_ * dt;
          desired.linear.x = Clamp(desired.linear.x,
                                   last_output_cmd_.linear.x - max_dv,
                                   last_output_cmd_.linear.x + max_dv);
        }
        desired.angular.z = Clamp(desired.angular.z,
                                  last_output_cmd_.angular.z - max_dw,
                                  last_output_cmd_.angular.z + max_dw);
      }
    }
    else
    {
      desired.linear.x = 0.0;
      desired.angular.z = 0.0;
      heading_gate_active_ = false;
      last_source_ = "stop";
    }

    output_pub_.publish(desired);
    last_output_cmd_ = desired;
    last_publish_time_ = now;
    last_reason_ = reason;
  }

  void PublishImmediateStop()
  {
    geometry_msgs::Twist stop;
    output_pub_.publish(stop);
    last_output_cmd_ = stop;
    last_source_ = "stop";
  }

  void DiagnosticsTimerCallback(const ros::TimerEvent &)
  {
    diagnostic_msgs::DiagnosticArray array_msg;
    array_msg.header.stamp = ros::Time::now();

    diagnostic_msgs::DiagnosticStatus status;
    status.name = "Subject-1 cmd_vel guard";
    status.hardware_id = "R300";
    status.level = (last_reason_ == "healthy" || last_reason_ == "initial_heading_align")
                       ? diagnostic_msgs::DiagnosticStatus::OK
                       : diagnostic_msgs::DiagnosticStatus::WARN;
    status.message = last_reason_.empty() ? "waiting_for_inputs" : last_reason_;

    auto add_kv = [&status](const std::string &key, const std::string &value)
    {
      diagnostic_msgs::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

    add_kv("enabled", enabled_ ? "true" : "false");
    add_kv("require_scan", require_scan_ ? "true" : "false");
    add_kv("active_source", last_source_);
    add_kv("heading_gate_enabled", heading_gate_enabled_ ? "true" : "false");
    add_kv("heading_gate_active", heading_gate_active_ ? "true" : "false");
    add_kv("heading_align_completed_for_goal",
           heading_gate_completed_for_goal_ ? "true" : "false");
    add_kv("heading_error_deg", std::to_string(last_heading_error_deg_));
    add_kv("goal_distance_m", std::to_string(last_goal_distance_m_));
    add_kv("last_output_linear_mps",
           std::to_string(last_output_cmd_.linear.x));
    add_kv("last_output_angular_radps",
           std::to_string(last_output_cmd_.angular.z));

    array_msg.status.push_back(status);
    diagnostics_pub_.publish(array_msg);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cmd_vel_guard");
  CmdVelGuard guard;
  ros::spin();
  return 0;
}
