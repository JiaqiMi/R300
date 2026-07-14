#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <geometry_msgs/TransformStamped.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "r300_1x_navigation/vision_snapshot_layer.hpp"

namespace r300_1x_navigation
{

VisionSnapshotLayer::VisionSnapshotLayer()
  : nh_(),
    hold_time_s_(1.0),
    expected_update_rate_s_(0.5),
    max_message_age_s_(0.5),
    transform_timeout_s_(0.10),
    min_range_m_(0.20),
    max_range_m_(10.0),
    dedup_resolution_m_(0.1),
    active_scan_angle_increment_(0.5 * M_PI / 180.0),
    stop_on_stale_(true),
    publish_active_scan_(true),
    accepted_scan_count_(0U),
    dropped_scan_count_(0U)
{
}

void VisionSnapshotLayer::onInitialize()
{
  ros::NodeHandle private_nh("~/" + name_);
  nh_ = ros::NodeHandle();

  global_frame_ = layered_costmap_->getGlobalFrameID();
  default_value_ = costmap_2d::FREE_SPACE;

  private_nh.param("enabled", enabled_, true);
  private_nh.param("topic", topic_, std::string("/r300_vision/obstacle_scan"));
  private_nh.param("hold_time_s", hold_time_s_, 1.0);
  private_nh.param("expected_update_rate_s", expected_update_rate_s_, 0.5);
  private_nh.param("max_message_age_s", max_message_age_s_, 0.5);
  private_nh.param("transform_timeout_s", transform_timeout_s_, 0.10);
  private_nh.param("min_range_m", min_range_m_, 0.20);
  private_nh.param("max_range_m", max_range_m_, 10.0);
  private_nh.param("dedup_resolution_m", dedup_resolution_m_, 0.05);
  private_nh.param("stop_on_stale", stop_on_stale_, true);
  private_nh.param("publish_active_scan", publish_active_scan_, true);
  private_nh.param("active_scan_topic", active_scan_topic_,
                   std::string("/r300_vision/active_obstacle_scan"));
  private_nh.param("active_scan_frame", active_scan_frame_, std::string("base_link"));

  double active_scan_increment_deg = 0.5;
  private_nh.param("active_scan_angle_increment_deg", active_scan_increment_deg, 0.5);
  active_scan_angle_increment_ = active_scan_increment_deg * M_PI / 180.0;

  if (hold_time_s_ < 0.0 || expected_update_rate_s_ < 0.0 ||
      max_message_age_s_ <= 0.0 || transform_timeout_s_ < 0.0 ||
      min_range_m_ < 0.0 || max_range_m_ <= min_range_m_ ||
      dedup_resolution_m_ <= 0.0 || active_scan_angle_increment_ <= 0.0)
  {
    throw std::runtime_error("Invalid VisionSnapshotLayer parameters");
  }

  matchSize();
  current_ = true;

  scan_sub_ = nh_.subscribe(topic_, 20, &VisionSnapshotLayer::scanCallback, this);
  if (publish_active_scan_)
  {
    active_scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>(active_scan_topic_, 2);
  }

  ROS_INFO("VisionSnapshotLayer ready: topic=%s global_frame=%s hold=%.2fs max_range=%.2fm",
           topic_.c_str(), global_frame_.c_str(), hold_time_s_, max_range_m_);
}

void VisionSnapshotLayer::matchSize()
{
  CostmapLayer::matchSize();
  resetMaps();
}

std::int64_t VisionSnapshotLayer::makeKey(double x, double y) const
{
  const std::int32_t qx = static_cast<std::int32_t>(std::floor(x / dedup_resolution_m_));
  const std::int32_t qy = static_cast<std::int32_t>(std::floor(y / dedup_resolution_m_));
  const std::uint64_t ux = static_cast<std::uint32_t>(qx);
  const std::uint64_t uy = static_cast<std::uint32_t>(qy);
  return static_cast<std::int64_t>((ux << 32U) | uy);
}

void VisionSnapshotLayer::scanCallback(const sensor_msgs::LaserScanConstPtr& msg)
{
  if (!enabled_ || !msg)
  {
    return;
  }

  const ros::Time receive_time = ros::Time::now();
  ros::Time stamp = msg->header.stamp;
  if (stamp.isZero())
  {
    stamp = receive_time;
  }

  const double age = (receive_time - stamp).toSec();
  if (age > max_message_age_s_ || age < -0.10)
  {
    ++dropped_scan_count_;
    ROS_WARN_THROTTLE(2.0,
                      "VisionSnapshotLayer dropping stale/future scan: age=%.3fs",
                      age);
    return;
  }

  geometry_msgs::TransformStamped transform_msg;
  try
  {
    transform_msg = tf_->lookupTransform(global_frame_, msg->header.frame_id,
                                         stamp, ros::Duration(transform_timeout_s_));
  }
  catch (const tf2::TransformException& ex)
  {
    ++dropped_scan_count_;
    ROS_WARN_THROTTLE(2.0,
                      "VisionSnapshotLayer TF failed %s <- %s: %s",
                      global_frame_.c_str(), msg->header.frame_id.c_str(), ex.what());
    return;
  }

  tf2::Transform global_from_scan;
  tf2::fromMsg(transform_msg.transform, global_from_scan);

  std::vector<TimedPoint> frame_points;
  frame_points.reserve(msg->ranges.size());

  double angle = msg->angle_min;
  const double usable_min = std::max<double>(msg->range_min, min_range_m_);
  const double usable_max = std::min<double>(msg->range_max, max_range_m_);
  const ros::Time expiry = receive_time + ros::Duration(hold_time_s_);

  for (const float range : msg->ranges)
  {
    if (std::isfinite(range) && range >= usable_min && range <= usable_max)
    {
      const tf2::Vector3 local_point(range * std::cos(angle),
                                    range * std::sin(angle), 0.0);
      const tf2::Vector3 global_point = global_from_scan * local_point;
      TimedPoint point;
      point.x = global_point.x();
      point.y = global_point.y();
      point.expiry = expiry;
      frame_points.push_back(point);
    }
    angle += msg->angle_increment;
  }

  std::size_t active_count = 0U;
  {
    // std::lock_guard<std::mutex> lock(mutex_);
    // last_scan_receive_time_ = receive_time;
    // purgeExpiredLocked(receive_time);
    // for (const TimedPoint& point : frame_points)
    // {
    //   active_points_[makeKey(point.x, point.y)] = point;
    // }
    // active_count = active_points_.size();

    std::lock_guard<std::mutex> lock(mutex_);
    last_scan_receive_time_ = receive_time;
    purgeExpiredLocked(receive_time);

    // 收到新的非空视觉结果时，用当前完整快照替换旧快照。
    // 这样仍然保持1秒，但不会累计最近10帧形成拖影。
    if (!frame_points.empty())
    {
      active_points_.clear();

      for (const TimedPoint& point : frame_points)
      {
        active_points_[makeKey(point.x, point.y)] = point;
      }
    }

    // 如果当前帧为空，不立刻清空。
    // 原快照继续保留，直到 hold_time_s 到期。
    active_count = active_points_.size();

  }

  ++accepted_scan_count_;
  current_ = true;

  ROS_INFO_THROTTLE(1.0,
                    "VisionSnapshotLayer scan accepted: points=%zu active=%zu age=%.3fs",
                    frame_points.size(), active_count, age);
}

void VisionSnapshotLayer::purgeExpiredLocked(const ros::Time& now)
{
  for (auto it = active_points_.begin(); it != active_points_.end(); )
  {
    if (it->second.expiry <= now)
    {
      it = active_points_.erase(it);
    }
    else
    {
      ++it;
    }
  }
}

bool VisionSnapshotLayer::sourceIsCurrent(const ros::Time& now) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (expected_update_rate_s_ <= 0.0)
  {
    return true;
  }
  if (last_scan_receive_time_.isZero())
  {
    return false;
  }
  return (now - last_scan_receive_time_).toSec() <= expected_update_rate_s_;
}

void VisionSnapshotLayer::publishActiveScan(double robot_x, double robot_y,
                                            double robot_yaw,
                                            const ros::Time& stamp)
{
  if (!publish_active_scan_ || !active_scan_pub_)
  {
    return;
  }

  const int beam_count = static_cast<int>(std::round((2.0 * M_PI) /
                                                      active_scan_angle_increment_));
  sensor_msgs::LaserScan scan;
  scan.header.stamp = stamp;
  scan.header.frame_id = active_scan_frame_;
  scan.angle_min = -M_PI;
  scan.angle_increment = active_scan_angle_increment_;
  scan.angle_max = scan.angle_min + (beam_count - 1) * scan.angle_increment;
  scan.time_increment = 0.0;
  scan.scan_time = 0.0;
  scan.range_min = min_range_m_;
  scan.range_max = max_range_m_;
  scan.ranges.assign(static_cast<std::size_t>(beam_count),
                     std::numeric_limits<float>::infinity());

  const double c = std::cos(robot_yaw);
  const double s = std::sin(robot_yaw);

  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& item : active_points_)
  {
    const double dx = item.second.x - robot_x;
    const double dy = item.second.y - robot_y;
    const double bx = c * dx + s * dy;
    const double by = -s * dx + c * dy;
    const double range = std::hypot(bx, by);
    if (range < min_range_m_ || range > max_range_m_)
    {
      continue;
    }

    double angle = std::atan2(by, bx);
    int index = static_cast<int>(std::round((angle - scan.angle_min) /
                                            scan.angle_increment));
    index %= beam_count;
    if (index < 0)
    {
      index += beam_count;
    }
    scan.ranges[static_cast<std::size_t>(index)] = std::min(
        scan.ranges[static_cast<std::size_t>(index)], static_cast<float>(range));
  }

  active_scan_pub_.publish(scan);
}

void VisionSnapshotLayer::updateBounds(double robot_x, double robot_y,
                                       double robot_yaw,
                                       double* min_x, double* min_y,
                                       double* max_x, double* max_y)
{
  if (!enabled_)
  {
    return;
  }

  if (layered_costmap_->isRolling())
  {
    updateOrigin(robot_x - getSizeInMetersX() / 2.0,
                 robot_y - getSizeInMetersY() / 2.0);
  }

  const ros::Time now = ros::Time::now();
  const bool source_current = sourceIsCurrent(now);
  current_ = source_current;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    // When the source is stale, retain the last obstacles and report current_=false.
    // move_base then fails safe instead of silently clearing the world.
    if (source_current || !stop_on_stale_)
    {
      purgeExpiredLocked(now);
    }

    // Snapshot semantics: rebuild this entire visual layer every costmap cycle.
    // This guarantees that expired obstacle cells cannot remain behind.
    resetMaps();

    for (const auto& item : active_points_)
    {
      unsigned int mx = 0U;
      unsigned int my = 0U;
      if (worldToMap(item.second.x, item.second.y, mx, my))
      {
        setCost(mx, my, costmap_2d::LETHAL_OBSTACLE);
      }
    }
  }

  // Force the master costmap to refresh the whole local window.  The current
  // map is only 240x240 cells, so this deterministic reset is inexpensive and
  // avoids the stale-bounds failure mode of raytrace-based clearing.
  *min_x = std::min(*min_x, getOriginX());
  *min_y = std::min(*min_y, getOriginY());
  *max_x = std::max(*max_x, getOriginX() + getSizeInMetersX());
  *max_y = std::max(*max_y, getOriginY() + getSizeInMetersY());

  publishActiveScan(robot_x, robot_y, robot_yaw, now);
}

void VisionSnapshotLayer::updateCosts(costmap_2d::Costmap2D& master_grid,
                                      int min_i, int min_j,
                                      int max_i, int max_j)
{
  if (!enabled_)
  {
    return;
  }
  updateWithMax(master_grid, min_i, min_j, max_i, max_j);
}

void VisionSnapshotLayer::reset()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_points_.clear();
    last_scan_receive_time_ = ros::Time(0);
  }
  resetMaps();
  current_ = true;
}

}  // namespace r300_1x_navigation

PLUGINLIB_EXPORT_CLASS(r300_1x_navigation::VisionSnapshotLayer, costmap_2d::Layer)
