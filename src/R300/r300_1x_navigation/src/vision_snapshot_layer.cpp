#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/TransformStamped.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "r300_1x_navigation/vision_snapshot_layer.hpp"

namespace r300_1x_navigation
{

VisionSnapshotLayer::VisionSnapshotLayer()
  : nh_(),
    hold_time_s_(5.0),
    expected_update_rate_s_(0.0),
    max_message_age_s_(0.5),
    transform_timeout_s_(0.10),
    min_range_m_(0.20),
    max_range_m_(10.0),
    dedup_resolution_m_(0.05),
    active_scan_angle_increment_(0.5 * M_PI / 180.0),
    cluster_tolerance_m_(0.45),
    association_distance_m_(0.80),
    min_cluster_points_(1U),
    stop_on_stale_(false),
    publish_active_scan_(true),
    next_track_id_(1U),
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
  private_nh.param("hold_time_s", hold_time_s_, 5.0);
  private_nh.param("expected_update_rate_s", expected_update_rate_s_, 0.0);
  private_nh.param("max_message_age_s", max_message_age_s_, 0.5);
  private_nh.param("transform_timeout_s", transform_timeout_s_, 0.10);
  private_nh.param("min_range_m", min_range_m_, 0.20);
  private_nh.param("max_range_m", max_range_m_, 10.0);
  private_nh.param("dedup_resolution_m", dedup_resolution_m_, 0.05);
  private_nh.param("cluster_tolerance_m", cluster_tolerance_m_, 0.45);
  private_nh.param("association_distance_m", association_distance_m_, 0.80);

  int min_cluster_points = 1;
  private_nh.param("min_cluster_points", min_cluster_points, 1);
  min_cluster_points_ = static_cast<std::size_t>(
      std::max(1, min_cluster_points));

  private_nh.param("stop_on_stale", stop_on_stale_, false);
  private_nh.param("publish_active_scan", publish_active_scan_, true);
  private_nh.param("active_scan_topic", active_scan_topic_,
                   std::string("/r300_vision/active_obstacle_scan"));
  private_nh.param("active_scan_frame", active_scan_frame_,
                   std::string("base_link"));

  double active_scan_increment_deg = 0.5;
  private_nh.param("active_scan_angle_increment_deg",
                   active_scan_increment_deg, 0.5);
  active_scan_angle_increment_ =
      active_scan_increment_deg * M_PI / 180.0;

  if (hold_time_s_ < 0.0 || expected_update_rate_s_ < 0.0 ||
      max_message_age_s_ <= 0.0 || transform_timeout_s_ < 0.0 ||
      min_range_m_ < 0.0 || max_range_m_ <= min_range_m_ ||
      dedup_resolution_m_ <= 0.0 ||
      active_scan_angle_increment_ <= 0.0 ||
      cluster_tolerance_m_ <= 0.0 ||
      association_distance_m_ <= 0.0)
  {
    throw std::runtime_error("Invalid VisionSnapshotLayer parameters");
  }

  matchSize();
  current_ = true;

  scan_sub_ = nh_.subscribe(
      topic_, 20, &VisionSnapshotLayer::scanCallback, this);

  if (publish_active_scan_)
  {
    active_scan_pub_ =
        nh_.advertise<sensor_msgs::LaserScan>(active_scan_topic_, 2);
  }

  ROS_INFO(
      "VisionSnapshotLayer ready: topic=%s global_frame=%s "
      "hold=%.2fs cluster=%.2fm association=%.2fm",
      topic_.c_str(), global_frame_.c_str(), hold_time_s_,
      cluster_tolerance_m_, association_distance_m_);
}

void VisionSnapshotLayer::matchSize()
{
  CostmapLayer::matchSize();
  resetMaps();
}

std::int64_t VisionSnapshotLayer::makeKey(double x, double y) const
{
  const std::int32_t qx = static_cast<std::int32_t>(
      std::floor(x / dedup_resolution_m_));
  const std::int32_t qy = static_cast<std::int32_t>(
      std::floor(y / dedup_resolution_m_));
  const std::uint64_t ux = static_cast<std::uint32_t>(qx);
  const std::uint64_t uy = static_cast<std::uint32_t>(qy);

  return static_cast<std::int64_t>((ux << 32U) | uy);
}

std::vector<VisionSnapshotLayer::FrameCluster>
VisionSnapshotLayer::clusterFramePoints(
    const std::vector<TimedPoint>& input_points) const
{
  std::unordered_map<std::int64_t, TimedPoint> deduplicated;
  deduplicated.reserve(input_points.size());

  for (const TimedPoint& point : input_points)
  {
    deduplicated[makeKey(point.x, point.y)] = point;
  }

  std::vector<TimedPoint> points;
  points.reserve(deduplicated.size());

  for (const auto& item : deduplicated)
  {
    points.push_back(item.second);
  }

  std::vector<FrameCluster> clusters;
  std::vector<bool> visited(points.size(), false);
  const double tolerance_sq =
      cluster_tolerance_m_ * cluster_tolerance_m_;

  for (std::size_t seed = 0U; seed < points.size(); ++seed)
  {
    if (visited[seed])
    {
      continue;
    }

    visited[seed] = true;
    std::vector<std::size_t> pending(1U, seed);
    FrameCluster cluster;

    while (!pending.empty())
    {
      const std::size_t index = pending.back();
      pending.pop_back();

      cluster.points.push_back(points[index]);

      for (std::size_t candidate = 0U;
           candidate < points.size(); ++candidate)
      {
        if (visited[candidate])
        {
          continue;
        }

        const double dx =
            points[candidate].x - points[index].x;
        const double dy =
            points[candidate].y - points[index].y;

        if (dx * dx + dy * dy <= tolerance_sq)
        {
          visited[candidate] = true;
          pending.push_back(candidate);
        }
      }
    }

    if (cluster.points.size() < min_cluster_points_)
    {
      continue;
    }

    double sum_x = 0.0;
    double sum_y = 0.0;

    for (const TimedPoint& point : cluster.points)
    {
      sum_x += point.x;
      sum_y += point.y;
    }

    const double count =
        static_cast<double>(cluster.points.size());
    cluster.centroid_x = sum_x / count;
    cluster.centroid_y = sum_y / count;
    clusters.push_back(cluster);
  }

  // 大障碍优先匹配，避免很小的噪声簇抢占已有轨迹。
  std::sort(
      clusters.begin(), clusters.end(),
      [](const FrameCluster& lhs, const FrameCluster& rhs)
      {
        return lhs.points.size() > rhs.points.size();
      });

  return clusters;
}

void VisionSnapshotLayer::updateTracksLocked(
    const std::vector<FrameCluster>& clusters,
    const ros::Time& now)
{
  purgeExpiredLocked(now);

  std::vector<bool> track_matched(tracks_.size(), false);
  const double association_sq =
      association_distance_m_ * association_distance_m_;
  const ros::Time expiry =
      now + ros::Duration(hold_time_s_);

  for (const FrameCluster& cluster : clusters)
  {
    std::size_t best_index = tracks_.size();
    double best_distance_sq =
        std::numeric_limits<double>::infinity();

    for (std::size_t index = 0U;
         index < tracks_.size(); ++index)
    {
      if (track_matched[index])
      {
        continue;
      }

      const double dx =
          cluster.centroid_x - tracks_[index].centroid_x;
      const double dy =
          cluster.centroid_y - tracks_[index].centroid_y;
      const double distance_sq = dx * dx + dy * dy;

      if (distance_sq <= association_sq &&
          distance_sq < best_distance_sq)
      {
        best_distance_sq = distance_sq;
        best_index = index;
      }
    }

    if (best_index < tracks_.size())
    {
      // 同一障碍的新观测：只替换该障碍的几何轮廓并刷新TTL。
      // 其他未匹配障碍继续保留到各自expiry。
      ObstacleTrack& track = tracks_[best_index];
      track.points = cluster.points;
      track.centroid_x = cluster.centroid_x;
      track.centroid_y = cluster.centroid_y;
      track.expiry = expiry;
      track_matched[best_index] = true;
    }
    else
    {
      // 新障碍：增加独立轨迹，不清除已有障碍。
      ObstacleTrack track;
      track.id = next_track_id_++;
      track.points = cluster.points;
      track.centroid_x = cluster.centroid_x;
      track.centroid_y = cluster.centroid_y;
      track.expiry = expiry;
      tracks_.push_back(track);
      track_matched.push_back(true);
    }
  }
}

void VisionSnapshotLayer::scanCallback(
    const sensor_msgs::LaserScanConstPtr& msg)
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
    ROS_WARN_THROTTLE(
        2.0,
        "VisionSnapshotLayer dropping stale/future scan: age=%.3fs",
        age);
    return;
  }

  geometry_msgs::TransformStamped transform_msg;

  try
  {
    transform_msg = tf_->lookupTransform(
        global_frame_, msg->header.frame_id,
        stamp, ros::Duration(transform_timeout_s_));
  }
  catch (const tf2::TransformException& ex)
  {
    ++dropped_scan_count_;
    ROS_WARN_THROTTLE(
        2.0,
        "VisionSnapshotLayer TF failed %s <- %s: %s",
        global_frame_.c_str(),
        msg->header.frame_id.c_str(),
        ex.what());
    return;
  }

  tf2::Transform global_from_scan;
  tf2::fromMsg(transform_msg.transform, global_from_scan);

  std::vector<TimedPoint> frame_points;
  frame_points.reserve(msg->ranges.size());

  double angle = msg->angle_min;
  const double usable_min =
      std::max<double>(msg->range_min, min_range_m_);
  const double usable_max =
      std::min<double>(msg->range_max, max_range_m_);

  for (const float range : msg->ranges)
  {
    if (std::isfinite(range) &&
        range >= usable_min &&
        range <= usable_max)
    {
      const tf2::Vector3 local_point(
          range * std::cos(angle),
          range * std::sin(angle),
          0.0);
      const tf2::Vector3 global_point =
          global_from_scan * local_point;

      TimedPoint point;
      point.x = global_point.x();
      point.y = global_point.y();
      frame_points.push_back(point);
    }

    angle += msg->angle_increment;
  }

  const std::vector<FrameCluster> frame_clusters =
      clusterFramePoints(frame_points);

  std::size_t track_count = 0U;
  std::size_t active_point_count = 0U;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    last_scan_receive_time_ = receive_time;

    // 空帧：不创建新轨迹，也不立即删除旧轨迹。
    // 非空帧：匹配并刷新对应障碍；未匹配旧障碍仍保持到TTL。
    updateTracksLocked(frame_clusters, receive_time);

    track_count = tracks_.size();
    for (const ObstacleTrack& track : tracks_)
    {
      active_point_count += track.points.size();
    }
  }

  ++accepted_scan_count_;
  current_ = true;

  ROS_INFO_THROTTLE(
      1.0,
      "VisionSnapshotLayer scan accepted: raw_points=%zu "
      "clusters=%zu tracks=%zu active_points=%zu age=%.3fs",
      frame_points.size(), frame_clusters.size(),
      track_count, active_point_count, age);
}

void VisionSnapshotLayer::purgeExpiredLocked(
    const ros::Time& now)
{
  tracks_.erase(
      std::remove_if(
          tracks_.begin(), tracks_.end(),
          [&now](const ObstacleTrack& track)
          {
            return track.expiry <= now;
          }),
      tracks_.end());
}

bool VisionSnapshotLayer::sourceIsCurrent(
    const ros::Time& now) const
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

  return (now - last_scan_receive_time_).toSec() <=
         expected_update_rate_s_;
}

void VisionSnapshotLayer::publishActiveScan(
    double robot_x, double robot_y,
    double robot_yaw, const ros::Time& stamp)
{
  if (!publish_active_scan_ || !active_scan_pub_)
  {
    return;
  }

  const int beam_count = static_cast<int>(
      std::round((2.0 * M_PI) /
                 active_scan_angle_increment_));

  sensor_msgs::LaserScan scan;
  scan.header.stamp = stamp;
  scan.header.frame_id = active_scan_frame_;
  scan.angle_min = -M_PI;
  scan.angle_increment = active_scan_angle_increment_;
  scan.angle_max =
      scan.angle_min +
      (beam_count - 1) * scan.angle_increment;
  scan.time_increment = 0.0;
  scan.scan_time = 0.0;
  scan.range_min = min_range_m_;
  scan.range_max = max_range_m_;
  scan.ranges.assign(
      static_cast<std::size_t>(beam_count),
      std::numeric_limits<float>::infinity());

  const double c = std::cos(robot_yaw);
  const double s = std::sin(robot_yaw);

  std::lock_guard<std::mutex> lock(mutex_);

  for (const ObstacleTrack& track : tracks_)
  {
    for (const TimedPoint& point : track.points)
    {
      const double dx = point.x - robot_x;
      const double dy = point.y - robot_y;
      const double bx = c * dx + s * dy;
      const double by = -s * dx + c * dy;
      const double range = std::hypot(bx, by);

      if (range < min_range_m_ ||
          range > max_range_m_)
      {
        continue;
      }

      const double angle = std::atan2(by, bx);
      int index = static_cast<int>(
          std::round((angle - scan.angle_min) /
                     scan.angle_increment));
      index %= beam_count;

      if (index < 0)
      {
        index += beam_count;
      }

      const std::size_t beam_index =
          static_cast<std::size_t>(index);
      scan.ranges[beam_index] = std::min(
          scan.ranges[beam_index],
          static_cast<float>(range));
    }
  }

  active_scan_pub_.publish(scan);
}

void VisionSnapshotLayer::updateBounds(
    double robot_x, double robot_y,
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
    updateOrigin(
        robot_x - getSizeInMetersX() / 2.0,
        robot_y - getSizeInMetersY() / 2.0);
  }

  const ros::Time now = ros::Time::now();
  const bool source_current = sourceIsCurrent(now);

  // stop_on_stale只控制current_状态，不再阻止TTL清理。
  current_ = source_current || !stop_on_stale_;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // 无论数据源是否stale，都按每个障碍自己的TTL清理。
    purgeExpiredLocked(now);

    // 每个costmap周期重建整个视觉层。
    resetMaps();

    for (const ObstacleTrack& track : tracks_)
    {
      for (const TimedPoint& point : track.points)
      {
        unsigned int mx = 0U;
        unsigned int my = 0U;

        if (worldToMap(point.x, point.y, mx, my))
        {
          setCost(
              mx, my,
              costmap_2d::LETHAL_OBSTACLE);
        }
      }
    }
  }

  *min_x = std::min(*min_x, getOriginX());
  *min_y = std::min(*min_y, getOriginY());
  *max_x = std::max(
      *max_x,
      getOriginX() + getSizeInMetersX());
  *max_y = std::max(
      *max_y,
      getOriginY() + getSizeInMetersY());

  publishActiveScan(
      robot_x, robot_y, robot_yaw, now);
}

void VisionSnapshotLayer::updateCosts(
    costmap_2d::Costmap2D& master_grid,
    int min_i, int min_j,
    int max_i, int max_j)
{
  if (!enabled_)
  {
    return;
  }

  updateWithMax(
      master_grid,
      min_i, min_j, max_i, max_j);
}

void VisionSnapshotLayer::reset()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tracks_.clear();
    next_track_id_ = 1U;
    last_scan_receive_time_ = ros::Time(0);
  }

  resetMaps();
  current_ = true;
}

}  // namespace r300_1x_navigation

PLUGINLIB_EXPORT_CLASS(
    r300_1x_navigation::VisionSnapshotLayer,
    costmap_2d::Layer)
