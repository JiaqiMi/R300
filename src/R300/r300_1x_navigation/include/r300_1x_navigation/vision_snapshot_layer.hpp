#ifndef R300_1X_NAVIGATION_VISION_SNAPSHOT_LAYER_HPP_
#define R300_1X_NAVIGATION_VISION_SNAPSHOT_LAYER_HPP_

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <costmap_2d/costmap_layer.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

namespace r300_1x_navigation
{

class VisionSnapshotLayer : public costmap_2d::CostmapLayer
{
public:
  VisionSnapshotLayer();
  ~VisionSnapshotLayer() override = default;

  void onInitialize() override;
  void matchSize() override;
  void updateBounds(double robot_x, double robot_y, double robot_yaw,
                    double* min_x, double* min_y,
                    double* max_x, double* max_y) override;
  void updateCosts(costmap_2d::Costmap2D& master_grid,
                   int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;

private:
  struct TimedPoint
  {
    double x = 0.0;
    double y = 0.0;
    ros::Time expiry;
  };

  void scanCallback(const sensor_msgs::LaserScanConstPtr& msg);
  void purgeExpiredLocked(const ros::Time& now);
  std::int64_t makeKey(double x, double y) const;
  bool sourceIsCurrent(const ros::Time& now) const;
  void publishActiveScan(double robot_x, double robot_y, double robot_yaw,
                         const ros::Time& stamp);

  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Publisher active_scan_pub_;

  std::string topic_;
  std::string global_frame_;
  std::string active_scan_topic_;
  std::string active_scan_frame_;

  double hold_time_s_;
  double expected_update_rate_s_;
  double max_message_age_s_;
  double transform_timeout_s_;
  double min_range_m_;
  double max_range_m_;
  double dedup_resolution_m_;
  double active_scan_angle_increment_;
  bool stop_on_stale_;
  bool publish_active_scan_;

  mutable std::mutex mutex_;
  std::unordered_map<std::int64_t, TimedPoint> active_points_;
  ros::Time last_scan_receive_time_;
  std::uint64_t accepted_scan_count_;
  std::uint64_t dropped_scan_count_;
};

}  // namespace r300_1x_navigation

#endif  // R300_1X_NAVIGATION_VISION_SNAPSHOT_LAYER_HPP_
