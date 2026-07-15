#ifndef R300_1X_NAVIGATION_VISION_SNAPSHOT_LAYER_HPP_
#define R300_1X_NAVIGATION_VISION_SNAPSHOT_LAYER_HPP_

#include <cstddef>
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
  };

  struct FrameCluster
  {
    std::vector<TimedPoint> points;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
  };

  struct ObstacleTrack
  {
    std::uint64_t id = 0U;
    std::vector<TimedPoint> points;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    ros::Time expiry;
  };

  void scanCallback(const sensor_msgs::LaserScanConstPtr& msg);
  std::vector<FrameCluster> clusterFramePoints(
      const std::vector<TimedPoint>& points) const;
  void updateTracksLocked(const std::vector<FrameCluster>& clusters,
                          const ros::Time& now);
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

  // 同一帧内将相邻几何点聚成一个障碍物。
  double cluster_tolerance_m_;

  // 新障碍簇与已有障碍轨迹的最大匹配距离。
  // 匹配成功时只替换该障碍，不会删除其他仍在TTL内的障碍。
  double association_distance_m_;

  std::size_t min_cluster_points_;

  bool stop_on_stale_;
  bool publish_active_scan_;

  mutable std::mutex mutex_;
  std::vector<ObstacleTrack> tracks_;
  std::uint64_t next_track_id_;
  ros::Time last_scan_receive_time_;
  std::uint64_t accepted_scan_count_;
  std::uint64_t dropped_scan_count_;
};

}  // namespace r300_1x_navigation

#endif  // R300_1X_NAVIGATION_VISION_SNAPSHOT_LAYER_HPP_
