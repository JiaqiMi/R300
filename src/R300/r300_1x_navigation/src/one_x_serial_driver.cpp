#include <algorithm>
#include <array>
#include <cstddef>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include "r300_1x_navigation/Attitude.h"
#include "r300_1x_navigation/InsImu.h"
#include "r300_1x_navigation/InsStatus.h"
#include "r300_1x_navigation/UpdateFlag.h"
#include "r300_1x_navigation/Velocity.h"
#include "r300_1x_navigation/geodesy.hpp"

namespace
{
constexpr std::size_t kFrameLength = 110U;
constexpr std::array<uint8_t, 4> kHeader{{0xAA, 0x55, 0x5A, 0xA5}};
constexpr double kPi = 3.14159265358979323846;

uint16_t U16Le(const uint8_t *p)
{
  return static_cast<uint16_t>(p[0]) |
         (static_cast<uint16_t>(p[1]) << 8U);
}

int16_t I16Le(const uint8_t *p)
{
  const uint16_t u = U16Le(p);
  if ((u & 0x8000U) == 0U)
  {
    return static_cast<int16_t>(u);
  }
  return static_cast<int16_t>(static_cast<int32_t>(u) - 65536);
}

uint32_t U32Le(const uint8_t *p)
{
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8U) |
         (static_cast<uint32_t>(p[2]) << 16U) |
         (static_cast<uint32_t>(p[3]) << 24U);
}

int32_t I32Le(const uint8_t *p)
{
  const uint32_t u = U32Le(p);
  if ((u & 0x80000000U) == 0U)
  {
    return static_cast<int32_t>(u);
  }
  return static_cast<int32_t>(static_cast<int64_t>(u) - 4294967296LL);
}

double WrapPi(double angle_rad)
{
  while (angle_rad > kPi)
  {
    angle_rad -= 2.0 * kPi;
  }
  while (angle_rad <= -kPi)
  {
    angle_rad += 2.0 * kPi;
  }
  return angle_rad;
}

speed_t BaudToTermiosSpeed(int baudrate)
{
  switch (baudrate)
  {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
#ifdef B230400
    case 230400: return B230400;
#endif
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
    default:
      throw std::runtime_error("Unsupported baudrate: " + std::to_string(baudrate));
  }
}

std::string ToString(double value, int precision = 3)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  return ss.str();
}

std::string ToString(uint64_t value)
{
  return std::to_string(value);
}

double RoundTo7Decimals(double value)
{
  constexpr double kScale = 10000000.0;
  return std::round(value * kScale) / kScale;
}

std::string JoinStrings(const std::vector<std::string> &values, const std::string &fallback)
{
  if (values.empty())
  {
    return fallback;
  }

  std::ostringstream ss;
  for (std::size_t i = 0U; i < values.size(); ++i)
  {
    if (i > 0U)
    {
      ss << ", ";
    }
    ss << values[i];
  }
  return ss.str();
}

}  // namespace

class OneXSerialDriver
{
public:
  OneXSerialDriver()
      : nh_(), pnh_("~"), serial_fd_(-1), origin_ready_(false), have_counter_(false),
        have_last_yaw_(false), valid_frames_(0U), checksum_failures_(0U), malformed_frames_(0U),
        skipped_bytes_(0U), counter_anomalies_(0U), last_valid_frame_time_(0.0),
        last_yaw_time_(0.0), last_yaw_(0.0), have_latest_position_(false)
  {
    pnh_.param<std::string>("serial_port", serial_port_, std::string("/dev/ttyUSB0"));
    pnh_.param<int>("baudrate", baudrate_, 460800);
    pnh_.param<std::string>("odom_frame", odom_frame_, std::string("odom"));
    pnh_.param<std::string>("base_frame", base_frame_, std::string("base_link"));
    pnh_.param<std::string>("fix_frame", fix_frame_, std::string("map"));
    pnh_.param<bool>("publish_full_attitude", publish_full_attitude_, false);
    pnh_.param<double>("position_std_m", position_std_m_, 0.50);
    pnh_.param<double>("yaw_std_deg", yaw_std_deg_, 3.0);
    pnh_.param<double>("max_yaw_rate_radps", max_yaw_rate_radps_, 2.5);
    pnh_.param<int>("max_buffer_bytes", max_buffer_bytes_, 8192);

    pnh_.param<std::string>("origin_mode", origin_mode_, std::string("first_valid"));
    pnh_.param<double>("origin_set_max_age_s", origin_set_max_age_s_, 0.50);
    origin_set_max_age_s_ = std::max(0.05, origin_set_max_age_s_);

    if (origin_mode_ == "fixed")
    {
      pnh_.getParam("origin_latitude_deg", origin_.lat_deg);
      pnh_.getParam("origin_longitude_deg", origin_.lon_deg);
      pnh_.param<double>("origin_altitude_m", origin_.alt_m, 0.0);
      if (!IsValidLatitudeLongitude(origin_.lat_deg, origin_.lon_deg))
      {
        throw std::runtime_error("origin_mode=fixed requires valid origin_latitude_deg and origin_longitude_deg");
      }
      origin_ready_ = true;
      ROS_INFO_STREAM("1X fixed local origin: lat=" << std::setprecision(10) << origin_.lat_deg
                                                      << ", lon=" << origin_.lon_deg
                                                      << ", alt=" << origin_.alt_m);
    }
    else if (origin_mode_ == "deferred")
    {
      ROS_WARN("1X origin is deferred: raw decoded topics are active, but /one_x/odom and "
               "odom->base_link will start only after /one_x/set_current_origin is called.");
    }
    else if (origin_mode_ != "first_valid")
    {
      throw std::runtime_error("origin_mode must be first_valid, deferred, or fixed");
    }

    // /one_x/fix remains the navigation/control position source.
    // These two extra topics expose the original INS and GPS coordinates
    // separately, at the same rate and with the same header stamp.
    fix_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/one_x/fix", 20);
    ins_fix_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/one_x/ins_fix", 20);
    gps_fix_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/one_x/gps_fix", 20);
    origin_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/one_x/origin", 1, true);
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/one_x/odom", 50);
    imu_pub_ = nh_.advertise<sensor_msgs::Imu>("/one_x/imu", 100);
    ins_imu_pub_ = nh_.advertise<r300_1x_navigation::InsImu>("/one_x/ins_imu", 100);
    attitude_pub_ = nh_.advertise<r300_1x_navigation::Attitude>("/one_x/attitude", 100);
    velocity_pub_ = nh_.advertise<r300_1x_navigation::Velocity>("/one_x/vel", 100);
    update_flag_pub_ = nh_.advertise<r300_1x_navigation::UpdateFlag>("/one_x/update_flag", 100);
    heading_pub_ = nh_.advertise<std_msgs::Float64>("/one_x/heading_deg", 20);
    pos_compare_pub_ = nh_.advertise<std_msgs::String>("/one_x/pos_compare", 2);
    ins_status_pub_ = nh_.advertise<r300_1x_navigation::InsStatus>("/one_x/ins_status", 100);
    diagnostics_pub_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>("/one_x/diagnostics", 2);
    set_origin_service_ = nh_.advertiseService(
        "/one_x/set_current_origin",
        &OneXSerialDriver::SetCurrentOriginCallback,
        this);

    if (origin_ready_)
    {
      PublishOrigin(ros::Time::now());
    }

    // /one_x/ins_status is published once per checksum-valid serial frame so its
    // rosbag rate and timestamp align with the other decoded 1X topics.
    pos_compare_timer_ = nh_.createTimer(ros::Duration(2.0), &OneXSerialDriver::PosCompareTimerCallback, this);
    diagnostics_timer_ = nh_.createTimer(ros::Duration(1.0), &OneXSerialDriver::DiagnosticsTimerCallback, this);
    OpenSerial();
  }

  ~OneXSerialDriver()
  {
    if (serial_fd_ >= 0)
    {
      close(serial_fd_);
      serial_fd_ = -1;
    }
  }

  void Spin()
  {
    ros::Rate rate(500.0);
    while (ros::ok())
    {
      ReadAvailableBytes();
      ParseBufferedFrames();
      ros::spinOnce();
      rate.sleep();
    }
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  // /one_x/fix is the position source currently used by navigation.
  // /one_x/ins_fix and /one_x/gps_fix preserve both raw position sources
  // with the same ROS header stamp for offline comparison / rosbag recording.
  ros::Publisher fix_pub_;
  ros::Publisher ins_fix_pub_;
  ros::Publisher gps_fix_pub_;
  ros::Publisher origin_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher imu_pub_;
  ros::Publisher ins_imu_pub_;
  ros::Publisher attitude_pub_;
  ros::Publisher velocity_pub_;
  ros::Publisher update_flag_pub_;
  ros::Publisher heading_pub_;
  ros::Publisher pos_compare_pub_;
  ros::Publisher ins_status_pub_;
  ros::Publisher diagnostics_pub_;
  ros::Timer pos_compare_timer_;
  ros::Timer diagnostics_timer_;
  ros::ServiceServer set_origin_service_;
  tf::TransformBroadcaster tf_broadcaster_;

  std::string serial_port_;
  int baudrate_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string fix_frame_;
  bool publish_full_attitude_;
  double position_std_m_;
  double yaw_std_deg_;
  double max_yaw_rate_radps_;
  int max_buffer_bytes_;
  std::string origin_mode_;
  double origin_set_max_age_s_;

  int serial_fd_;
  std::vector<uint8_t> rx_buffer_;

  r300_1x_navigation::Geodetic origin_{0.0, 0.0, 0.0};
  r300_1x_navigation::Geodetic latest_control_position_{0.0, 0.0, 0.0};
  ros::Time latest_control_position_stamp_;
  bool have_latest_control_position_ = false;
  bool origin_ready_;
  bool have_counter_;
  uint16_t previous_counter_;
  bool have_last_yaw_;
  ros::Time last_yaw_time_;
  double last_yaw_;

  uint64_t valid_frames_;
  uint64_t checksum_failures_;
  uint64_t malformed_frames_;
  uint64_t skipped_bytes_;
  uint64_t counter_anomalies_;
  ros::Time last_valid_frame_time_;

  static bool IsValidLatitudeLongitude(double lat_deg, double lon_deg)
  {
    return std::isfinite(lat_deg) && std::isfinite(lon_deg) &&
           std::fabs(lat_deg) <= 90.0 && std::fabs(lon_deg) <= 180.0;
  }

  void OpenSerial()
  {
    serial_fd_ = open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0)
    {
      throw std::runtime_error("Failed to open " + serial_port_ + ": " + std::strerror(errno));
    }

    termios options;
    std::memset(&options, 0, sizeof(options));
    if (tcgetattr(serial_fd_, &options) != 0)
    {
      const std::string error = std::strerror(errno);
      close(serial_fd_);
      serial_fd_ = -1;
      throw std::runtime_error("tcgetattr failed: " + error);
    }

    const speed_t speed = BaudToTermiosSpeed(baudrate_);
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
#ifdef CRTSCTS
    options.c_cflag &= ~CRTSCTS;
#endif
    options.c_iflag = IGNPAR;
    options.c_oflag = 0;
    options.c_lflag = 0;
    options.c_cc[VTIME] = 0;
    options.c_cc[VMIN] = 0;

    tcflush(serial_fd_, TCIOFLUSH);
    if (tcsetattr(serial_fd_, TCSANOW, &options) != 0)
    {
      const std::string error = std::strerror(errno);
      close(serial_fd_);
      serial_fd_ = -1;
      throw std::runtime_error("tcsetattr failed: " + error);
    }

    ROS_INFO_STREAM("1X serial driver opened " << serial_port_ << " at " << baudrate_ << " bps");
  }

  void ReadAvailableBytes()
  {
    std::array<uint8_t, 2048> temp{{0U}};
    while (true)
    {
      const ssize_t read_size = read(serial_fd_, temp.data(), temp.size());
      if (read_size > 0)
      {
        rx_buffer_.insert(rx_buffer_.end(), temp.begin(), temp.begin() + read_size);
      }
      else if (read_size == 0)
      {
        break;
      }
      else
      {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
          break;
        }
        ROS_ERROR_THROTTLE(1.0, "1X serial read error: %s", std::strerror(errno));
        break;
      }
    }

    if (static_cast<int>(rx_buffer_.size()) > max_buffer_bytes_)
    {
      const std::size_t preserve = std::min<std::size_t>(rx_buffer_.size(), kFrameLength - 1U);
      const std::size_t drop_count = rx_buffer_.size() - preserve;
      rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<std::ptrdiff_t>(drop_count));
      skipped_bytes_ += drop_count;
      ROS_WARN_THROTTLE(1.0, "1X receive buffer overflow protection dropped %zu bytes", drop_count);
    }
  }

  void ParseBufferedFrames()
  {
    while (rx_buffer_.size() >= kHeader.size())
    {
      const auto header_it = std::search(rx_buffer_.begin(), rx_buffer_.end(), kHeader.begin(), kHeader.end());
      if (header_it == rx_buffer_.end())
      {
        const std::size_t preserve = std::min<std::size_t>(rx_buffer_.size(), kHeader.size() - 1U);
        const std::size_t drop_count = rx_buffer_.size() - preserve;
        rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<std::ptrdiff_t>(drop_count));
        skipped_bytes_ += drop_count;
        return;
      }

      if (header_it != rx_buffer_.begin())
      {
        const std::size_t drop_count = static_cast<std::size_t>(header_it - rx_buffer_.begin());
        rx_buffer_.erase(rx_buffer_.begin(), header_it);
        skipped_bytes_ += drop_count;
      }

      if (rx_buffer_.size() < kFrameLength)
      {
        return;
      }

      uint32_t checksum_sum = 0U;
      for (std::size_t i = 0; i < 108U; ++i)
      {
        checksum_sum += rx_buffer_[i];
      }
      const uint16_t checksum_received = U16Le(rx_buffer_.data() + 108U);
      const uint16_t checksum_calculated = static_cast<uint16_t>(checksum_sum & 0xFFFFU);

      if (checksum_received != checksum_calculated)
      {
        ++checksum_failures_;
        rx_buffer_.erase(rx_buffer_.begin());
        ++skipped_bytes_;
        continue;
      }

      ParseValidFrame(rx_buffer_.data());
      rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<std::ptrdiff_t>(kFrameLength));
    }
  }

  void ParseValidFrame(const uint8_t *frame)
  {
    const uint16_t counter = U16Le(frame + 4U);

    // 110-byte frame position fields:
    //   bytes  6.. 9 : INS latitude
    //   bytes 10..13 : INS longitude
    //   bytes 34..37 : GPS latitude
    //   bytes 38..41 : GPS longitude
    //
    // Keep both sources independently.  Do not overwrite the INS variables
    // with GPS values; otherwise /one_x/pos_compare cannot show a real
    // INS-versus-GPS comparison.
    const double ins_latitude_deg =
        static_cast<double>(I32Le(frame + 6U)) * 180.0 / 2147483648.0;
    const double ins_longitude_deg =
        static_cast<double>(I32Le(frame + 10U)) * 180.0 / 2147483648.0;
    const double gps_latitude_deg =
        static_cast<double>(I32Le(frame + 34U)) * 180.0 / 2147483648.0;
    const double gps_longitude_deg =
        static_cast<double>(I32Le(frame + 38U)) * 180.0 / 2147483648.0;

    // Current navigation/control source.  Keep these two assignments if you
    // want move_base and odom->base_link to use GPS position.
    // To switch back to INS control later, change only these two lines.

    // 当前控制仍使用 GPS。
    const double latitude_deg = gps_latitude_deg;
    const double longitude_deg = gps_longitude_deg;
    
    // 当前控制使用 INS。
    // const double latitude_deg = ins_latitude_deg;
    // const double longitude_deg = ins_longitude_deg;

    // The 110-byte layout currently verified in this driver supplies the
    // navigation/INS altitude at bytes 14..17.  It is attached to both
    // NavSatFix messages only because a separate raw GPS altitude field has
    // not been verified yet.
    const double altitude_m = static_cast<double>(I32Le(frame + 14U)) * 1.0e-3;
    // 经实车“车头朝北直线前进”测试确认：协议速度字段顺序为 Ve、Vn、Vd。
    const int16_t ve_raw = I16Le(frame + 20U);
    const int16_t vn_raw = I16Le(frame + 22U);
    const int16_t vd_raw = I16Le(frame + 24U);
    const double ve_mps = static_cast<double>(ve_raw) * 1.0e-3;
    const double vn_mps = static_cast<double>(vn_raw) * 1.0e-3;
    const double vd_mps = static_cast<double>(vd_raw) * 1.0e-3;

    const int16_t roll_raw = I16Le(frame + 26U);
    const int16_t pitch_raw = I16Le(frame + 28U);
    const uint16_t heading_raw = U16Le(frame + 30U);
    const double roll_deg = static_cast<double>(roll_raw) * 180.0 / 32768.0;
    const double pitch_deg = static_cast<double>(pitch_raw) * 180.0 / 32768.0;
    const double heading_deg = static_cast<double>(heading_raw) * 360.0 / 65536.0;
    const uint16_t ins_status = U16Le(frame + 32U);
    const uint8_t gps_status = frame[50U];

    const int32_t gx_raw = I32Le(frame + 62U);
    const int32_t gy_raw = I32Le(frame + 66U);
    const int32_t gz_raw = I32Le(frame + 70U);
    const int32_t ax_raw = I32Le(frame + 74U);
    const int32_t ay_raw = I32Le(frame + 78U);
    const int32_t az_raw = I32Le(frame + 82U);
    // Quantise the scaled physical values to 7 decimal places before publishing.
    // The integer raw fields are also published, so the original serial values
    // can always be reconstructed exactly during offline analysis.
    const double gx_rfu_dps = RoundTo7Decimals(static_cast<double>(gx_raw) * 1.0e-6);
    const double gy_rfu_dps = RoundTo7Decimals(static_cast<double>(gy_raw) * 1.0e-6);
    const double gz_rfu_dps = RoundTo7Decimals(static_cast<double>(gz_raw) * 1.0e-6);
    const double ax_rfu_mps2 = RoundTo7Decimals(static_cast<double>(ax_raw) * 1.0e-7);
    const double ay_rfu_mps2 = RoundTo7Decimals(static_cast<double>(ay_raw) * 1.0e-7);
    const double az_rfu_mps2 = RoundTo7Decimals(static_cast<double>(az_raw) * 1.0e-7);
    const double temperature_c = static_cast<double>(I16Le(frame + 86U)) * 0.01;
    const uint16_t imu_status = U16Le(frame + 88U);
    const uint8_t update_flag = frame[107U];

    const bool ins_position_valid =
        IsValidLatitudeLongitude(ins_latitude_deg, ins_longitude_deg);
    const bool gps_position_valid =
        IsValidLatitudeLongitude(gps_latitude_deg, gps_longitude_deg);

    // Reject only when the active control source is invalid.  INS and GPS
    // remain independently available below for monitoring and logging.
    const bool control_position_valid =
        IsValidLatitudeLongitude(latitude_deg, longitude_deg);
    if (!control_position_valid || !std::isfinite(altitude_m))
    {
      ++malformed_frames_;
      ROS_WARN_THROTTLE(
          1.0,
          "1X parsed invalid control position: control_lat=%.10f control_lon=%.10f "
          "gps_lat=%.10f gps_lon=%.10f ins_lat=%.10f ins_lon=%.10f",
          latitude_deg, longitude_deg,
          gps_latitude_deg, gps_longitude_deg,
          ins_latitude_deg, ins_longitude_deg);
      return;
    }

    const ros::Time stamp = ros::Time::now();
    last_valid_frame_time_ = stamp;
    ++valid_frames_;

    if (have_counter_)
    {
      const uint16_t expected = static_cast<uint16_t>(previous_counter_ + 1U);
      // Keep the MATLAB convention: repeated counter values are tolerated and only abnormal jumps are counted.
      if (counter != previous_counter_ && counter != expected)
      {
        ++counter_anomalies_;
        ROS_WARN_THROTTLE(1.0, "1X counter discontinuity: previous=%u current=%u expected=%u",
                          previous_counter_, counter, expected);
      }
    }
    previous_counter_ = counter;
    have_counter_ = true;

    latest_control_position_ = r300_1x_navigation::Geodetic{latitude_deg, longitude_deg, altitude_m};
    latest_control_position_stamp_ = stamp;
    have_latest_control_position_ = true;

    if (!origin_ready_ && origin_mode_ == "first_valid")
    {
      origin_ = latest_control_position_;
      origin_ready_ = true;
      PublishOrigin(stamp);
      ROS_INFO_STREAM("1X first-valid local origin locked: lat=" << std::setprecision(10) << origin_.lat_deg
                                                                   << ", lon=" << origin_.lon_deg
                                                                   << ", alt=" << origin_.alt_m);
    }

    const r300_1x_navigation::Geodetic current{latitude_deg, longitude_deg, altitude_m};
    r300_1x_navigation::Enu enu{0.0, 0.0, 0.0};
    if (origin_ready_)
    {
      enu = r300_1x_navigation::GeodeticToEnu(current, origin_);
    }

    // INS heading: North=0 deg, East=90 deg, clockwise positive.
    // ROS ENU yaw: East=0 rad, North=+pi/2 rad, counter-clockwise positive.
    const double heading_rad = heading_deg * kPi / 180.0;
    const double yaw_ros = WrapPi(kPi / 2.0 - heading_rad);

    double yaw_rate = 0.0;
    if (have_last_yaw_)
    {
      const double dt = (stamp - last_yaw_time_).toSec();
      if (dt > 1.0e-4 && dt < 1.0)
      {
        yaw_rate = WrapPi(yaw_ros - last_yaw_) / dt;
        yaw_rate = std::max(-max_yaw_rate_radps_, std::min(max_yaw_rate_radps_, yaw_rate));
      }
    }
    last_yaw_ = yaw_ros;
    last_yaw_time_ = stamp;
    have_last_yaw_ = true;

    // Navigation-frame speed (NED) -> ROS ENU for pose; body-frame FLU for Odometry twist.
    const double vx_enu = ve_mps;
    const double vy_enu = vn_mps;
    const double vz_enu = -vd_mps;
    const double v_forward = std::cos(heading_rad) * vn_mps + std::sin(heading_rad) * ve_mps;
    const double v_left = std::sin(heading_rad) * vn_mps - std::cos(heading_rad) * ve_mps;

    // ROS_WARN_THROTTLE(
    //     0.2,
    //     "VELDBG hdg=%.1f | vn=%.3f ve=%.3f | body_x=%.3f body_y=%.3f",
    //     heading_deg,
    //     vn_mps,
    //     ve_mps,
    //     v_forward,
    //     v_left);

    const geometry_msgs::Quaternion yaw_quaternion = tf::createQuaternionMsgFromYaw(yaw_ros);

    // Publish the two original latitude/longitude sources separately.
    // Both messages deliberately share the same header stamp and frame ID,
    // so a rosbag can be aligned sample-by-sample without interpolation.
    sensor_msgs::NavSatFix ins_fix_msg;
    ins_fix_msg.header.stamp = stamp;
    ins_fix_msg.header.frame_id = fix_frame_;
    ins_fix_msg.status.status = ins_position_valid
                                    ? sensor_msgs::NavSatStatus::STATUS_FIX
                                    : sensor_msgs::NavSatStatus::STATUS_NO_FIX;
    ins_fix_msg.status.service = 0U;
    ins_fix_msg.latitude = ins_latitude_deg;
    ins_fix_msg.longitude = ins_longitude_deg;
    ins_fix_msg.altitude = altitude_m;
    ins_fix_msg.position_covariance_type =
        sensor_msgs::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
    ins_fix_pub_.publish(ins_fix_msg);

    sensor_msgs::NavSatFix gps_fix_msg = ins_fix_msg;
    gps_fix_msg.status.status = gps_position_valid
                                    ? sensor_msgs::NavSatStatus::STATUS_FIX
                                    : sensor_msgs::NavSatStatus::STATUS_NO_FIX;
    gps_fix_msg.latitude = gps_latitude_deg;
    gps_fix_msg.longitude = gps_longitude_deg;
    gps_fix_pub_.publish(gps_fix_msg);

    // Preserve /one_x/fix as the active control source for compatibility
    // with the rest of the navigation stack.  It is GPS in this version.
    sensor_msgs::NavSatFix fix_msg = gps_fix_msg;
    fix_msg.latitude = latitude_deg;
    fix_msg.longitude = longitude_deg;
    fix_pub_.publish(fix_msg);

    const double pos_var = position_std_m_ * position_std_m_;
    const double yaw_var = std::pow(yaw_std_deg_ * kPi / 180.0, 2.0);

    if (origin_ready_)
    {
      nav_msgs::Odometry odom_msg;
      odom_msg.header.stamp = stamp;
      odom_msg.header.frame_id = odom_frame_;
      odom_msg.child_frame_id = base_frame_;
      odom_msg.pose.pose.position.x = enu.east;
      odom_msg.pose.pose.position.y = enu.north;
      odom_msg.pose.pose.position.z = enu.up;
      odom_msg.pose.pose.orientation = yaw_quaternion;
      odom_msg.pose.covariance[0] = pos_var;
      odom_msg.pose.covariance[7] = pos_var;
      odom_msg.pose.covariance[14] = 4.0 * pos_var;
      odom_msg.pose.covariance[21] = 1.0e6;
      odom_msg.pose.covariance[28] = 1.0e6;
      odom_msg.pose.covariance[35] = yaw_var;
      odom_msg.twist.twist.linear.x = v_forward;
      odom_msg.twist.twist.linear.y = v_left;
      odom_msg.twist.twist.linear.z = 0.0;
      odom_msg.twist.twist.angular.z = yaw_rate;
      odom_msg.twist.covariance[0] = 0.25;
      odom_msg.twist.covariance[7] = 0.25;
      odom_msg.twist.covariance[35] = 0.25;
      odom_pub_.publish(odom_msg);

      geometry_msgs::TransformStamped tf_msg;
      tf_msg.header.stamp = stamp;
      tf_msg.header.frame_id = odom_frame_;
      tf_msg.child_frame_id = base_frame_;
      tf_msg.transform.translation.x = enu.east;
      tf_msg.transform.translation.y = enu.north;
      tf_msg.transform.translation.z = enu.up;
      tf_msg.transform.rotation = yaw_quaternion;
      tf_broadcaster_.sendTransform(tf_msg);
    }
    else
    {
      ROS_INFO_THROTTLE(5.0,
                        "1X parser is running; waiting for /one_x/set_current_origin before publishing odometry/TF");
    }

    sensor_msgs::Imu imu_msg;
    imu_msg.header.stamp = stamp;
    imu_msg.header.frame_id = base_frame_;
    if (publish_full_attitude_)
    {
      // Must be verified in RViz before enabling: protocol RFU -> ROS FLU.
      // Roll positive (port/left side up) is retained; pitch positive (bow up) changes sign.
      imu_msg.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll_deg * kPi / 180.0,
                                                                      -pitch_deg * kPi / 180.0,
                                                                      yaw_ros);
    }
    else
    {
      imu_msg.orientation = yaw_quaternion;
    }
    imu_msg.orientation_covariance[0] = 1.0e6;
    imu_msg.orientation_covariance[4] = 1.0e6;
    imu_msg.orientation_covariance[8] = yaw_var;
    // Raw IMU protocol uses RFU: x=right, y=front, z=up. ROS body convention is FLU.
    imu_msg.angular_velocity.x = gy_rfu_dps * kPi / 180.0;
    imu_msg.angular_velocity.y = -gx_rfu_dps * kPi / 180.0;
    imu_msg.angular_velocity.z = gz_rfu_dps * kPi / 180.0;
    imu_msg.linear_acceleration.x = ay_rfu_mps2;
    imu_msg.linear_acceleration.y = -ax_rfu_mps2;
    imu_msg.linear_acceleration.z = az_rfu_mps2;
    imu_pub_.publish(imu_msg);

    // Raw protocol IMU in the original RFU axes and original physical units.
    r300_1x_navigation::InsImu ins_imu_msg;
    ins_imu_msg.header.stamp = stamp;
    ins_imu_msg.header.frame_id = "one_x_rfu";
    ins_imu_msg.counter = counter;
    ins_imu_msg.gyro_x_raw = gx_raw;
    ins_imu_msg.gyro_y_raw = gy_raw;
    ins_imu_msg.gyro_z_raw = gz_raw;
    ins_imu_msg.gyro_x_dps = gx_rfu_dps;
    ins_imu_msg.gyro_y_dps = gy_rfu_dps;
    ins_imu_msg.gyro_z_dps = gz_rfu_dps;
    ins_imu_msg.accel_x_raw = ax_raw;
    ins_imu_msg.accel_y_raw = ay_raw;
    ins_imu_msg.accel_z_raw = az_raw;
    ins_imu_msg.accel_x_mps2 = ax_rfu_mps2;
    ins_imu_msg.accel_y_mps2 = ay_rfu_mps2;
    ins_imu_msg.accel_z_mps2 = az_rfu_mps2;
    ins_imu_pub_.publish(ins_imu_msg);

    r300_1x_navigation::Attitude attitude_msg;
    attitude_msg.header.stamp = stamp;
    attitude_msg.header.frame_id = base_frame_;
    attitude_msg.counter = counter;
    attitude_msg.pitch_raw = pitch_raw;
    attitude_msg.roll_raw = roll_raw;
    attitude_msg.heading_raw = heading_raw;
    attitude_msg.pitch_deg = pitch_deg;
    attitude_msg.roll_deg = roll_deg;
    attitude_msg.heading_deg = heading_deg;
    attitude_pub_.publish(attitude_msg);

    r300_1x_navigation::Velocity velocity_msg;
    velocity_msg.header.stamp = stamp;
    velocity_msg.header.frame_id = odom_frame_;
    velocity_msg.counter = counter;
    velocity_msg.ve_raw = ve_raw;
    velocity_msg.vn_raw = vn_raw;
    velocity_msg.vd_raw = vd_raw;
    velocity_msg.ve_mps = ve_mps;
    velocity_msg.vn_mps = vn_mps;
    velocity_msg.vu_mps = -vd_mps;
    velocity_pub_.publish(velocity_msg);

    r300_1x_navigation::UpdateFlag update_flag_msg;
    update_flag_msg.header.stamp = stamp;
    update_flag_msg.header.frame_id = base_frame_;
    update_flag_msg.counter = counter;
    update_flag_msg.raw = update_flag;
    update_flag_msg.gnss_position_updated = (update_flag & (1U << 0U)) != 0U;
    update_flag_msg.gnss_velocity_updated = (update_flag & (1U << 1U)) != 0U;
    update_flag_msg.gnss_attitude_updated = (update_flag & (1U << 2U)) != 0U;
    update_flag_msg.dvl_bottom_velocity_updated = (update_flag & (1U << 3U)) != 0U;
    update_flag_msg.dvl_water_velocity_updated = (update_flag & (1U << 4U)) != 0U;
    update_flag_msg.usbl_position_updated = (update_flag & (1U << 5U)) != 0U;
    update_flag_msg.emc_attitude_updated = (update_flag & (1U << 6U)) != 0U;
    update_flag_msg.odo_velocity_updated = (update_flag & (1U << 7U)) != 0U;
    std::vector<std::string> active_updates;
    if (update_flag_msg.gnss_position_updated) active_updates.push_back("GNSS位置更新");
    if (update_flag_msg.gnss_velocity_updated) active_updates.push_back("GNSS速度更新");
    if (update_flag_msg.gnss_attitude_updated) active_updates.push_back("GNSS姿态更新");
    if (update_flag_msg.dvl_bottom_velocity_updated) active_updates.push_back("DVL对底速度更新");
    if (update_flag_msg.dvl_water_velocity_updated) active_updates.push_back("DVL对水速度更新");
    if (update_flag_msg.usbl_position_updated) active_updates.push_back("USBL位置更新");
    if (update_flag_msg.emc_attitude_updated) active_updates.push_back("EMC姿态更新");
    if (update_flag_msg.odo_velocity_updated) active_updates.push_back("ODO速度更新");
    update_flag_msg.active_updates = JoinStrings(active_updates, "本帧无外部信息更新");
    update_flag_pub_.publish(update_flag_msg);

    PublishInsStatus(ins_status, counter, stamp);

    std_msgs::Float64 heading_msg;
    heading_msg.data = heading_deg;
    heading_pub_.publish(heading_msg);

    // Keep the latest fields for the low-rate, human-readable monitoring topics.
    latest_counter_ = counter;
    latest_ins_latitude_deg_ = ins_latitude_deg;
    latest_ins_longitude_deg_ = ins_longitude_deg;
    latest_gps_latitude_deg_ = gps_latitude_deg;
    latest_gps_longitude_deg_ = gps_longitude_deg;
    latest_ins_status_ = ins_status;
    latest_gps_status_ = gps_status;
    latest_imu_status_ = imu_status;
    latest_update_flag_ = update_flag;
    have_latest_position_ = true;
    latest_temperature_c_ = temperature_c;
    latest_vx_enu_ = vx_enu;
    latest_vy_enu_ = vy_enu;
    latest_vz_enu_ = vz_enu;
  }

  bool SetCurrentOriginCallback(std_srvs::Trigger::Request &,
                                std_srvs::Trigger::Response &response)
  {
    if (!have_latest_control_position_)
    {
      response.success = false;
      response.message = "No valid 1X control position has been decoded yet";
      return true;
    }

    const ros::Time now = ros::Time::now();
    const double age_s = latest_control_position_stamp_.isZero()
                             ? std::numeric_limits<double>::infinity()
                             : (now - latest_control_position_stamp_).toSec();
    if (!std::isfinite(age_s) || age_s < 0.0 || age_s > origin_set_max_age_s_)
    {
      std::ostringstream message;
      message << "Latest valid 1X position is stale: age=" << std::fixed
              << std::setprecision(3) << age_s << "s, limit="
              << origin_set_max_age_s_ << "s";
      response.success = false;
      response.message = message.str();
      return true;
    }

    origin_ = latest_control_position_;
    origin_ready_ = true;

    // Prevent a false angular-rate spike after re-anchoring between navigation runs.
    have_last_yaw_ = false;
    last_yaw_time_ = ros::Time(0);
    last_yaw_ = 0.0;

    PublishOrigin(now);

    std::ostringstream message;
    message << std::fixed << std::setprecision(10)
            << "Current 1X position set as ENU origin: lat=" << origin_.lat_deg
            << ", lon=" << origin_.lon_deg
            << std::setprecision(3) << ", alt=" << origin_.alt_m
            << ", sample_age=" << age_s << "s";
    response.success = true;
    response.message = message.str();

    ROS_WARN_STREAM(response.message);
    return true;
  }

  void PublishOrigin(const ros::Time &stamp)
  {
    sensor_msgs::NavSatFix origin_msg;
    origin_msg.header.stamp = stamp;
    origin_msg.header.frame_id = fix_frame_;
    origin_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
    origin_msg.status.service = 0U;
    origin_msg.latitude = origin_.lat_deg;
    origin_msg.longitude = origin_.lon_deg;
    origin_msg.altitude = origin_.alt_m;
    origin_msg.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
    origin_pub_.publish(origin_msg);
  }

  void PosCompareTimerCallback(const ros::TimerEvent &)
  {
    // Do not publish an uninitialised all-zero comparison before receiving a valid frame.
    if (!have_latest_position_)
    {
      return;
    }

    std_msgs::String msg;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(10)
       << "counter=" << latest_counter_
       << " | INS(lat_deg=" << latest_ins_latitude_deg_
       << ", lon_deg=" << latest_ins_longitude_deg_ << ")"
       << " | GPS(lat_deg=" << latest_gps_latitude_deg_
       << ", lon_deg=" << latest_gps_longitude_deg_
       << ", status=" << static_cast<unsigned int>(latest_gps_status_) << ")";
    msg.data = ss.str();
    pos_compare_pub_.publish(msg);
  }

  static const char *InsStatusBitName(unsigned int bit)
  {
    // Bytes 32..33: unsigned 16-bit little-endian INS status.
    switch (bit)
    {
      case 0U: return "准备（待机）";
      case 1U: return "粗对准";
      case 2U: return "精对准";
      case 3U: return "纯惯性导航";
      case 4U: return "GNSS组合导航";
      case 5U: return "DVL组合导航";
      case 6U: return "GNSS+DVL组合导航";
      case 7U: return "无位置参考";
      case 8U: return "装订位置/参考位置修正";
      case 9U: return "GNSS位置";
      case 10U: return "无速度参考";
      case 11U: return "零速";
      case 12U: return "GNSS速度";
      case 13U: return "DVL速度";
      case 14U: return "INS数据有效";
      case 15U: return "故障";
      default: return "未知";
    }
  }

  static bool StatusBit(uint16_t status, unsigned int bit)
  {
    return (status & (static_cast<uint16_t>(1U) << bit)) != 0U;
  }

  void PublishInsStatus(uint16_t status, uint16_t counter, const ros::Time &stamp)
  {
    r300_1x_navigation::InsStatus msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = base_frame_;
    msg.counter = counter;
    msg.raw = status;
    msg.ready = StatusBit(status, 0U);
    msg.coarse_alignment = StatusBit(status, 1U);
    msg.fine_alignment = StatusBit(status, 2U);
    msg.pure_ins_navigation = StatusBit(status, 3U);
    msg.gnss_integrated_navigation = StatusBit(status, 4U);
    msg.dvl_integrated_navigation = StatusBit(status, 5U);
    msg.gnss_dvl_integrated_navigation = StatusBit(status, 6U);
    msg.no_position_reference = StatusBit(status, 7U);
    msg.bound_position = StatusBit(status, 8U);
    msg.gnss_position = StatusBit(status, 9U);
    msg.no_velocity_reference = StatusBit(status, 10U);
    msg.zero_velocity = StatusBit(status, 11U);
    msg.gnss_velocity = StatusBit(status, 12U);
    msg.dvl_velocity = StatusBit(status, 13U);
    msg.ins_data_valid = StatusBit(status, 14U);
    msg.fault = StatusBit(status, 15U);

    if (msg.fault)
    {
      msg.work_state = "故障";
    }
    else if (msg.pure_ins_navigation || msg.gnss_integrated_navigation ||
             msg.dvl_integrated_navigation || msg.gnss_dvl_integrated_navigation)
    {
      msg.work_state = "导航";
    }
    else if (msg.coarse_alignment || msg.fine_alignment)
    {
      msg.work_state = "对准";
    }
    else if (msg.ready)
    {
      msg.work_state = "准备（待机）";
    }
    else
    {
      msg.work_state = "状态未标记";
    }

    std::vector<std::string> navigation_modes;
    if (msg.pure_ins_navigation) navigation_modes.push_back("纯惯性导航");
    if (msg.gnss_integrated_navigation) navigation_modes.push_back("GNSS组合导航");
    if (msg.dvl_integrated_navigation) navigation_modes.push_back("DVL组合导航");
    if (msg.gnss_dvl_integrated_navigation) navigation_modes.push_back("GNSS+DVL组合导航");
    msg.navigation_mode = JoinStrings(navigation_modes, "无导航模式");

    std::vector<std::string> position_references;
    if (msg.no_position_reference) position_references.push_back("无位置参考");
    if (msg.bound_position) position_references.push_back("装订位置/参考位置修正");
    if (msg.gnss_position) position_references.push_back("GNSS位置");
    msg.position_reference = JoinStrings(position_references, "位置参考未标记");

    std::vector<std::string> velocity_references;
    if (msg.no_velocity_reference) velocity_references.push_back("无速度参考");
    if (msg.zero_velocity) velocity_references.push_back("零速");
    if (msg.gnss_velocity) velocity_references.push_back("GNSS速度");
    if (msg.dvl_velocity) velocity_references.push_back("DVL速度");
    msg.velocity_reference = JoinStrings(velocity_references, "速度参考未标记");

    std::vector<std::string> active_bits;
    for (unsigned int bit = 0U; bit < 16U; ++bit)
    {
      if (StatusBit(status, bit))
      {
        std::ostringstream bit_text;
        bit_text << "b" << bit << "=" << InsStatusBitName(bit);
        active_bits.push_back(bit_text.str());
      }
    }
    msg.active_bits = JoinStrings(active_bits, "无有效状态位");

    std::ostringstream summary;
    summary << "工作状态=" << msg.work_state
            << " | 导航模式=" << msg.navigation_mode
            << " | 位置参考=" << msg.position_reference
            << " | 速度参考=" << msg.velocity_reference
            << " | INS数据有效=" << (msg.ins_data_valid ? "是" : "否")
            << " | 故障=" << (msg.fault ? "是" : "否")
            << " | raw=" << status << " (0x"
            << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << status
            << std::dec << std::setfill(' ') << ")";
    msg.summary = summary.str();

    ins_status_pub_.publish(msg);
  }

  void DiagnosticsTimerCallback(const ros::TimerEvent &)
  {
    diagnostic_msgs::DiagnosticArray array_msg;
    array_msg.header.stamp = ros::Time::now();

    diagnostic_msgs::DiagnosticStatus status;
    status.name = "1X INS serial navigation";
    status.hardware_id = serial_port_;
    const double age_s = last_valid_frame_time_.isZero() ? std::numeric_limits<double>::infinity()
                                                          : (ros::Time::now() - last_valid_frame_time_).toSec();
    if (serial_fd_ < 0 || age_s > 0.30)
    {
      status.level = diagnostic_msgs::DiagnosticStatus::ERROR;
      status.message = "No fresh checksum-valid 1X frame";
    }
    else if (checksum_failures_ > 0U || counter_anomalies_ > 0U)
    {
      status.level = diagnostic_msgs::DiagnosticStatus::WARN;
      status.message = "Frames are arriving, but parser health warnings exist";
    }
    else
    {
      status.level = diagnostic_msgs::DiagnosticStatus::OK;
      status.message = "1X frames are fresh";
    }

    auto add_kv = [&status](const std::string &key, const std::string &value)
    {
      diagnostic_msgs::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

    add_kv("valid_frames", ToString(valid_frames_));
    add_kv("checksum_failures", ToString(checksum_failures_));
    add_kv("malformed_frames", ToString(malformed_frames_));
    add_kv("skipped_bytes", ToString(skipped_bytes_));
    add_kv("counter_anomalies", ToString(counter_anomalies_));
    add_kv("last_valid_frame_age_s", std::isfinite(age_s) ? ToString(age_s, 3) : "inf");
    add_kv("origin_mode", origin_mode_);
    add_kv("origin_ready", origin_ready_ ? "true" : "false");
    add_kv("ins_status", std::to_string(latest_ins_status_));
    add_kv("gps_status", std::to_string(latest_gps_status_));
    add_kv("imu_status", std::to_string(latest_imu_status_));
    add_kv("update_flag", std::to_string(latest_update_flag_));
    add_kv("temperature_c", ToString(latest_temperature_c_, 2));
    add_kv("vx_enu_mps", ToString(latest_vx_enu_));
    add_kv("vy_enu_mps", ToString(latest_vy_enu_));
    add_kv("vz_enu_mps", ToString(latest_vz_enu_));

    array_msg.status.push_back(status);
    diagnostics_pub_.publish(array_msg);
  }

  // Latest decoded protocol data used by /one_x/pos_compare and diagnostics.
  bool have_latest_position_;
  uint16_t latest_counter_ = 0U;
  double latest_ins_latitude_deg_ = 0.0;
  double latest_ins_longitude_deg_ = 0.0;
  double latest_gps_latitude_deg_ = 0.0;
  double latest_gps_longitude_deg_ = 0.0;
  uint16_t latest_ins_status_ = 0U;
  uint8_t latest_gps_status_ = 0U;
  uint16_t latest_imu_status_ = 0U;
  uint8_t latest_update_flag_ = 0U;
  double latest_temperature_c_ = 0.0;
  double latest_vx_enu_ = 0.0;
  double latest_vy_enu_ = 0.0;
  double latest_vz_enu_ = 0.0;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "one_x_serial_driver");
  try
  {
    OneXSerialDriver driver;
    driver.Spin();
  }
  catch (const std::exception &e)
  {
    ROS_FATAL("1X serial driver failed: %s", e.what());
    return 1;
  }
  return 0;
}
