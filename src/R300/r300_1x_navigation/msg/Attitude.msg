# Native 1X attitude fields decoded from the serial frame.
# Values are not converted to ROS ENU yaw conventions.
std_msgs/Header header
uint16 counter

int16 pitch_raw
int16 roll_raw
uint16 heading_raw
float64 pitch_deg
float64 roll_deg
float64 heading_deg
