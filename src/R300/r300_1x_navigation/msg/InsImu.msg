# Raw 1X IMU data decoded directly from the serial frame.
# Protocol axes are preserved without RFU->FLU remapping:
#   x = right, y = front, z = up.
# Gyroscope physical values are degrees per second.
# Accelerometer physical values are metres per second squared.
std_msgs/Header header
uint16 counter

int32 gyro_x_raw
int32 gyro_y_raw
int32 gyro_z_raw
float64 gyro_x_dps
float64 gyro_y_dps
float64 gyro_z_dps

int32 accel_x_raw
int32 accel_y_raw
int32 accel_z_raw
float64 accel_x_mps2
float64 accel_y_mps2
float64 accel_z_mps2
