# Native navigation-frame velocity fields decoded from the serial frame.
# ve: east velocity, vn: north velocity, vu: up velocity.
std_msgs/Header header
uint16 counter

int16 ve_raw
int16 vn_raw
int16 vd_raw
float64 ve_mps
float64 vn_mps
float64 vu_mps
