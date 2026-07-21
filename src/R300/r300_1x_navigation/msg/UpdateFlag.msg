# Byte 107 external-information update flags.
std_msgs/Header header
uint16 counter

uint8 raw
bool gnss_position_updated
bool gnss_velocity_updated
bool gnss_attitude_updated
bool dvl_bottom_velocity_updated
bool dvl_water_velocity_updated
bool usbl_position_updated
bool emc_attitude_updated
bool odo_velocity_updated
string active_updates
