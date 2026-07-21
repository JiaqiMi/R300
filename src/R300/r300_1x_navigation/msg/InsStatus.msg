# Decoded 16-bit INS status word (bytes 32..33).
std_msgs/Header header
uint16 counter

uint16 raw
bool ready
bool coarse_alignment
bool fine_alignment
bool pure_ins_navigation
bool gnss_integrated_navigation
bool dvl_integrated_navigation
bool gnss_dvl_integrated_navigation
bool no_position_reference
bool bound_position
bool gnss_position
bool no_velocity_reference
bool zero_velocity
bool gnss_velocity
bool dvl_velocity
bool ins_data_valid
bool fault

string work_state
string navigation_mode
string position_reference
string velocity_reference
string active_bits
string summary
