
修改后会新增两个高频话题：
/one_x/ins_fix   # 110 字节：Bytes 6~9 纬度，10~13 经度
/one_x/gps_fix   # 110 字节：Bytes 34~37 纬度，38~41 经度

同时保持：
/one_x/fix       # 当前导航控制位置，仍使用 GPS
/one_x/odom      # 当前导航控制位置生成的 ENU / TF，仍使用 GPS

核心代码会变成：
const double ins_latitude_deg =
    static_cast<double>(I32Le(frame + 6U)) * 180.0 / 2147483648.0;
const double ins_longitude_deg =
    static_cast<double>(I32Le(frame + 10U)) * 180.0 / 2147483648.0;

const double gps_latitude_deg =
    static_cast<double>(I32Le(frame + 34U)) * 180.0 / 2147483648.0;
const double gps_longitude_deg =
    static_cast<double>(I32Le(frame + 38U)) * 180.0 / 2147483648.0;

// 当前控制仍使用 GPS。
const double latitude_deg = gps_latitude_deg;
const double longitude_deg = gps_longitude_deg;

以后切回惯导位置控制时，只改这两行：

const double latitude_deg = ins_latitude_deg;
const double longitude_deg = ins_longitude_deg;

/one_x/pos_compare 也会修正为真正显示：

INS(lat_deg=..., lon_deg=...)
GPS(lat_deg=..., lon_deg=...)

// 保存一段时间的数据
rosbag record -O ~/one_x_position_compare.bag \
  /one_x/ins_fix \
  /one_x/gps_fix \
  /one_x/fix \
  /one_x/odom \
  /one_x/heading_deg