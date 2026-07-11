#ifndef R300_1X_NAVIGATION_GEODESY_HPP
#define R300_1X_NAVIGATION_GEODESY_HPP

#include <cmath>

namespace r300_1x_navigation
{

struct Geodetic
{
  double lat_deg;
  double lon_deg;
  double alt_m;
};

struct Ecef
{
  double x;
  double y;
  double z;
};

struct Enu
{
  double east;
  double north;
  double up;
};

inline double DegToRad(double deg)
{
  return deg * M_PI / 180.0;
}

inline Ecef GeodeticToEcef(const Geodetic &geo)
{
  // WGS-84 ellipsoid.
  const double a = 6378137.0;
  const double f = 1.0 / 298.257223563;
  const double e2 = f * (2.0 - f);

  const double lat = DegToRad(geo.lat_deg);
  const double lon = DegToRad(geo.lon_deg);
  const double sin_lat = std::sin(lat);
  const double cos_lat = std::cos(lat);
  const double sin_lon = std::sin(lon);
  const double cos_lon = std::cos(lon);
  const double n = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);

  Ecef out;
  out.x = (n + geo.alt_m) * cos_lat * cos_lon;
  out.y = (n + geo.alt_m) * cos_lat * sin_lon;
  out.z = (n * (1.0 - e2) + geo.alt_m) * sin_lat;
  return out;
}

inline Enu GeodeticToEnu(const Geodetic &point, const Geodetic &origin)
{
  const Ecef p = GeodeticToEcef(point);
  const Ecef o = GeodeticToEcef(origin);
  const double dx = p.x - o.x;
  const double dy = p.y - o.y;
  const double dz = p.z - o.z;

  const double lat0 = DegToRad(origin.lat_deg);
  const double lon0 = DegToRad(origin.lon_deg);
  const double sin_lat = std::sin(lat0);
  const double cos_lat = std::cos(lat0);
  const double sin_lon = std::sin(lon0);
  const double cos_lon = std::cos(lon0);

  Enu out;
  out.east = -sin_lon * dx + cos_lon * dy;
  out.north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
  out.up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
  return out;
}

}  // namespace r300_1x_navigation

#endif  // R300_1X_NAVIGATION_GEODESY_HPP
