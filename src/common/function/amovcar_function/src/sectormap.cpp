#include "sectormap.h"

#include <cmath>
#include <algorithm>

void SectorMap::SetUavPosition(Point2D& uav) {
	Uavp.x = uav.x;
	Uavp.y = uav.y;
}

void SectorMap::SetUavHeading(float hd) {
	heading = hd;
	heading += 270;
	heading = (int)heading % 360;
	heading = 360 - heading;
}

void SectorMap::ComputeMV(vector<float> r) {
	float dist[360] = { 0 };
	ranges.clear();
	map_cv.clear();
	int range_size = r.size();

	for (size_t i = 0; i < range_size; i++)
	{
		//A non-zero value (true) if x is a NaN value; and zero (false) otherwise.

		//isinf A non-zero value (true) if x is an infinity; and zero (false) otherwise.
		if (!std::isnan(r[i]) && !std::isinf(r[i]))
		{
			float scan_distance = r[i];
			int sector_index = std::floor((i*angle_resolution) / sector_value);
			if (scan_distance >= scan_distance_max || scan_distance < scan_distance_min)
				scan_distance = 0;
			else
				scan_distance = scan_distance_max - scan_distance;

			dist[sector_index] += scan_distance;
		}
		ranges.push_back(r[i]);
	}

	for (int j = 0; j < (int)(360 / sector_value); j++)
	{
		map_cv.push_back(dist[j]);
	}
}

bool SectorMap::IsFrontSafety()
{
	float goal_sector = (int)(0 - (sector_value - sector_scale) + 360) % 360;
	int start_index = goal_sector / angle_resolution;
	float scan_distance = 0;
	for (int i = 0; i < (sector_value - sector_scale) * 2 / angle_resolution; i++)
	{
		int real_index = (start_index + i) % (int)(360 / angle_resolution);
		if (!std::isnan(ranges[real_index]) && !std::isinf(ranges[real_index]))
		{
			if (ranges[real_index] < scan_distance_max && ranges[real_index] >= scan_distance_min)
				scan_distance = scan_distance_max - ranges[real_index] + scan_distance;
		}
	}
	if (scan_distance < 0.1)
	{
		return true;
	}

	return false;
}

float SectorMap::CalculDirection(Point2D& goal) {
	float ori;
	//Compute arc tangent with two parameters
	//return Principal arc tangent of y/x, in the interval [-pi,+pi] radians.
	//One radian is equivalent to 180/PI degrees.
	float G_theta = atan2((goal.y - Uavp.y), (goal.x - Uavp.x));
	float goal_ori = G_theta * 180 / PI;
	if (goal_ori < 0)
	{
		goal_ori += 360;
	}
	//heading = 90
	goal_ori -= heading;
	goal_ori += 360;
	goal_ori = (int)goal_ori % 360;

	float goal_sector = (int)(goal_ori - sector_value + 360) % 360;
	int start_index = goal_sector / angle_resolution;
	float scan_distance = 0;
	for (int i = 0; i < sector_value * 2 / angle_resolution; i++)
	{
		int real_index = (start_index + i) % (int)(360 / angle_resolution);
		if (!std::isnan(ranges[real_index]) && !std::isinf(ranges[real_index]))
		{
			if (ranges[real_index] < scan_distance_max && ranges[real_index] >= scan_distance_min)
				scan_distance = scan_distance_max - ranges[real_index] + scan_distance;
		}
	}
	if (scan_distance < 0.1)
	{
		ori = goal_ori;
		ori += heading;
		ori = (int)ori % 360;

		return ori;
	}

	vector<int> mesh;
	for (int i = 0; i < map_cv.size(); i++)
	{
		if (map_cv[i] < 0.1)
			mesh.push_back(0);
		else if (map_cv[i] >= 0.1 && map_cv[i] < 0.3)
			mesh.push_back(2);
		else
			mesh.push_back(4);
	}

	vector<float> cand_dir;
	for (int j = 0; j < mesh.size(); j++)
	{
		if (j == mesh.size() - 1)
		{
			if (mesh[0] + mesh[mesh.size() - 1] == 0)
				cand_dir.push_back(0.0);
		}
		else
		{
			if (mesh[j] + mesh[j + 1] == 0)
				cand_dir.push_back((j + 1)*sector_value);
		}
	}

	if (cand_dir.size() != 0) {
		vector<float> delta;
		for (auto &dir_ite : cand_dir) {
			float delte_theta1 = fabs(dir_ite - goal_ori);
			float delte_theta2 = 360 - delte_theta1;
			float delte_theta = delte_theta1 < delte_theta2 ? delte_theta1 : delte_theta2;
			delta.push_back(delte_theta);
		}
		int min_index = min_element(delta.begin(), delta.end()) - delta.begin();
		ori = cand_dir.at(min_index);

		ori += heading;
		ori = (int)ori % 360;

		return ori;
	}

	return -1;
}
