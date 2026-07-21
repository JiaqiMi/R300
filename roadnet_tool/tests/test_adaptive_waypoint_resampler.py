import math
import tempfile
import unittest

from roadnet.adaptive_waypoint_resampler import (
    AdaptiveWaypointConfig,
    adaptive_resample_waypoints,
    generate_vehicle_waypoints_adaptive,
)


class MetricCalibration:
    is_valid = True

    @staticmethod
    def pixel_to_world(x, y):
        return float(x), -float(y)

    @staticmethod
    def pixel_to_wgs84(x, y):
        return 117.0 + float(x) * 1e-5, 39.0 - float(y) * 1e-5


class AdaptiveWaypointResamplerTests(unittest.TestCase):
    def test_straight_path_uses_about_ten_metre_spacing(self):
        dense = [[x, 0] for x in range(101)]
        result = adaptive_resample_waypoints(dense, MetricCalibration())
        self.assertEqual(result.report["coordinate_mode"], "enu_meter")
        # densify may add midpoints; vehicle waypoints still ~10m
        self.assertGreaterEqual(result.report["dense_path_point_count"], 101)
        self.assertLessEqual(result.report["sparse_waypoint_count"], 14)
        self.assertAlmostEqual(result.report["average_spacing_m"], 10.0, delta=1.5)
        self.assertEqual(result.waypoints[0]["tag"], "start")
        self.assertEqual(result.waypoints[-1]["tag"], "goal")

    def test_rdp_sparse_vertices_still_resample_to_ten_metres(self):
        # Simulate RDP-simplified centerline: vertices every ~50m
        sparse_vertices = [[x, 0] for x in range(0, 201, 50)]
        result = generate_vehicle_waypoints_adaptive(
            sparse_vertices, geo_calibration=MetricCalibration()
        )
        self.assertGreater(result.report["dense_path_point_count"], len(sparse_vertices))
        self.assertGreaterEqual(result.report["vehicle_waypoint_count"], 18)
        self.assertLessEqual(result.report.get("max_spacing_m", 99), 12.0)
        modes = {wp["spacing_mode"] for wp in result.waypoints}
        self.assertIn("straight_10m", modes)

    def test_sharp_corner_is_forced_and_locally_denser(self):
        dense = [[x, 0] for x in range(51)] + [[50, y] for y in range(1, 51)]
        result = adaptive_resample_waypoints(dense, MetricCalibration())
        sharp = [item for item in result.waypoints if item["tag"] == "sharp_turn"]
        self.assertTrue(sharp)
        self.assertTrue(any(math.hypot(item["x_pixel"] - 50, item["y_pixel"]) < 1e-6 for item in sharp))
        local = [item for item in result.waypoints if 44 <= item["path_distance_m"] <= 56]
        self.assertGreaterEqual(len(local), 5)

    def test_intersection_and_all_task_points_are_forced(self):
        dense = [[x, 0] for x in range(101)]
        graph = {
            "nodes": [
                {"id": 1, "x": 50, "y": 0},
                {"id": 2, "x": 40, "y": 0},
                {"id": 3, "x": 60, "y": 0},
                {"id": 4, "x": 50, "y": 10},
            ],
            "edges": [
                {"start": 1, "end": 2},
                {"start": 1, "end": 3},
                {"start": 1, "end": 4},
            ],
        }
        tasks = [
            {"seq": 1, "point_type": 0, "snapped_x": 0, "snapped_y": 0},
            {"seq": 2, "point_type": 2, "snapped_x": 33, "snapped_y": 0},
            {"seq": 3, "point_type": 1, "snapped_x": 100, "snapped_y": 0},
        ]
        result = adaptive_resample_waypoints(
            dense, MetricCalibration(), graph, tasks, path_node_sequence=[2, 1, 3]
        )
        self.assertTrue(any(item["tag"] == "intersection" and item["forced"] for item in result.waypoints))
        self.assertTrue(any(item["tag"] == "task" and abs(item["x_pixel"] - 33) < 1e-6 for item in result.waypoints))
        self.assertFalse(result.waypoints[0]["pass_through"])
        self.assertTrue(next(item for item in result.waypoints if item["tag"] == "task")["pass_through"])
        self.assertFalse(result.waypoints[-1]["pass_through"])

    def test_yaw_speed_and_pixel_fallback_report(self):
        result = adaptive_resample_waypoints([[0, 0], [20, 0]], None)
        self.assertEqual(result.report["coordinate_mode"], "image_pixel_fallback")
        self.assertTrue(result.report["warnings"])
        self.assertEqual(result.sparse_waypoints_geo, [])
        self.assertAlmostEqual(result.waypoints[0]["yaw_deg"], 0.0)
        self.assertIn("target_speed_mps", result.waypoints[0])
        self.assertIn("arrival_radius_m", result.waypoints[0])

    def test_failed_snap_preserves_original_mandatory_task_point(self):
        task = {
            "seq": 2, "point_type": 2, "status": "failed",
            "original_x": 37, "original_y": 0,
            "snapped_x": 0, "snapped_y": 0,
        }
        result = adaptive_resample_waypoints(
            [[x, 0] for x in range(101)], MetricCalibration(),
            snapped_task_points=[task],
        )
        preserved = [item for item in result.waypoints if item.get("task_seq") == 2]
        self.assertEqual(len(preserved), 1)
        self.assertAlmostEqual(preserved[0]["x_pixel"], 37.0)
        self.assertEqual(preserved[0]["task_status"], "failed")
        self.assertEqual(result.report["failed_task_points_preserved"], 1)

    def test_report_can_be_saved(self):
        with tempfile.TemporaryDirectory() as output_dir:
            adaptive_resample_waypoints(
                [[0, 0], [20, 0]], MetricCalibration(), output_dir=output_dir
            )
            import os
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "waypoint_resample_report.json")))


if __name__ == "__main__":
    unittest.main()
