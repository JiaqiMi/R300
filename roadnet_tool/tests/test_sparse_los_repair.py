"""Tests for cutting-corner LOS repair and edge polyline expansion."""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.adaptive_waypoint_resampler import (  # noqa: E402
    AdaptiveWaypointConfig,
    adaptive_resample_waypoints,
    fix_sparse_cutting_corners,
    repair_sparse_cutting_corners,
    validate_sparse_line_of_sight,
)
from roadnet.global_planner import (  # noqa: E402
    EdgeGeometryMissingError,
    expand_edge_path_to_polyline,
)


class MetricCalibration:
    is_valid = True
    pixel_resolution_estimated_m = 1.0

    @staticmethod
    def pixel_to_world(x, y):
        return float(x), -float(y)

    @staticmethod
    def pixel_to_wgs84(x, y):
        return 117.0 + float(x) * 1e-5, 39.0 - float(y) * 1e-5


class EdgePolylineExpandTests(unittest.TestCase):
    def test_uses_polyline_points(self):
        nodes = [{"id": 1, "x": 0, "y": 0}, {"id": 2, "x": 100, "y": 0}]
        edges = [{
            "id": 10, "start": 1, "end": 2,
            "points_pixel": [[0, 0], [50, 20], [100, 0]],
        }]
        poly = expand_edge_path_to_polyline([1, 2], [10], nodes, edges)
        self.assertGreaterEqual(len(poly), 3)
        self.assertTrue(any(abs(p[1] - 20) < 1e-6 for p in poly))

    def test_missing_polyline_raises_edge_geometry_missing(self):
        nodes = [{"id": 1, "x": 0, "y": 0}, {"id": 2, "x": 100, "y": 0}]
        edges = [{"id": 10, "start": 1, "end": 2, "points_pixel": []}]
        with self.assertRaises(EdgeGeometryMissingError) as caught:
            expand_edge_path_to_polyline([1, 2], [10], nodes, edges)
        self.assertIn("edge_geometry_missing", str(caught.exception))


class SparseLosRepairTests(unittest.TestCase):
    def test_defaults_match_competition_spacing(self):
        cfg = AdaptiveWaypointConfig()
        self.assertAlmostEqual(cfg.straight_spacing_m, 10.0)
        self.assertAlmostEqual(cfg.curve_spacing_m, 2.0)
        self.assertAlmostEqual(cfg.intersection_spacing_m, 2.0)
        self.assertAlmostEqual(cfg.task_point_buffer_m, 5.0)
        self.assertAlmostEqual(cfg.max_chord_error_m, 1.0)
        self.assertAlmostEqual(cfg.min_mask_support_ratio, 0.75)
        self.assertEqual(cfg.max_insert_iterations, 6)

    def test_chord_across_curve_is_repaired_by_insert(self):
        # Dense path follows an L bend; sparse only keeps endpoints → large chord error
        dense = [[x, 0.0] for x in range(0, 41)] + [[40.0, y] for y in range(1, 41)]
        metric = [[p[0], -p[1]] for p in dense]
        from roadnet.adaptive_waypoint_resampler import _cumulative_distance
        cumulative = _cumulative_distance(metric)
        sparse_pixel = [dense[0], dense[-1]]
        sparse_s = [0.0, cumulative[-1]]
        config = AdaptiveWaypointConfig(max_chord_error_m=1.5, max_insert_iterations=5)
        repaired = repair_sparse_cutting_corners(
            sparse_pixel, sparse_s, dense, metric, cumulative,
            config=config, road_mask=None, metres_per_pixel=1.0,
        )
        self.assertGreater(repaired["inserted_midpoint_count"], 0)
        self.assertGreater(len(repaired["sparse_waypoints_pixel"]), 2)
        self.assertTrue(repaired["geometry_valid"])

    def test_mask_support_rejects_offroad_chord(self):
        # Road is a thin horizontal strip; chord goes through empty area above
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[38:43, :] = 255
        dense = [[x, 40.0] for x in range(0, 101)]
        metric = [[p[0], -p[1]] for p in dense]
        from roadnet.adaptive_waypoint_resampler import _cumulative_distance
        cumulative = _cumulative_distance(metric)
        # Fake sparse that jumps above the road
        sparse_pixel = [[0.0, 40.0], [50.0, 10.0], [100.0, 40.0]]
        sparse_s = [0.0, 50.0, 100.0]
        rows, ok = validate_sparse_line_of_sight(
            sparse_pixel, sparse_s, dense, metric, cumulative,
            config=AdaptiveWaypointConfig(min_mask_support_ratio=0.75),
            road_mask=mask, metres_per_pixel=1.0,
        )
        self.assertFalse(ok)
        self.assertTrue(any("mask_support_low" in r["reason"] for r in rows if r["is_suspicious"]))

    def test_adaptive_resample_reports_geometry_valid_on_straight(self):
        dense = [[x, 0] for x in range(101)]
        result = adaptive_resample_waypoints(dense, MetricCalibration())
        self.assertIn("geometry_valid", result.report)
        self.assertTrue(result.report["geometry_valid"])

    def test_fix_api_inserts_and_marks_valid(self):
        dense = [[x, 0.0] for x in range(0, 51)] + [[50.0, y] for y in range(1, 51)]
        sparse = [
            {"seq": 1, "x_pixel": 0.0, "y_pixel": 0.0, "tag": "start"},
            {"seq": 2, "x_pixel": 50.0, "y_pixel": 50.0, "tag": "goal"},
        ]
        out = fix_sparse_cutting_corners(dense, sparse, MetricCalibration())
        self.assertTrue(out["ok"])
        self.assertGreater(out["inserted_midpoint_count"], 0)
        self.assertTrue(out["geometry_valid"])


class UiWiringTests(unittest.TestCase):
    def test_fix_button_exists(self):
        path = os.path.join(ROOT, "gui", "parameter_panel.py")
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn("修复稀疏航点切弯", source)
        self.assertIn('fix_sparse_cutting_corners', source)


if __name__ == "__main__":
    unittest.main()
