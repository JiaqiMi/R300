"""Tests for vehicle waypoint post-export validation."""

from __future__ import annotations

import math
import tempfile
import unittest

from roadnet.waypoint_validator import (
    WaypointValidationConfig,
    build_subject1_yaml_text,
    check_s_m_monotonic,
    remove_and_report_duplicate_waypoints,
    validate_subject1_yaml_text,
    validate_vehicle_waypoints,
)


class MetricCalibration:
    is_valid = True
    image_width = 500
    image_height = 500
    pixel_resolution_estimated_m = 1.0

    @staticmethod
    def pixel_to_world(x, y):
        return float(x), -float(y)

    @staticmethod
    def pixel_to_wgs84(x, y):
        return 117.0 + float(x) * 1e-5, 39.0 - float(y) * 1e-5


def _wp(seq, x, y, *, mode="straight_10m", tag="straight", s_m=None, dense_index=None, keep=False):
    lon, lat = MetricCalibration.pixel_to_wgs84(x, y)
    xe, ye = MetricCalibration.pixel_to_world(x, y)
    s = float(s_m if s_m is not None else x)
    return {
        "seq": seq,
        "name": f"wp_{seq:03d}",
        "x_pixel": float(x),
        "y_pixel": float(y),
        "x_enu": float(xe),
        "y_enu": float(ye),
        "longitude": lon,
        "latitude": lat,
        "longitude_deg": lon,
        "latitude_deg": lat,
        "altitude": 21.741,
        "altitude_m": 21.741,
        "tag": tag,
        "spacing_mode": mode,
        "source_mode": mode,
        "path_distance_m": s,
        "s_m": s,
        "dense_index": int(dense_index if dense_index is not None else round(s)),
        "segment_index": 0,
        "keep": bool(keep),
        "near_junction": False,
        "near_task_point": False,
        "local_angle_deg": 0.0,
        "mask_support_ratio": 1.0,
        "inside_image": True,
    }


class DuplicateCleanupTests(unittest.TestCase):
    def test_removes_consecutive_duplicates(self):
        wps = [_wp(1, 0, 0), _wp(2, 0.1, 0), _wp(3, 10, 0)]
        cleaned, report = remove_and_report_duplicate_waypoints(wps)
        self.assertEqual(report["duplicate_consecutive_count"], 1)
        self.assertEqual(report["duplicate_removed_count"], 1)
        self.assertEqual(len(cleaned), 2)

    def test_keeps_non_consecutive_near_duplicates(self):
        # Ring-like: first and last are near, but consecutive triples are not ABA
        wps = [
            _wp(1, 0, 0, s_m=0, dense_index=0),
            _wp(2, 25, 0, s_m=25, dense_index=25),
            _wp(3, 50, 0, s_m=50, dense_index=50),
            _wp(4, 75, 0, s_m=75, dense_index=75),
            _wp(5, 0.2, 0, s_m=100, dense_index=100),
        ]
        cleaned, report = remove_and_report_duplicate_waypoints(wps)
        self.assertEqual(len(cleaned), 5)
        self.assertGreaterEqual(report["non_consecutive_near_duplicate_count"], 1)
        self.assertEqual(report["duplicate_removed_count"], 0)
        self.assertEqual(report["aba_backtrack_count"], 0)

    def test_does_not_delete_keep_consecutive(self):
        wps = [
            _wp(1, 0, 0, tag="start", keep=True),
            _wp(2, 0.1, 0, tag="task", keep=True),
            _wp(3, 10, 0),
        ]
        cleaned, report = remove_and_report_duplicate_waypoints(wps)
        self.assertEqual(len(cleaned), 3)

    def test_auto_fixes_aba_backtrack(self):
        wps = [_wp(1, 0, 0), _wp(2, 10, 0), _wp(3, 0.05, 0)]
        cleaned, report = remove_and_report_duplicate_waypoints(wps)
        self.assertEqual(report["aba_backtrack_count"], 0)
        self.assertGreaterEqual(report["aba_fixed_count"], 1)
        self.assertEqual(len(cleaned), 2)

    def test_aba_keep_point_not_deleted(self):
        wps = [
            _wp(1, 0, 0),
            _wp(2, 10, 0, tag="task", keep=True),
            _wp(3, 0.05, 0),
        ]
        cleaned, report = remove_and_report_duplicate_waypoints(wps)
        self.assertGreaterEqual(report["aba_backtrack_count"], 1)
        self.assertEqual(len(cleaned), 3)


class YamlFormatTests(unittest.TestCase):
    def test_subject1_format(self):
        wps = [_wp(1, 0, 0), _wp(2, 10, 0), _wp(3, 20, 0)]
        text = build_subject1_yaml_text(wps)
        ok, errors = validate_subject1_yaml_text(text)
        self.assertTrue(ok, errors)


class MonotonicTests(unittest.TestCase):
    def test_s_m_monotonic(self):
        wps = [
            _wp(1, 0, 0, s_m=0),
            _wp(2, 10, 0, s_m=10),
            _wp(3, 20, 0, s_m=5),  # regress
        ]
        ok, bad = check_s_m_monotonic(wps)
        self.assertFalse(ok)
        self.assertEqual(bad[0]["reason"], "waypoint_order_invalid_by_s")


class ValidationPipelineTests(unittest.TestCase):
    def test_straight_path_passes(self):
        dense = [[x, 0] for x in range(0, 101)]
        wps = [
            _wp(i + 1, x, 0, s_m=float(x), dense_index=x)
            for i, x in enumerate(range(0, 101, 10))
        ]
        result = validate_vehicle_waypoints(
            wps,
            dense_path_pixel=dense,
            geo_calibration=MetricCalibration(),
            config=WaypointValidationConfig(hard_fail_spacing_m=20.0),
        )
        self.assertLessEqual(result.report["max_spacing_m"], 12.0)
        self.assertEqual(result.report["aba_backtrack_count"], 0)
        self.assertTrue(result.report.get("s_m_monotonic_valid", False))
        self.assertTrue(result.report["yaml_format_valid"])
        self.assertTrue(result.export_valid, result.report.get("failure_reasons"))
        self.assertIsNotNone(result.yaml_text)
        # dense_index / s_m preserved
        for wp in result.waypoints:
            self.assertIn("dense_index", wp)
            self.assertIn("s_m", wp)

    def test_ring_near_duplicate_does_not_block_export(self):
        # Long path that returns near start geographically but s_m still increases
        dense = [[x, 0] for x in range(0, 81)] + [[80, y] for y in range(1, 21)] \
            + [[x, 20] for x in range(79, -1, -1)]
        # Sparse samples along dense path
        indices = [0, 20, 40, 60, 80, 90, 100, 110, 120]
        wps = []
        for i, di in enumerate(indices):
            di = min(di, len(dense) - 1)
            x, y = dense[di]
            wps.append(_wp(i + 1, x, y, s_m=float(di), dense_index=di))
        # Force a non-consecutive near dup geographically: last near first
        wps[-1]["x_pixel"] = 0.2
        wps[-1]["y_pixel"] = 0.0
        wps[-1]["x_enu"], wps[-1]["y_enu"] = MetricCalibration.pixel_to_world(0.2, 0.0)
        result = validate_vehicle_waypoints(
            wps,
            dense_path_pixel=dense,
            geo_calibration=MetricCalibration(),
            config=WaypointValidationConfig(max_insert_iterations=8),
        )
        self.assertGreaterEqual(
            result.report.get("non_consecutive_near_duplicate_count", 0), 0
        )
        # near dup alone must not be the sole failure reason listing
        fr = result.report.get("failure_reasons") or []
        self.assertFalse(any("non_consecutive" in r for r in fr))

    def test_hard_fail_spacing(self):
        dense = [[0, 0], [50, 0]]
        wps = [
            _wp(1, 0, 0, s_m=0, dense_index=0),
            _wp(2, 50, 0, s_m=50, dense_index=1),
        ]
        result = validate_vehicle_waypoints(
            wps,
            dense_path_pixel=dense,
            geo_calibration=MetricCalibration(),
            config=WaypointValidationConfig(
                hard_fail_spacing_m=20.0,
                max_insert_iterations=0,
            ),
        )
        self.assertGreater(result.report["max_spacing_m"], 20.0)
        self.assertFalse(result.export_valid)
        self.assertIsNone(result.yaml_text)

    def test_max_spacing_auto_insert(self):
        dense = [[x, 0] for x in range(0, 51)]
        wps = [
            _wp(1, 0, 0, s_m=0, dense_index=0),
            _wp(2, 50, 0, s_m=50, dense_index=50),
        ]
        result = validate_vehicle_waypoints(
            wps,
            dense_path_pixel=dense,
            geo_calibration=MetricCalibration(),
            config=WaypointValidationConfig(max_insert_iterations=8),
        )
        self.assertLessEqual(result.report["max_spacing_m"], 12.0 + 1e-6)
        self.assertTrue(result.export_valid, result.report.get("failure_reasons"))
        self.assertGreater(result.report["waypoint_count"], 2)

    def test_curve_zone_gets_densified(self):
        dense = [[x, 0] for x in range(0, 41)] + [[40, y] for y in range(1, 41)]
        wps = [
            _wp(1, 0, 0, s_m=0, dense_index=0),
            _wp(2, 40, 0, s_m=40, dense_index=40),
            _wp(3, 40, 40, s_m=80, dense_index=80),
        ]
        result = validate_vehicle_waypoints(
            wps,
            dense_path_pixel=dense,
            geo_calibration=MetricCalibration(),
            config=WaypointValidationConfig(max_insert_iterations=6),
        )
        self.assertLessEqual(result.report["max_spacing_m"], 20.0)
        self.assertGreater(result.report["waypoint_count"], 3)

    def test_writes_artifacts_when_invalid(self):
        dense = [[0, 0], [50, 0]]
        wps = [
            _wp(1, 0, 0, s_m=0, dense_index=0),
            _wp(2, 50, 0, s_m=50, dense_index=1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_vehicle_waypoints(
                wps,
                dense_path_pixel=dense,
                geo_calibration=MetricCalibration(),
                config=WaypointValidationConfig(max_insert_iterations=0),
                output_dir=tmp,
            )
            import os
            self.assertTrue(os.path.isfile(os.path.join(tmp, "waypoint_validation_report.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "bad_segments.csv")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "waypoint_validation_overlay.png")))
            self.assertFalse(os.path.isfile(os.path.join(tmp, "subject1_waypoints.yaml")))
            self.assertFalse(result.export_valid)
            # CSV has new columns
            with open(os.path.join(tmp, "bad_segments.csv"), encoding="utf-8") as fh:
                header = fh.readline()
            self.assertIn("from_dense_index", header)
            self.assertIn("fix_attempted", header)


if __name__ == "__main__":
    unittest.main()
