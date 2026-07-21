import csv
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.path_export import (  # noqa: E402
    EXPORT_FILENAMES,
    InvalidGeoCalibrationError,
    PathExportError,
    convert_pixel_path_to_geo,
    export_planned_path,
    resample_pixel_path,
)


class FakeCalibration:
    is_valid = True
    pixel_resolution_estimated_m = 1.0

    @staticmethod
    def pixel_to_lonlat(x, y):
        return 117.0 + float(x) * 0.00001, 39.0 + float(y) * 0.00001


class InvalidCalibration:
    is_valid = False


class MetricCalibration:
    is_valid = True
    pixel_resolution_estimated_m = 1.0

    @staticmethod
    def pixel_to_world(x, y):
        return float(x), -float(y)

    @staticmethod
    def pixel_to_wgs84(x, y):
        return 117.0 + float(x) * 0.00001, 39.0 - float(y) * 0.00001


class PathExportTests(unittest.TestCase):
    def _planning_result(self):
        segment = SimpleNamespace(
            from_seq=1,
            to_seq=2,
            status="ok",
            length_px=10.0,
            node_path=[1, 2],
            edge_path=[10],
            error="",
            unexpected_task_virtual_nodes=[],
        )
        return SimpleNamespace(success=True, segments=[segment], task_sequence=[1, 2])

    def _mini_graph(self):
        return {
            "nodes": [
                {"id": 1, "x": 0.0, "y": 0.0},
                {"id": 2, "x": 10.0, "y": 0.0},
            ],
            "edges": [
                {
                    "id": 10,
                    "start": 1,
                    "end": 2,
                    "enabled": True,
                    "points_pixel": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
                    "polyline": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
                }
            ],
        }

    def test_exports_complete_vehicle_path_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "outputs", "path_planning", "run_test")
            result = export_planned_path(
                [[0, 0], [5, 0], [10, 0]],
                output_dir,
                FakeCalibration(),
                planning_result=self._planning_result(),
                task_point_count=2,
                planned_path_edges=[10],
                resample_spacing_m=2.0,
                final_graph=self._mini_graph(),
                image_width=100,
                image_height=100,
            )

            names = sorted(os.listdir(output_dir))
            for required in (
                "subject1_waypoints.yaml",
                "waypoints_sparse_10m.yaml",
                "planned_segments_debug.csv",
                "dense_path_debug.csv",
                "dense_path_validation_report.json",
                "virtual_node_split_debug.csv",
            ):
                self.assertIn(required, names)
            with open(os.path.join(output_dir, "global_path_pixel.json"), encoding="utf-8") as stream:
                pixel_doc = json.load(stream)
            self.assertEqual(pixel_doc["coordinate_system"], "image_pixel")
            self.assertGreater(pixel_doc["point_count"], 2)
            self.assertEqual(set(pixel_doc["path"][0]), {"seq", "x", "y"})
            self.assertEqual(pixel_doc["path"][0]["seq"], 1)
            self.assertEqual(pixel_doc["path"][-1]["x"], 10.0)

            with open(os.path.join(output_dir, "global_path_geo.json"), encoding="utf-8") as stream:
                geo_doc = json.load(stream)
            self.assertEqual(geo_doc["coordinate_system"], "EPSG:4326")
            self.assertEqual(geo_doc["point_count"], pixel_doc["point_count"])
            self.assertEqual(
                set(geo_doc["path"][0]),
                {"seq", "longitude", "latitude", "altitude"},
            )

            with open(os.path.join(output_dir, "global_path.csv"), newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            # Expanded CSV now includes ENU, yaw, spacing, tag, speed, and validation fields.
            basic_cols = ["seq", "longitude", "latitude", "altitude", "x_pixel", "y_pixel"]
            for col in basic_cols:
                self.assertIn(col, rows[0])
            self.assertEqual(len(rows) - 1, pixel_doc["point_count"])

            with open(os.path.join(output_dir, "waypoints.yaml"), encoding="utf-8") as stream:
                yaml_text = stream.read()
            # waypoints.yaml is now an identical copy of subject1_waypoints.yaml
            self.assertTrue(yaml_text.startswith("subject1_waypoints:"))
            self.assertIn("  waypoints:\n", yaml_text)
            self.assertIn("name: wp_001", yaml_text)
            self.assertIn("latitude_deg:", yaml_text)
            self.assertIn("longitude_deg:", yaml_text)
            self.assertNotIn("yaw_deg:", yaml_text)
            self.assertNotIn("pass_through:", yaml_text)

            # ── Verify subject1_waypoints.yaml is the CLEAN vehicle file ──
            with open(os.path.join(output_dir, "subject1_waypoints.yaml"), encoding="utf-8") as stream:
                s1_text = stream.read()
            self.assertEqual(yaml_text, s1_text)
            # Root node
            self.assertTrue(s1_text.startswith("subject1_waypoints:"))
            self.assertIn("  waypoints:\n", s1_text)
            # Clean format: only name, latitude_deg, longitude_deg, altitude_m
            self.assertIn("name: wp_001", s1_text)
            self.assertNotIn("name: wp_01\n", s1_text)  # name_digits=3 → wp_001
            self.assertIn("latitude_deg:", s1_text)
            self.assertIn("longitude_deg:", s1_text)
            self.assertIn("altitude_m:", s1_text)
            # NO extra fields
            self.assertNotIn("yaw_deg:", s1_text)
            self.assertNotIn("pass_through:", s1_text)
            self.assertNotIn("target_speed_mps:", s1_text)
            self.assertNotIn("arrival_radius_m:", s1_text)
            self.assertNotIn("tag:", s1_text)
            self.assertNotIn("seq:", s1_text)  # no seq in vehicle file

            self.assertTrue(os.path.isfile(os.path.join(output_dir, "global_path_dense_pixel.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "global_path_dense_geo.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "waypoints_sparse_10m.yaml")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "waypoints_sparse_10m.csv")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "waypoint_resample_report.json")))
            self.assertGreater(os.path.getsize(os.path.join(output_dir, "waypoint_preview.png")), 0)

            with open(os.path.join(output_dir, "planning_report.json"), encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertTrue(report["success"])
            self.assertEqual(report["task_point_count"], 2)
            self.assertEqual(report["sparse_waypoint_count"], pixel_doc["point_count"])
            self.assertGreater(report["path_length_m"], 0)
            self.assertEqual(report["segments"][0]["from_seq"], 1)
            self.assertEqual(report["segments"][0]["to_seq"], 2)
            self.assertIn("length_m", report["segments"][0])
            self.assertIn("export_valid", report)
            self.assertIn("recommended_vehicle_file", report)

    def test_rejects_empty_path_without_creating_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "run")
            with self.assertRaisesRegex(PathExportError, "少于 2"):
                export_planned_path([[1, 2]], output_dir, FakeCalibration())
            self.assertFalse(os.path.exists(output_dir))

    def test_rejects_invalid_calibration_without_creating_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "run")
            with self.assertRaises(InvalidGeoCalibrationError):
                export_planned_path([[0, 0], [10, 0]], output_dir, InvalidCalibration())
            self.assertFalse(os.path.exists(output_dir))

    def test_supports_pixel_spacing_resampling(self):
        sampled = resample_pixel_path([[0, 0], [10, 0]], 5)
        self.assertEqual(sampled, [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])

    def test_supports_pixel_to_wgs84_converter_name(self):
        calibration = SimpleNamespace(
            is_valid=True,
            pixel_to_wgs84=lambda x, y: (120.0 + x, 30.0 + y),
        )
        converted = convert_pixel_path_to_geo([[0, 0], [1, 1]], calibration)
        self.assertEqual(converted[1], [121.0, 31.0, 0.0])

    def test_default_vehicle_export_is_sparse_ten_metre_not_dense_path(self):
        dense = [[x, 0] for x in range(101)]
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_planned_path(dense, output_dir, MetricCalibration())
            # dense_path_pixel is export-time densified centerline (≥ source vertices)
            self.assertGreaterEqual(len(result["dense_path_pixel"]), 101)
            self.assertLessEqual(len(result["sparse_waypoints_pixel"]), 16)
            self.assertEqual(result["waypoint_resample_report"]["parameters"]["straight_spacing_m"], 10.0)
            self.assertTrue(result.get("export_valid"), result.get("waypoint_validation_report"))
            with open(os.path.join(output_dir, "waypoints_sparse_10m.yaml"), encoding="utf-8") as stream:
                yaml_text = stream.read()
            self.assertTrue(yaml_text.startswith("subject1_waypoints:"))
            self.assertEqual(yaml_text.count("    - name:"), len(result["waypoints"]))
            self.assertNotIn("coordinate_system:", yaml_text)
            self.assertNotIn("  - seq:", yaml_text)
            with open(os.path.join(output_dir, "waypoints.yaml"), encoding="utf-8") as stream:
                self.assertEqual(stream.read(), yaml_text)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "waypoint_validation_report.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "subject1_waypoints.yaml")))

    def test_gui_path_export_bindings_do_not_use_mask_export(self):
        main_window_path = os.path.join(ROOT, "gui", "main_window.py")
        with open(main_window_path, encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn('"export":          self._on_export_planned_path', source)
        self.assertNotIn('"export":          self._on_export_mask', source)
        self.assertIn("act_export_path.triggered.connect(self._on_export_planned_path)", source)
        self.assertIn('if tool_id in ("plan", "export"):', source)
        self.assertIn("self._on_export_planned_path()", source)

    def test_coarse_dense_step_does_not_block_vehicle_export(self):
        """dense_path step_distance_too_large is a warning; vehicle pass → YAML ok."""
        # Sparse polyline vertices (~50m steps at 1m/px) → raw dense warning
        dense = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]
        graph = {
            "nodes": [
                {"id": 1, "x": 0.0, "y": 0.0},
                {"id": 2, "x": 100.0, "y": 0.0},
            ],
            "edges": [
                {
                    "id": 10,
                    "start": 1,
                    "end": 2,
                    "enabled": True,
                    "points_pixel": dense,
                    "polyline": dense,
                }
            ],
        }
        segment = SimpleNamespace(
            from_seq=1, to_seq=2, status="ok", length_px=100.0,
            node_path=[1, 2], edge_path=[10], error="",
            unexpected_task_virtual_nodes=[],
        )
        planning = SimpleNamespace(
            success=True, segments=[segment], task_sequence=[1, 2],
        )
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_planned_path(
                dense, output_dir, MetricCalibration(),
                planning_result=planning,
                task_point_count=2,
                planned_path_edges=[10],
                final_graph=graph,
                image_width=200,
                image_height=50,
            )
            self.assertTrue(
                result.get("vehicle_waypoints_valid"),
                result.get("waypoint_validation_report"),
            )
            self.assertTrue(result.get("export_valid"), result.get("planning_report"))
            report = result.get("planning_report") or {}
            self.assertIn("dense_path_raw_valid", report)
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "subject1_waypoints.yaml"))
            )
            val = result.get("waypoint_validation_report") or {}
            if report.get("dense_path_raw_valid") is False:
                self.assertTrue(
                    val.get("dense_path_warning_resolved_by_resampling")
                    or report.get("dense_path_warning_resolved_by_resampling")
                )


if __name__ == "__main__":
    unittest.main()
