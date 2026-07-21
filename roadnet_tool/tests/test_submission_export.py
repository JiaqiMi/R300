import csv
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.path_visualization import (  # noqa: E402
    ordered_task_markers,
    sample_direction_arrows,
    validate_task_sequence,
)
from roadnet.submission_export import (  # noqa: E402
    SUBMISSION_CORE_FILES,
    SUBMISSION_FILES,
    SubmissionExportError,
    extract_image_array_from_layer,
    export_competition_submission,
    image_input_to_bgr,
)


def _valid_vehicle_report(**overrides):
    report = {
        "export_valid": True,
        "geometry_valid": True,
        "bad_segment_count": 0,
        "aba_backtrack_count": 0,
        "duplicate_consecutive_count": 0,
        "line_of_sight_failed_count": 0,
        "max_spacing_m": 10.0,
        "average_spacing_m": 7.0,
        "yaml_format_valid": True,
    }
    report.update(overrides)
    return report


def _vehicle_wps():
    return [
        {
            "seq": 1, "name": "wp_001",
            "x_pixel": 30, "y_pixel": 120,
            "longitude": 117.0003, "latitude": 39.0012,
            "altitude_m": 21.741, "spacing_mode": "straight_10m",
            "segment_distance_m": 0.0, "local_angle_deg": 0.0,
        },
        {
            "seq": 2, "name": "wp_002",
            "x_pixel": 150, "y_pixel": 120,
            "longitude": 117.0015, "latitude": 39.0012,
            "altitude_m": 21.741, "spacing_mode": "straight_10m",
            "segment_distance_m": 10.0, "local_angle_deg": 0.0,
        },
        {
            "seq": 3, "name": "wp_003",
            "x_pixel": 280, "y_pixel": 120,
            "longitude": 117.0028, "latitude": 39.0012,
            "altitude_m": 21.741, "spacing_mode": "straight_10m",
            "segment_distance_m": 10.0, "local_angle_deg": 0.0,
        },
    ]


def _core_expected_files(*, with_subject1: bool):
    files = set(SUBMISSION_CORE_FILES) | {
        "track_waypoints_overlay.csv",
        "waypoint_validation_report.json",
        "bad_segments.csv",
        "vehicle_waypoints_adaptive.csv",
    }
    if with_subject1:
        files |= {"subject1_waypoints.yaml", "waypoints.yaml", "waypoint_validation_overlay.png"}
    else:
        files |= {"vehicle_waypoints_adaptive_INVALID.csv"}
        # overlay may still be written
        files.add("waypoint_validation_overlay.png")
    return files


class QImage:
    """Headless RGBA8888 QImage test double, including row padding."""

    def __init__(self, rgba, padding=8):
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        self._height, self._width = rgba.shape[:2]
        self._bytes_per_line = self._width * 4 + int(padding)
        rows = np.zeros((self._height, self._bytes_per_line), dtype=np.uint8)
        rows[:, :self._width * 4] = rgba.reshape(self._height, self._width * 4)
        self._storage = rows

    def width(self):
        return self._width

    def height(self):
        return self._height

    def bytesPerLine(self):
        return self._bytes_per_line

    def bits(self):
        return memoryview(self._storage)


class QPixmap:
    def __init__(self, image):
        self._image = image

    def toImage(self):
        return self._image


class FakeCalibration:
    is_valid = True
    pixel_resolution_estimated_m = 1.0

    @staticmethod
    def pixel_to_lonlat(x, y):
        return 117.0 + x * 0.00001, 39.0 + y * 0.00001


def _tasks():
    return [
        SimpleNamespace(seq=1, point_type=0, pixel_x=30.0, pixel_y=120.0),
        SimpleNamespace(seq=2, point_type=2, pixel_x=150.0, pixel_y=120.0),
        SimpleNamespace(seq=3, point_type=1, pixel_x=280.0, pixel_y=120.0),
    ]


def _snapped():
    return [
        SimpleNamespace(seq=1, point_type=0, snapped_x=30.0, snapped_y=120.0,
                        original_x=30.0, original_y=120.0, status="ok"),
        SimpleNamespace(seq=2, point_type=2, snapped_x=150.0, snapped_y=120.0,
                        original_x=150.0, original_y=120.0, status="warning"),
        SimpleNamespace(seq=3, point_type=1, snapped_x=280.0, snapped_y=120.0,
                        original_x=280.0, original_y=120.0, status="ok"),
    ]


class PathVisualizationTests(unittest.TestCase):
    def test_direction_arrows_follow_path_order(self):
        arrows = sample_direction_arrows([[0, 0], [250, 0]], spacing_px=80, size_px=12)
        self.assertEqual(len(arrows), 3)
        self.assertGreater(arrows[0]["direction"][0], 0)
        self.assertAlmostEqual(arrows[0]["direction"][1], 0)
        self.assertLess(arrows[0]["triangle"][1][0], arrows[0]["triangle"][0][0])

    def test_sequence_is_controlled_by_seq(self):
        tasks = list(reversed(_tasks()))
        self.assertEqual(validate_task_sequence(tasks), [])
        markers = ordered_task_markers(tasks, _snapped())
        self.assertEqual([marker["seq"] for marker in markers], [1, 2, 3])
        self.assertEqual([marker["role"] for marker in markers], ["start", "waypoint", "goal"])
        self.assertEqual(markers[1]["label"], "P1 · seq=2")

    def test_invalid_start_goal_sequence_is_rejected(self):
        tasks = _tasks()
        tasks[0].point_type = 2
        errors = validate_task_sequence(tasks)
        self.assertTrue(any("起点数量" in error for error in errors))


class SubmissionExportTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((240, 320, 3), 180, dtype=np.uint8)
        self.nodes = [
            {"id": 1, "x": 20, "y": 120},
            {"id": 2, "x": 300, "y": 120},
        ]
        self.edges = [{
            "id": 10, "start": 1, "end": 2, "enabled": True,
            "points_pixel": [[20, 120], [160, 120], [300, 120]],
        }]
        self.path = [[30, 120], [150, 120], [280, 120]]

    def _rgba_image(self):
        rgba = np.empty((240, 320, 4), dtype=np.uint8)
        rgba[..., 0] = np.arange(320, dtype=np.uint8)[None, :]
        rgba[..., 1] = np.arange(240, dtype=np.uint8)[:, None]
        rgba[..., 2] = 90
        rgba[..., 3] = 255
        return rgba

    def _assert_exported_overlay(self, output_dir):
        rendered = cv2.imread(os.path.join(output_dir, "competition_roadnet_overlay.png"))
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape, (240, 320, 3))
        self.assertGreater(float(rendered.std()), 1.0)

    def test_exports_clean_debug_and_submission_data(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_competition_submission(
                output_dir, self.image, self.nodes, self.edges,
                planned_path_pixel=self.path,
                vehicle_waypoints=_vehicle_wps(),
                waypoint_validation_report=_valid_vehicle_report(),
                sparse_waypoints=_vehicle_wps(),
                task_points=_tasks(), snapped_points=_snapped(),
                road_mask=np.ones((240, 320), dtype=np.uint8) * 255,
                skeleton=np.pad(np.ones((1, 280), dtype=np.uint8) * 255,
                                ((120, 119), (20, 20))),
                geo_calibration=FakeCalibration(), image_path="map.png",
                project_name="Competition Test", arrow_spacing_px=80,
            )
            names = set(os.listdir(output_dir))
            for required in _core_expected_files(with_subject1=True):
                self.assertIn(required, names, required)
            self.assertEqual(result["report"]["path_point_count"], 3)
            self.assertEqual(result["report"]["track_waypoint_count"], 3)
            self.assertTrue(result["report"]["has_planned_path"])
            self.assertTrue(result["report"]["geo_calibrated"])
            self.assertIn("WGS84", result["report"]["coordinate_system"])
            self.assertTrue(result["report"]["yaml_export_valid"])
            self.assertEqual(result["report"]["official_vehicle_yaml"], "subject1_waypoints.yaml")

            for filename in SUBMISSION_CORE_FILES[:3]:
                rendered = cv2.imread(os.path.join(output_dir, filename))
                self.assertIsNotNone(rendered)
                self.assertEqual(rendered.shape[:2], (240, 320))

            with open(os.path.join(output_dir, "global_path_geo.csv"), encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                rows[0][0:4],
                ["seq", "longitude_wgs84", "latitude_wgs84", "altitude_m"],
            )
            self.assertEqual(rows[1][6], "WGS84/EPSG:4326")

            with open(os.path.join(output_dir, "track_waypoints_overlay.csv"),
                      encoding="utf-8", newline="") as stream:
                track_rows = list(csv.reader(stream))
            self.assertEqual(len(track_rows), 4)
            self.assertEqual(track_rows[1][6], "WGS84/EPSG:4326")

            with open(os.path.join(output_dir, "subject1_waypoints.yaml"), encoding="utf-8") as stream:
                s1 = stream.read()
            self.assertTrue(s1.startswith("subject1_waypoints:"))
            self.assertIn("name: wp_001", s1)
            self.assertIn("latitude_deg:", s1)
            self.assertIn("longitude_deg:", s1)
            self.assertNotIn("yaw_deg:", s1)
            with open(os.path.join(output_dir, "waypoints.yaml"), encoding="utf-8") as stream:
                self.assertEqual(stream.read(), s1)

            with open(os.path.join(output_dir, "vehicle_waypoints_adaptive.csv"),
                      encoding="utf-8", newline="") as stream:
                adaptive_rows = list(csv.reader(stream))
            self.assertIn("segment_distance_m", adaptive_rows[0])
            self.assertIn("latitude_deg", adaptive_rows[0])

            with open(os.path.join(output_dir, "submission_report.json"), encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(report["node_count"], 2)
            self.assertEqual(report["edge_count"], 1)
            self.assertEqual(report["task_point_count"], 3)
            self.assertGreater(report["path_length_m"], 0)
            self.assertEqual(report["official_vehicle_yaml"], "subject1_waypoints.yaml")
            self.assertTrue(report["yaml_export_valid"])
            self.assertEqual(report["vehicle_waypoint_count"], 3)
            with open(os.path.join(output_dir, "final_graph.json"), encoding="utf-8") as stream:
                graph = json.load(stream)
            self.assertEqual(graph["coordinate_system"], "image_pixel")
            self.assertEqual(graph["geodetic_crs"], "WGS84")
            self.assertIn("x_pixel", graph["nodes"][0])

    def test_rejects_old_geo_path_yaml_without_validation(self):
        """Without validation report, must NOT invent old waypoints.yaml from geo path."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_competition_submission(
                output_dir, self.image, self.nodes, self.edges,
                planned_path_pixel=self.path,
                sparse_waypoints=[
                    {"seq": 1, "x_pixel": 30, "y_pixel": 120,
                     "longitude": 117.0003, "latitude": 39.0012},
                ],
                geo_calibration=FakeCalibration(),
            )
            self.assertFalse(result["report"]["yaml_export_valid"])
            self.assertIsNone(result["report"]["official_vehicle_yaml"])
            self.assertFalse(
                os.path.isfile(os.path.join(output_dir, "subject1_waypoints.yaml"))
            )
            # No old-format waypoints.yaml either
            self.assertFalse(os.path.isfile(os.path.join(output_dir, "waypoints.yaml")))
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "vehicle_waypoints_adaptive_INVALID.csv"))
                or os.path.isfile(os.path.join(output_dir, "vehicle_waypoints_adaptive.csv"))
            )

    def test_allows_graph_only_export_without_calibration(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_competition_submission(
                output_dir, self.image, self.nodes, self.edges,
                planned_path_pixel=[], geo_calibration=None,
            )
            self.assertFalse(result["report"]["has_planned_path"])
            self.assertFalse(result["report"]["geo_calibrated"])
            self.assertFalse(result["report"]["yaml_export_valid"])
            names = set(os.listdir(output_dir))
            for required in SUBMISSION_CORE_FILES:
                self.assertIn(required, names, required)
            self.assertIn("track_waypoints_overlay.csv", names)
            self.assertNotIn("subject1_waypoints.yaml", names)

    def test_uint8_ndarray_input_exports_non_blank_native_size(self):
        image = self._rgba_image()[..., :3].copy()
        with tempfile.TemporaryDirectory() as output_dir:
            export_competition_submission(output_dir, image, self.nodes, self.edges)
            self._assert_exported_overlay(output_dir)

    def test_qimage_input_exports_non_blank_native_size(self):
        with tempfile.TemporaryDirectory() as output_dir:
            export_competition_submission(
                output_dir, QImage(self._rgba_image()), self.nodes, self.edges,
            )
            self._assert_exported_overlay(output_dir)

    def test_qpixmap_input_exports_non_blank_native_size(self):
        with tempfile.TemporaryDirectory() as output_dir:
            export_competition_submission(
                output_dir, QPixmap(QImage(self._rgba_image())), self.nodes, self.edges,
            )
            self._assert_exported_overlay(output_dir)

    def test_object_array_is_rejected_with_source_diagnostics(self):
        bad_mask = np.empty((2,), dtype=object)
        bad_mask[0] = np.zeros((10, 10), dtype=np.uint8)
        bad_mask[1] = np.zeros((20, 20), dtype=np.uint8)
        with self.assertRaises(SubmissionExportError) as caught:
            image_input_to_bgr(bad_mask, "road_mask")
        message = str(caught.exception)
        self.assertIn("dtype=object", message)
        self.assertIn("source=road_mask", message)
        self.assertIn("shape=(2,)", message)
        self.assertIn("type=numpy.ndarray", message)

    def test_skeleton_state_dict_prefers_optimized_image(self):
        raw = np.zeros((240, 320), dtype=np.uint8)
        optimized = raw.copy()
        optimized[120, 20:300] = 255
        payload = {
            "raw_skeleton": raw,
            "optimized_skeleton": optimized,
            "state": "optimized",
            "stats": {"pixel_count": 280},
        }
        extracted = extract_image_array_from_layer(payload, "skeleton")
        self.assertIs(extracted, optimized)
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_competition_submission(
                output_dir, self.image, self.nodes, self.edges,
                skeleton=payload,
            )
            self.assertNotIn("skeleton", result["report"]["skipped_layers"])
            self.assertEqual(result["report"]["warnings"], [])
            debug = cv2.imread(os.path.join(output_dir, "competition_roadnet_overlay_debug.png"))
            self.assertIsNotNone(debug)
            self.assertEqual(debug.shape[:2], (240, 320))

    def test_unusable_skeleton_dict_is_skipped_and_reported(self):
        payload = {
            "state": "optimized",
            "stats": {"pixel_count": 280},
        }
        with tempfile.TemporaryDirectory() as output_dir:
            result = export_competition_submission(
                output_dir, self.image, self.nodes, self.edges,
                skeleton=payload,
            )
            expected = set(SUBMISSION_CORE_FILES) | {
                "track_waypoints_overlay.csv",
                "waypoint_validation_report.json",
                "bad_segments.csv",
                "vehicle_waypoints_adaptive_INVALID.csv",
            }
            names = set(os.listdir(output_dir))
            for required in expected:
                self.assertIn(required, names, required)
            self.assertIn("skeleton", result["report"]["skipped_layers"])
            self.assertTrue(any(
                "skeleton layer is dict and no image array could be extracted" in warning
                for warning in result["report"]["warnings"]
            ))
            for filename in (
                    "competition_roadnet_overlay.png",
                    "competition_roadnet_overlay_debug.png",
                    "final_graph.json",
                    "global_path_geo.csv",
                    "track_waypoints_overlay.csv"):
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)), filename)
            self.assertFalse(
                os.path.isfile(os.path.join(output_dir, "subject1_waypoints.yaml"))
            )
            with open(os.path.join(output_dir, "submission_report.json"), encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertIn("skeleton", report["skipped_layers"])
            self.assertEqual(report["warnings"], result["report"]["warnings"])
            self.assertFalse(report.get("yaml_export_valid"))
            self.assertIsNone(report.get("official_vehicle_yaml"))

    def test_list_dict_and_none_inputs_are_rejected_before_opencv(self):
        for invalid in (None, [], {}):
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as output_dir:
                with self.assertRaises(SubmissionExportError) as caught:
                    export_competition_submission(
                        output_dir, invalid, self.nodes, self.edges,
                    )
                self.assertIn("source=image_rgb", str(caught.exception))

    def test_samroad_dialog_has_scroll_and_persistent_geometry(self):
        path = os.path.join(ROOT, "gui", "samroad_single_run_dialog.py")
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn("self.setMinimumSize(760, 560)", source)
        self.assertIn("self.resize(900, 720)", source)
        self.assertIn("self._settings_scroll = QScrollArea()", source)
        self.assertIn('QSettings("RoadNetStudio", "SAMRoadSingleRunDialog")', source)
        self.assertIn("available.height() * 0.90", source)
        self.assertIn("main_layout.addLayout(btn_layout)", source)
        self.assertLess(
            source.index("main_layout.addWidget(self._progress_bar)"),
            source.index("main_layout.addLayout(btn_layout)"),
        )

        parameter_path = os.path.join(ROOT, "gui", "parameter_panel.py")
        with open(parameter_path, encoding="utf-8") as stream:
            parameter_source = stream.read()
        self.assertIn("导出比赛路网图", parameter_source)
        self.assertIn('apply_requested.emit("export_competition")', parameter_source)


if __name__ == "__main__":
    unittest.main()
