"""Tests for the unified vehicle_waypoint_pipeline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.task_points import TaskPoint  # noqa: E402
from roadnet.vehicle_waypoint_pipeline import (  # noqa: E402
    PipelineConfig,
    build_vehicle_waypoint_summary,
    classify_dense_path_zones,
    cleanup_anchor_aware_duplicates,
    expand_route_edges_to_dense_path,
    export_subject1_yaml_from_vehicle_csv,
    interpolate_dense_path_at_s,
    repair_vehicle_waypoints_using_dense_path,
    run_vehicle_waypoint_pipeline,
    sample_vehicle_waypoints_from_dense_path,
    validate_vehicle_waypoints_csv,
)


class MetricCalibration:
    is_valid = True
    pixel_resolution_estimated_m = 1.0
    image_width = 500
    image_height = 200

    @staticmethod
    def pixel_to_world(x, y):
        return float(x), -float(y)

    @staticmethod
    def world_to_pixel(x, y):
        return float(x), -float(y)

    @staticmethod
    def pixel_to_wgs84(x, y):
        return 117.0 + float(x) * 0.00001, 39.0 - float(y) * 0.00001

    @staticmethod
    def wgs84_to_pixel(lon, lat):
        x = (float(lon) - 117.0) / 0.00001
        y = (39.0 - float(lat)) / 0.00001
        return x, y


def _straight_graph():
    # Long straight corridor with a T-junction mid-way for junction zoning.
    nodes = [
        {"id": 1, "x": 0.0, "y": 0.0},
        {"id": 2, "x": 50.0, "y": 0.0},
        {"id": 3, "x": 100.0, "y": 0.0},
        {"id": 4, "x": 150.0, "y": 0.0},
        {"id": 5, "x": 50.0, "y": 40.0},  # branch for degree>=3 at node 2
    ]
    edges = [
        {
            "id": 10, "start": 1, "end": 2, "enabled": True,
            "points_pixel": [[0.0, 0.0], [25.0, 0.0], [50.0, 0.0]],
            "polyline": [[0.0, 0.0], [25.0, 0.0], [50.0, 0.0]],
        },
        {
            "id": 11, "start": 2, "end": 3, "enabled": True,
            "points_pixel": [[50.0, 0.0], [75.0, 0.0], [100.0, 0.0]],
            "polyline": [[50.0, 0.0], [75.0, 0.0], [100.0, 0.0]],
        },
        {
            "id": 12, "start": 3, "end": 4, "enabled": True,
            "points_pixel": [[100.0, 0.0], [125.0, 0.0], [150.0, 0.0]],
            "polyline": [[100.0, 0.0], [125.0, 0.0], [150.0, 0.0]],
        },
        {
            "id": 13, "start": 2, "end": 5, "enabled": True,
            "points_pixel": [[50.0, 0.0], [50.0, 20.0], [50.0, 40.0]],
            "polyline": [[50.0, 0.0], [50.0, 20.0], [50.0, 40.0]],
        },
    ]
    return {"nodes": nodes, "edges": edges}


def _task_points():
    return [
        TaskPoint(seq=1, longitude=117.0, latitude=39.0, point_type=0,
                  pixel_x=0.0, pixel_y=0.0, source="file_import"),
        TaskPoint(seq=2, longitude=117.0015, latitude=39.0, point_type=1,
                  pixel_x=150.0, pixel_y=0.0, source="file_import"),
    ]


class VehicleWaypointPipelineTests(unittest.TestCase):
    def test_full_pipeline_artifacts(self):
        graph = _straight_graph()
        tasks = _task_points()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_vehicle_waypoint_pipeline(
                graph, tasks, MetricCalibration(), tmp,
                config=PipelineConfig(straight_spacing_m=15.0, curve_spacing_m=2.0),
            )
            self.assertIsNone(result.error, msg=result.error)
            self.assertTrue(result.status.dense_path_generated)
            self.assertTrue(result.status.yaml_exported)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "snapped_task_points.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "route_segments.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "dense_path.csv")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "dense_path_labeled.csv")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "vehicle_waypoints.csv")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "vehicle_waypoints_repaired.csv")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "waypoint_validation_report.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "subject1_waypoints.yaml")))

            dense = result.dense_path
            self.assertGreaterEqual(len(dense), 2)
            self.assertEqual(dense[0]["dense_index"], 0)
            self.assertIn("s_m", dense[0])
            self.assertIn("segment_index", dense[0])
            self.assertIn("edge_id", dense[0])
            for i in range(1, len(dense)):
                self.assertEqual(dense[i]["dense_index"], i)
                self.assertGreaterEqual(dense[i]["s_m"], dense[i - 1]["s_m"] - 1e-9)

            yaml_text = open(os.path.join(tmp, "subject1_waypoints.yaml"), encoding="utf-8").read()
            self.assertIn("subject1_waypoints:", yaml_text)
            self.assertIn("waypoints:", yaml_text)
            self.assertIn("wp_001", yaml_text)
            self.assertIn("latitude_deg:", yaml_text)
            self.assertIn("longitude_deg:", yaml_text)
            self.assertNotIn("coordinate_system:", yaml_text)

            report = json.loads(
                open(os.path.join(tmp, "waypoint_validation_report.json"), encoding="utf-8").read()
            )
            self.assertTrue(report.get("export_ready"))
            self.assertEqual(report.get("duplicate_consecutive_count"), 0)
            self.assertEqual(report.get("aba_backtrack_count"), 0)

    def test_sample_spacing_modes(self):
        dense = []
        for i in range(0, 61):
            dense.append({
                "dense_index": i,
                "segment_index": 0,
                "task_from_seq": 1,
                "task_to_seq": 2,
                "edge_id": 10,
                "edge_point_index": i,
                "x_pixel": float(i),
                "y_pixel": 0.0,
                "latitude_deg": 39.0,
                "longitude_deg": 117.0 + i * 0.00001,
                "altitude_m": 21.7,
                "s_m": float(i),
                "step_distance_m": 1.0 if i else 0.0,
                "source": "edge_polyline",
                "spacing_mode": "straight",
                "local_heading_deg": 0.0,
                "local_turn_angle_deg": 0.0,
                "nearest_junction_node_id": None,
                "distance_to_junction_m": None,
            })
        # mark a curve zone
        for i in range(20, 26):
            dense[i]["spacing_mode"] = "curve"

        wps = sample_vehicle_waypoints_from_dense_path(dense, config=PipelineConfig())
        self.assertGreaterEqual(len(wps), 2)
        self.assertEqual(wps[0]["name"], "wp_001")
        self.assertEqual(wps[0]["dense_index"], 0)
        self.assertEqual(wps[-1]["dense_index"], 60)
        # straight ~15m + curve densify → more than pick-only, but not 1m grid
        self.assertGreater(len(wps), 5)
        self.assertLess(len(wps), 50)
        summary = build_vehicle_waypoint_summary(wps, dense)
        self.assertTrue(summary.get("pass_straight_max_le_16_5"))
        self.assertTrue(summary.get("pass_curve_max_le_3"))

    def test_interpolate_and_sample_sparse_dense_path(self):
        """Even with ~50 sparse dense vertices over ~1737m, sample along s_m."""
        total_s = 1736.9
        n_dense = 56
        dense = []
        for i in range(n_dense):
            s = total_s * (i / (n_dense - 1))
            dense.append({
                "dense_index": i,
                "segment_index": 0,
                "edge_id": 1,
                "x_pixel": s,
                "y_pixel": 0.0,
                "latitude_deg": 39.0,
                "longitude_deg": 117.0 + s * 1e-5,
                "altitude_m": 21.7,
                "s_m": s,
                "spacing_mode": "straight",
            })
        # Mid-path curve bracket (two adjacent sparse vertices)
        dense[20]["spacing_mode"] = "curve"
        dense[21]["spacing_mode"] = "curve"

        mid = interpolate_dense_path_at_s(dense, 100.0)
        self.assertAlmostEqual(mid["s_m"], 100.0, places=3)
        self.assertAlmostEqual(mid["x_pixel"], 100.0, places=3)
        self.assertTrue(0.0 < float(mid["dense_index"]) < float(n_dense - 1))

        wps = sample_vehicle_waypoints_from_dense_path(dense, config=PipelineConfig())
        # All-straight lower bound ≈ 1736.9/15 + 1 ≈ 117; with curve zone → more
        self.assertGreater(len(wps), 100)
        self.assertGreater(len(wps), n_dense)  # must invent points, not only pick dense
        summary = build_vehicle_waypoint_summary(wps, dense)
        self.assertLessEqual(summary["straight_max_spacing_m"] or 0, 16.5)
        self.assertTrue(summary.get("pass_straight_max_le_16_5"))
        if summary.get("curve_max_spacing_m") is not None:
            self.assertLessEqual(summary["curve_max_spacing_m"], 3.0)

    def test_cleanup_anchor_aware_duplicates(self):
        wps = [
            {
                "seq": 1, "name": "wp_001", "s_m": 0.0, "dense_index": 0,
                "x_pixel": 0.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0, "altitude_m": 21.7,
                "spacing_mode": "straight", "keep": True, "source_mode": "endpoint",
            },
            {
                "seq": 2, "name": "wp_002", "s_m": 0.1, "dense_index": 0.1,
                "x_pixel": 0.1, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.000001, "altitude_m": 21.7,
                "spacing_mode": "straight", "keep": False, "source_mode": "straight_sample",
            },
            {
                "seq": 3, "name": "wp_003", "s_m": 15.0, "dense_index": 15,
                "x_pixel": 15.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.00015, "altitude_m": 21.7,
                "spacing_mode": "straight", "keep": False, "source_mode": "straight_sample",
            },
            {
                "seq": 4, "name": "wp_004", "s_m": 15.05, "dense_index": 15.05,
                "x_pixel": 15.05, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0001505, "altitude_m": 21.7,
                "spacing_mode": "straight", "keep": False, "source_mode": "straight_sample",
            },
            {
                "seq": 5, "name": "wp_005", "s_m": 30.0, "dense_index": 30,
                "x_pixel": 30.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0003, "altitude_m": 21.7,
                "spacing_mode": "junction", "keep": True, "source_mode": "junction_sample",
            },
            {
                "seq": 6, "name": "wp_006", "s_m": 30.1, "dense_index": 30.1,
                "x_pixel": 30.1, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.000301, "altitude_m": 21.7,
                "spacing_mode": "task", "keep": True, "source_mode": "task_anchor",
            },
            {
                "seq": 7, "name": "wp_007", "s_m": 45.0, "dense_index": 45,
                "x_pixel": 45.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.00045, "altitude_m": 21.7,
                "spacing_mode": "straight", "keep": True, "source_mode": "endpoint",
            },
        ]
        cleaned, warns = cleanup_anchor_aware_duplicates(wps, PipelineConfig())
        # Dropped: near-endpoint ordinary (0.1), near-ordinary latter (15.05)
        # Kept both junction+task keep-keep with warning
        sources = [w["source_mode"] for w in cleaned]
        self.assertIn("endpoint", sources)
        self.assertIn("task_anchor", sources)
        self.assertIn("junction_sample", sources)
        self.assertNotIn(0.1, [round(float(w["s_m"]), 2) for w in cleaned])
        self.assertEqual(len(warns), 1)
        self.assertEqual(cleaned[0]["name"], "wp_001")
        self.assertEqual(cleaned[-1]["name"], f"wp_{len(cleaned):03d}")
        # distances recomputed
        self.assertEqual(cleaned[0]["distance_from_prev_m"], 0.0)
        for i in range(1, len(cleaned)):
            self.assertAlmostEqual(
                cleaned[i]["distance_from_prev_m"],
                abs(cleaned[i]["s_m"] - cleaned[i - 1]["s_m"]),
                places=5,
            )
        report, _ = validate_vehicle_waypoints_csv(
            cleaned, config=PipelineConfig(),
        )
        self.assertEqual(report["duplicate_consecutive_count"], 0)

    def test_repair_spacing(self):
        dense = []
        for i in range(0, 31):
            dense.append({
                "dense_index": i, "s_m": float(i),
                "x_pixel": float(i), "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0 + i * 1e-5,
                "altitude_m": 21.7, "spacing_mode": "straight",
                "segment_index": 0, "edge_id": 1,
            })
        sparse = [
            {
                "seq": 1, "name": "wp_001", "dense_index": 0, "s_m": 0.0,
                "x_pixel": 0.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": True,
                "segment_index": 0, "edge_id": 1, "source_mode": "straight_sample",
            },
            {
                "seq": 2, "name": "wp_002", "dense_index": 30, "s_m": 30.0,
                "x_pixel": 30.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0003,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": True,
                "segment_index": 0, "edge_id": 1, "source_mode": "straight_sample",
            },
        ]
        repaired, _ = repair_vehicle_waypoints_using_dense_path(
            sparse, dense, config=PipelineConfig(max_straight_spacing_m=16.5),
        )
        report, _ = validate_vehicle_waypoints_csv(
            repaired, dense_path_labeled=dense, config=PipelineConfig(),
        )
        self.assertEqual(report["spacing_violation_count"], 0)
        self.assertTrue(report["export_ready"])
        self.assertGreater(len(repaired), 2)

    def test_straight_allowed_is_16_5_not_3(self):
        """straight gaps of 4–15m must NOT be spacing_violation."""
        dense = []
        for i in range(0, 61):
            dense.append({
                "dense_index": i, "s_m": float(i),
                "x_pixel": float(i), "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0 + i * 1e-5,
                "altitude_m": 21.7, "spacing_mode": "straight",
                "segment_index": 0, "edge_id": 1,
            })
        wps = [
            {
                "seq": 1, "name": "wp_001", "dense_index": 0, "s_m": 0.0,
                "x_pixel": 0.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": True,
            },
            {
                "seq": 2, "name": "wp_002", "dense_index": 4, "s_m": 4.4,
                "x_pixel": 4.4, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.000044,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": False,
            },
            {
                "seq": 3, "name": "wp_003", "dense_index": 12, "s_m": 12.0,
                "x_pixel": 12.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.00012,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": False,
            },
            {
                "seq": 4, "name": "wp_004", "dense_index": 27, "s_m": 27.0,
                "x_pixel": 27.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.00027,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": False,
            },
            {
                "seq": 5, "name": "wp_005", "dense_index": 60, "s_m": 60.0,
                "x_pixel": 60.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0006,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": True,
            },
        ]
        # Fix distances to match s_m gaps: 4.4, 7.6, 15.0, 33.0
        wps[1]["s_m"] = 4.4
        wps[2]["s_m"] = 4.4 + 7.6  # 12.0
        wps[3]["s_m"] = 12.0 + 15.0  # 27.0
        wps[4]["s_m"] = 60.0

        with tempfile.TemporaryDirectory() as tmp:
            report, bad = validate_vehicle_waypoints_csv(
                wps, dense_path_labeled=dense, config=PipelineConfig(),
                output_dir=tmp,
            )
        # 4.4, 7.6, 15.0 all OK under 16.5; only 33.0 is violation
        self.assertEqual(report["straight_spacing_violation_count"], 1)
        self.assertEqual(report["spacing_violation_count"], 1)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["pair_spacing_mode"], "straight")
        self.assertEqual(float(bad[0]["allowed_m"]), 16.5)
        self.assertGreater(float(bad[0]["distance_m"]), 16.5)

    def test_curve_allowed_is_3(self):
        dense = []
        for i in range(0, 21):
            dense.append({
                "dense_index": i, "s_m": float(i),
                "x_pixel": float(i), "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0 + i * 1e-5,
                "altitude_m": 21.7,
                "spacing_mode": "curve" if 5 <= i <= 15 else "straight",
                "segment_index": 0, "edge_id": 1,
            })
        wps = [
            {
                "seq": 1, "name": "wp_001", "dense_index": 5, "s_m": 5.0,
                "x_pixel": 5.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.00005,
                "altitude_m": 21.7, "spacing_mode": "curve", "keep": True,
            },
            {
                "seq": 2, "name": "wp_002", "dense_index": 12, "s_m": 12.0,
                "x_pixel": 12.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.00012,
                "altitude_m": 21.7, "spacing_mode": "curve", "keep": False,
            },
        ]
        report, bad = validate_vehicle_waypoints_csv(
            wps, dense_path_labeled=dense, config=PipelineConfig(),
        )
        self.assertEqual(report["spacing_violation_count"], 1)
        self.assertEqual(bad[0]["pair_spacing_mode"], "curve")
        self.assertEqual(float(bad[0]["allowed_m"]), 3.0)

    def test_repair_does_not_insert_on_short_straight(self):
        dense = []
        for i in range(0, 21):
            dense.append({
                "dense_index": i, "s_m": float(i),
                "x_pixel": float(i), "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0 + i * 1e-5,
                "altitude_m": 21.7, "spacing_mode": "straight",
                "segment_index": 0, "edge_id": 1,
            })
        sparse = [
            {
                "seq": 1, "name": "wp_001", "dense_index": 0, "s_m": 0.0,
                "x_pixel": 0.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": True,
                "source_mode": "straight_sample",
            },
            {
                "seq": 2, "name": "wp_002", "dense_index": 10, "s_m": 10.0,
                "x_pixel": 10.0, "y_pixel": 0.0,
                "latitude_deg": 39.0, "longitude_deg": 117.0001,
                "altitude_m": 21.7, "spacing_mode": "straight", "keep": True,
                "source_mode": "straight_sample",
            },
        ]
        repaired, report = repair_vehicle_waypoints_using_dense_path(
            sparse, dense, config=PipelineConfig(),
        )
        # 10m < 16.5 → no inserts
        self.assertEqual(len(repaired), 2)
        self.assertTrue(report.get("repair_success"))
        insert_actions = [a for a in report["actions"] if a.startswith("insert")]
        self.assertEqual(insert_actions, [])
        rows = [
            {
                "seq": 1, "name": "wp_001",
                "latitude_deg": 39.1, "longitude_deg": 117.2, "altitude_m": 21.7,
            },
            {
                "seq": 2, "name": "wp_002",
                "latitude_deg": 39.2, "longitude_deg": 117.3, "altitude_m": 21.7,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subject1_waypoints.yaml")
            export_subject1_yaml_from_vehicle_csv(rows, path)
            text = open(path, encoding="utf-8").read()
            self.assertTrue(text.startswith("subject1_waypoints:"))
            self.assertIn("latitude_deg:", text)
            # latitude before longitude in each block
            i_lat = text.find("latitude_deg:")
            i_lon = text.find("longitude_deg:")
            self.assertLess(i_lat, i_lon)


if __name__ == "__main__":
    unittest.main()
