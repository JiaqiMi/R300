import csv
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.task_point_coordinates import (  # noqa: E402
    build_task_point_debug_rows, convert_task_points_to_image, save_task_points_debug_csv,
)
from roadnet.task_points import (  # noqa: E402
    TaskPoint, apply_lon_lat_swap, get_plan_sequence, load_task_points,
    normalize_task_point_sequence, parse_task_points_txt, save_task_points_import_report,
)


class FakeCalibration:
    is_valid = True
    image_width = 1000
    image_height = 500

    def __init__(self):
        self.calls = []

    def wgs84_to_pixel(self, lon, lat):
        self.calls.append((lon, lat))
        return (lon - 117.0) * 10000.0, (39.0 - lat) * 10000.0

    def pixel_to_wgs84(self, x, y):
        return 117.0 + x / 10000.0, 39.0 - y / 10000.0


class TaskPointImportTests(unittest.TestCase):
    def _file(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_five_columns_seq_lon_lat_alt_type(self):
        path = self._file(
            "1;105.62300;39.29636;0;0\n"
            "2;105.62337;39.29744;0;2\n"
            "5;105.61180;39.30524;0;1\n"
        )
        points = load_task_points(path)
        self.assertEqual([p.seq for p in points], [1, 2, 5])
        self.assertEqual([p.point_type for p in points], [0, 2, 1])
        self.assertAlmostEqual(points[0].longitude, 105.62300)
        self.assertAlmostEqual(points[0].latitude, 39.29636)
        self.assertEqual(points[0].source, "file_import")

    def test_header_and_line_comments(self):
        path = self._file(
            "序号;经度;纬度;高程;属性\n"
            "1;105.62300;39.29636;0;0   // 起点\n"
            "2;105.62337;39.29744;0;2   # 必经\n"
            "3;105.61180;39.30524;0;1\n"
        )
        parsed = parse_task_points_txt(path)
        self.assertTrue(parsed["ok"], parsed.get("error"))
        self.assertEqual(len(parsed["points"]), 3)
        self.assertEqual(parsed["start_count"], 1)
        self.assertEqual(parsed["goal_count"], 1)

    def test_english_header_and_chinese_semicolon(self):
        path = self._file(
            "seq;longitude;latitude;altitude;type\n"
            "1；105.62；39.29；0；0\n"
            "2；105.63；39.30；0；1\n"
        )
        parsed = parse_task_points_txt(path)
        self.assertTrue(parsed["ok"], parsed.get("error"))
        self.assertEqual(len(parsed["points"]), 2)

    def test_optional_sixth_reserve_column_still_accepted(self):
        path = self._file(
            "序列;经度;纬度;高程;点属性;备用\n"
            "1;117.0123;38.9876;5.5;0;start-note\n"
            "2;117.0200;38.9900;0;1;\n"
        )
        point = load_task_points(path)[0]
        self.assertEqual(point.seq, 1)
        self.assertAlmostEqual(point.longitude, 117.0123)
        self.assertAlmostEqual(point.latitude, 38.9876)
        self.assertEqual(point.reserve, "start-note")

    def test_swap_suspect_does_not_silent_swap(self):
        path = self._file(
            "1;38.9876;117.0123;0;0\n"
            "2;38.9900;117.0200;0;1\n"
        )
        parsed = parse_task_points_txt(path)
        self.assertTrue(parsed["swap_suspect"])
        self.assertTrue(parsed["ok"] or parsed.get("points"))
        points = list(parsed["points"])
        self.assertAlmostEqual(points[0].longitude, 38.9876)
        self.assertAlmostEqual(points[0].latitude, 117.0123)
        apply_lon_lat_swap(points)
        self.assertAlmostEqual(points[0].longitude, 117.0123)
        self.assertAlmostEqual(points[0].latitude, 38.9876)

    def test_missing_fifth_column_is_rejected(self):
        path = self._file("1;117.0;39.0;0\n")
        with self.assertRaisesRegex(ValueError, "序号;经度;纬度;高程;属性"):
            load_task_points(path)

    def test_invalid_point_type_rejected(self):
        path = self._file("1;105.62;39.29;0;9\n2;105.63;39.30;0;1\n")
        parsed = parse_task_points_txt(path)
        self.assertFalse(parsed["ok"])
        self.assertIn("0 起点", parsed["error"])

    def test_start_goal_count_must_be_one(self):
        path = self._file(
            "1;105.62;39.29;0;0\n"
            "2;105.63;39.30;0;0\n"
            "3;105.64;39.31;0;1\n"
        )
        parsed = parse_task_points_txt(path)
        self.assertFalse(parsed["ok"])
        self.assertIn("起点/终点数量异常", parsed["error"])

    def test_plan_order_is_seq_not_spatial(self):
        points = [
            TaskPoint(3, 105.64, 39.31, point_type=1, source="file_import"),
            TaskPoint(1, 105.62, 39.29, point_type=0, source="file_import"),
            TaskPoint(2, 105.90, 39.50, point_type=2, source="file_import"),
        ]
        order = get_plan_sequence(points)
        self.assertEqual([p.seq for p in order], [1, 2, 3])
        normalize_task_point_sequence(points)
        self.assertEqual([p.seq for p in points], [1, 2, 3])
        self.assertEqual([p.point_type for p in points], [0, 2, 1])

    def test_manual_role_normalization_and_plan_order(self):
        points = [
            TaskPoint(1, None, None, point_type=0, created_order=0, source="manual_click"),
            TaskPoint(2, None, None, point_type=1, created_order=1, source="manual_click"),
            TaskPoint(3, None, None, point_type=2, created_order=2, source="manual_click"),
        ]
        normalize_task_point_sequence(points)
        self.assertEqual([(point.seq, point.point_type) for point in points], [(1, 0), (2, 2), (3, 1)])
        self.assertEqual([point.seq for point in get_plan_sequence(list(reversed(points)))], [1, 2, 3])

    def test_geo_conversion_uses_lon_lat_order_and_scales_image_size(self):
        point = TaskPoint(1, 117.04, 38.98, point_type=0)
        calibration = FakeCalibration()
        rows, size_info = convert_task_points_to_image([point], calibration, (500, 250))
        self.assertEqual(calibration.calls, [(117.04, 38.98)])
        self.assertAlmostEqual(point.pixel_x, 200.0)
        self.assertAlmostEqual(point.pixel_y, 100.0)
        self.assertTrue(point.inside_image)
        self.assertTrue(size_info["size_mismatch"])
        self.assertLess(rows[0]["roundtrip_error_m"], 1e-5)

    def test_import_report_json(self):
        with tempfile.TemporaryDirectory() as directory:
            report = {
                "file_path": "a.txt",
                "point_count": 2,
                "start_count": 1,
                "goal_count": 1,
                "waypoint_count": 0,
                "invalid_rows": 0,
                "geo_calibration_valid": True,
                "converted_pixel_count": 2,
                "out_of_image_count": 0,
                "warnings": [],
            }
            path = save_task_points_import_report(report, directory)
            self.assertTrue(path.endswith("task_points_import_report.json"))
            with open(path, encoding="utf-8") as stream:
                loaded = json.load(stream)
            self.assertEqual(loaded["point_count"], 2)

    def test_debug_csv_contains_nearest_edge_and_required_columns(self):
        point = TaskPoint(1, 117.0, 39.0, point_type=0, pixel_x=10.0, pixel_y=12.0)
        edges = [{"id": 7, "points_pixel": [[0, 10], [30, 10]], "enabled": True}]
        rows = build_task_point_debug_rows([point], (100, 100), edges)
        self.assertEqual(rows[0]["nearest_edge_id"], 7)
        self.assertAlmostEqual(rows[0]["distance_px"], 2.0)
        with tempfile.TemporaryDirectory() as directory:
            path = save_task_points_debug_csv(rows, os.path.join(directory, "task_points_debug.csv"))
            with open(path, encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, [
                    "seq", "lon", "lat", "altitude", "point_type", "pixel_x", "pixel_y",
                    "inside_image", "nearest_edge_id", "distance_px", "status",
                ])


class TaskPointUiWiringTests(unittest.TestCase):
    def test_import_uses_parse_and_geo_conversion(self):
        with open(os.path.join(ROOT, "gui", "main_window.py"), encoding="utf-8") as stream:
            source = stream.read()
        import_body = source.split("def _on_import_task_points", 1)[1].split("def _on_manage_task_points", 1)[0]
        self.assertIn("parse_task_points_txt", import_body)
        self.assertIn("convert_task_points_to_image", import_body)
        self.assertIn("geo_calibration", import_body)
        self.assertIn("请先完成坐标校准", import_body)
        self.assertIn("检测到经纬度可能填反", import_body)
        self.assertNotIn("< 20000", import_body)
        self.assertIn("source=\"manual_click\"", source)
        self.assertIn("pixel_to_wgs84", source)
        self.assertIn("验证任务点坐标", source)
        self.assertIn("TaskPointManagerDialog", source)


if __name__ == "__main__":
    unittest.main()
