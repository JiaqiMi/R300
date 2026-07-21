"""Tests for competition image-corner calibration TXT (序号;经度;纬度;高程)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from roadnet.gcp_io import (
    build_control_points_from_corner_records,
    looks_like_corner_calibration_txt,
    parse_corner_calibration_txt,
)
from roadnet.geo_calibration import GeoCalibration


SAMPLE = """\
1;105.61346407;39.03046265;0   // 左下角
2;105.63379325;39.03047974;0   // 右下角
3;105.61346378;39.04365226;0   # 左上角
"""


class ParseCornerCalibrationTxtTests(unittest.TestCase):
    def _write(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        Path(path).write_text(text, encoding="utf-8")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_parse_semicolon_and_comments(self):
        path = self._write(SAMPLE)
        parsed = parse_corner_calibration_txt(path)
        self.assertTrue(parsed["ok"], parsed.get("error"))
        self.assertEqual(len(parsed["records"]), 3)
        ids = [r["id"] for r in parsed["records"]]
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(parsed["records"][0]["corner"], "left_bottom")
        self.assertEqual(parsed["records"][1]["name"], "bottom_right")
        self.assertEqual(parsed["records"][2]["name"], "top_left")
        self.assertAlmostEqual(parsed["records"][0]["longitude"], 105.61346407)
        self.assertAlmostEqual(parsed["records"][0]["latitude"], 39.03046265)

    def test_chinese_semicolon_and_comma(self):
        path = self._write(
            "1；105.1；39.0；0\n"
            "2,105.2,39.0,0\n"
            "3\t105.1\t39.1\t0\n"
        )
        parsed = parse_corner_calibration_txt(path)
        self.assertTrue(parsed["ok"], parsed.get("error"))
        self.assertEqual(len(parsed["records"]), 3)

    def test_missing_vertex_errors(self):
        path = self._write("1;105.1;39.0;0\n2;105.2;39.0;0\n")
        parsed = parse_corner_calibration_txt(path)
        self.assertFalse(parsed["ok"])
        self.assertIn("1=左下角", parsed["error"])

    def test_swap_suspect(self):
        # lon/lat swapped: first field looks like lat, second like China lon
        path = self._write(
            "1;39.03046265;105.61346407;0\n"
            "2;39.03047974;105.63379325;0\n"
            "3;39.04365226;105.61346378;0\n"
        )
        parsed = parse_corner_calibration_txt(path)
        self.assertTrue(parsed["ok"] or parsed.get("swap_suspect"))
        # lat=105 is out of range → either error with swap_suspect or ok+swap
        if parsed["ok"]:
            self.assertTrue(parsed["swap_suspect"])
        else:
            self.assertTrue(parsed.get("swap_suspect") or "超出范围" in (parsed.get("error") or ""))

    def test_bind_pixels_and_affine(self):
        path = self._write(SAMPLE)
        self.assertTrue(looks_like_corner_calibration_txt(path))
        parsed = parse_corner_calibration_txt(path)
        w, h = 1000, 800
        cps = build_control_points_from_corner_records(parsed["records"], w, h)
        by_name = {c["name"]: c for c in cps}
        self.assertEqual(by_name["bottom_left"]["pixel"], [0, h - 1])
        self.assertEqual(by_name["bottom_right"]["pixel"], [w - 1, h - 1])
        self.assertEqual(by_name["top_left"]["pixel"], [0, 0])

        geo = GeoCalibration()
        geo.set_control_points(cps)
        geo.set_calibration_metadata("image_corner_3point_affine", w, h)
        geo.calibration_mode = "image_corner_3point_affine"
        self.assertTrue(geo.setup_projection(force_local_enu=True))
        self.assertTrue(geo.compute_affine())
        self.assertTrue(geo.is_valid)

        rt_lon, rt_lat = geo.pixel_to_lonlat(w - 1, 0)
        self.assertTrue(-180 <= rt_lon <= 180)
        self.assertTrue(-90 <= rt_lat <= 90)

        with tempfile.TemporaryDirectory() as tmp:
            cal_path = os.path.join(tmp, "calibration.json")
            self.assertTrue(geo.save(cal_path))
            data = json.loads(Path(cal_path).read_text(encoding="utf-8"))
            self.assertTrue(data["is_valid"])
            self.assertEqual(data["method"], "image_corner_3point_affine")

            geo2 = GeoCalibration()
            self.assertTrue(geo2.load(cal_path))
            self.assertTrue(geo2.is_valid)


if __name__ == "__main__":
    unittest.main()
