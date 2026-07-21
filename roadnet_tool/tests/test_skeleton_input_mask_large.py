"""Tests for large-image skeleton input mask selection heuristics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class SkeletonInputPathHeuristicsTests(unittest.TestCase):
    def test_preview_name_detection(self):
        from gui.main_window import MainWindow

        hints = MainWindow._PREVIEW_MASK_NAME_HINTS
        self.assertTrue(any("preview" in h for h in hints))

        class Dummy:
            _PREVIEW_MASK_NAME_HINTS = MainWindow._PREVIEW_MASK_NAME_HINTS
            _is_preview_mask_path = MainWindow._is_preview_mask_path

        d = Dummy()
        self.assertTrue(d._is_preview_mask_path("/x/working_road_mask_preview.png"))
        self.assertTrue(d._is_preview_mask_path("/x/cleaned_working_mask_preview.png"))
        self.assertTrue(d._is_preview_mask_path("/x/final_edited_mask_preview.png"))
        self.assertTrue(d._is_preview_mask_path("/x/global_road_mask_preview.png"))
        self.assertFalse(d._is_preview_mask_path("/x/working_road_mask.png"))
        self.assertFalse(d._is_preview_mask_path("/x/cleaned_working_mask.png"))
        self.assertFalse(d._is_preview_mask_path("/x/final_edited_mask.png"))
        self.assertFalse(d._is_preview_mask_path("/x/global_road_mask.png"))

    def test_priority_order_prefers_final_over_cleaned(self):
        """Priority: final > working > cleaned > refined > global."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            final_m = tmp / "final_edited_mask.png"
            cleaned = tmp / "cleaned_working_mask.png"
            working = tmp / "working_road_mask.png"
            global_m = tmp / "global_road_mask.png"
            for p, val in (
                (final_m, 10),
                (cleaned, 50),
                (working, 100),
                (global_m, 200),
            ):
                arr = np.zeros((64, 64), dtype=np.uint8)
                arr[20:40, 10:50] = val
                cv2.imwrite(str(p), arr)

            candidates = [
                ("final_edited_mask", str(final_m)),
                ("working_road_mask", str(working)),
                ("cleaned_working_mask", str(cleaned)),
                ("global_road_mask", str(global_m)),
            ]
            selected = None
            for label, path in candidates:
                if path and os.path.isfile(path):
                    selected = (label, path)
                    break
            self.assertEqual(selected[0], "final_edited_mask")
            arr = cv2.imread(selected[1], cv2.IMREAD_GRAYSCALE)
            self.assertEqual(int(arr.max()), 10)

    def test_priority_falls_back_to_cleaned_without_final(self):
        """Without final, prefer working then cleaned over global."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cleaned = tmp / "cleaned_working_mask.png"
            global_m = tmp / "global_road_mask.png"
            for p, val in ((cleaned, 50), (global_m, 200)):
                arr = np.zeros((64, 64), dtype=np.uint8)
                arr[20:40, 10:50] = val
                cv2.imwrite(str(p), arr)

            candidates = [
                ("final_edited_mask", ""),
                ("working_road_mask", ""),
                ("cleaned_working_mask", str(cleaned)),
                ("global_road_mask", str(global_m)),
            ]
            selected = None
            for label, path in candidates:
                if path and os.path.isfile(path):
                    selected = (label, path)
                    break
            self.assertEqual(selected[0], "cleaned_working_mask")

    def test_report_json_fields(self):
        report = {
            "selected_mask_path": "/a/final_edited_mask.png",
            "mask_source": "final_edited_mask",
            "mask_edit_base": "cleaned_working_mask",
            "mask_shape": [64, 64],
            "mask_nonzero_ratio": 0.1,
            "used_preview_mask": False,
            "cache_used": False,
            "cache_invalidated": True,
            "raw_skeleton_pixel_count": 10,
            "optimized_skeleton_pixel_count": 8,
            "endpoint_count": 2,
            "junction_count": 1,
            "elapsed_seconds": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large_skeleton_generation_report.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(report, f)
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["mask_source"], "final_edited_mask")
            self.assertEqual(data["mask_edit_base"], "cleaned_working_mask")
            self.assertFalse(data["used_preview_mask"])

    def test_large_image_project_has_final_fields(self):
        from roadnet.large_image_project import LargeImageProject

        fields = LargeImageProject.__dataclass_fields__
        self.assertIn("final_edited_mask_path", fields)
        self.assertIn("final_edited_mask_preview_path", fields)
        self.assertIn("mask_edit_base", fields)


if __name__ == "__main__":
    unittest.main()
