import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from roadnet.valid_image import (
    analyze_valid_image_mask, save_valid_mask_outputs,
)


class ValidImageMaskTests(unittest.TestCase):
    def test_only_large_border_connected_black_component_is_invalid(self):
        image = np.full((200, 240, 3), 120, dtype=np.uint8)
        image[:, :30] = 0                     # large border-connected void
        image[70:120, 90:150] = 0             # internal dark road/shadow
        valid, report = analyze_valid_image_mask(
            image, black_threshold=10, min_black_component_area=1000
        )
        self.assertTrue(np.all(valid[:, :25] == 0))
        self.assertTrue(np.all(valid[75:115, 95:145] == 255))
        self.assertEqual(report["border_connected_invalid_pixels"], 6000)
        self.assertGreater(report["internal_black_pixels_kept"], 0)

    def test_small_border_black_component_is_kept(self):
        image = np.full((100, 100, 3), 100, dtype=np.uint8)
        image[:5, :5] = 0
        valid, report = analyze_valid_image_mask(
            image, black_threshold=10, min_black_component_area=100
        )
        self.assertTrue(np.all(valid == 255))
        self.assertEqual(report["border_connected_invalid_pixels"], 0)
        self.assertEqual(report["small_border_components_kept"], 1)

    def test_report_and_mask_are_saved(self):
        image = np.full((64, 64, 3), 80, dtype=np.uint8)
        image[:, :16] = 0
        road = np.zeros((64, 64), dtype=np.uint8)
        road[20:30, :20] = 255
        valid, report = analyze_valid_image_mask(image, 10, 100, road)
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path, report_path = save_valid_mask_outputs(temp_dir, valid, report)
            self.assertIsNotNone(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            data = json.loads(Path(report_path).read_text(encoding="utf-8"))
            for key in (
                "black_threshold", "black_candidate_pixels",
                "border_connected_invalid_pixels", "internal_black_pixels_kept",
                "removed_road_pixels_estimate", "valid_area_ratio",
            ):
                self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
