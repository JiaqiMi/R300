"""Tests for large-image seed-based mask cleaner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadnet.large_mask_cleaner import (
    clean_working_road_mask,
    save_cleaned_mask_artifacts,
)


def _dirty_mask(h=400, w=600):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[195:205, 20:580] = 255
    # noise
    mask[40:70, 40:80] = 255
    mask[300:340, 400:460] = 255
    mask[100:115, 500:530] = 255
    return mask


class CleanWorkingRoadMaskTests(unittest.TestCase):
    def test_refuse_without_seed(self):
        mask = _dirty_mask()
        cleaned, report = clean_working_road_mask(mask)
        self.assertTrue(report.get("refused"))
        self.assertEqual(np.count_nonzero(cleaned), 0)

    def test_seed_keeps_road_removes_noise(self):
        mask = _dirty_mask()
        seeds = [[(30, 200), (570, 200)]]
        cleaned, report = clean_working_road_mask(
            mask, main_road_seed_strokes=seeds,
        )
        self.assertFalse(report.get("refused"))
        self.assertGreater(np.count_nonzero(cleaned), 0)
        self.assertLess(np.count_nonzero(cleaned), np.count_nonzero(mask))
        # far noise should be gone
        self.assertLess(np.count_nonzero(cleaned[40:70, 40:80]), 20)
        self.assertEqual(report["seed_stroke_count"], 1)
        self.assertGreater(report["removed_component_count"], 0)
        for key in (
            "component_count_before", "component_count_after",
            "kept_component_count", "mask_nonzero_ratio_before",
            "mask_nonzero_ratio_after", "close_kernel", "open_kernel",
            "elapsed_seconds", "warnings",
        ):
            self.assertIn(key, report)

    def test_save_artifacts(self):
        mask = _dirty_mask()
        seeds = [[(30, 200), (570, 200)]]
        cleaned, report = clean_working_road_mask(
            mask, main_road_seed_strokes=seeds,
        )
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_cleaned_mask_artifacts(
                cleaned, report, tmp, preview_size=(60, 40),
            )
            for name in (
                "cleaned_working_mask.png",
                "cleaned_working_mask_preview.png",
                "large_mask_clean_report.json",
            ):
                self.assertTrue((Path(tmp) / name).is_file(), name)
            with open(saved["large_mask_clean_report.json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("removed_component_count", data)


if __name__ == "__main__":
    unittest.main()
