"""Tests for large-image skeleton preprocess (large_image_mode only)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from roadnet.large_skeleton_preprocess import (
    generate_cleaned_skeleton_large,
    prepare_mask_for_skeleton_large,
)


def _dirty_mask(h=400, w=600):
    """Horizontal main road + noise blobs (buildings / grass)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[195:205, 20:580] = 255
    # gap
    mask[195:205, 290:310] = 0
    # noise blobs
    mask[40:70, 40:80] = 255
    mask[300:340, 400:460] = 255
    mask[100:110, 500:520] = 255
    return mask


class PrepareMaskTests(unittest.TestCase):
    def test_seed_removes_noise_components(self):
        mask = _dirty_mask()
        seeds = [[(30, 200), (570, 200)]]
        cleaned, report = prepare_mask_for_skeleton_large(
            mask, main_road_seed_strokes=seeds,
        )
        self.assertEqual(cleaned.shape, mask.shape)
        self.assertGreater(np.count_nonzero(cleaned), 0)
        # noise blob far from seed should be mostly gone
        self.assertLess(np.count_nonzero(cleaned[40:70, 40:80]), 50)
        self.assertEqual(report["seed_stroke_count"], 1)
        self.assertFalse(report.get("warning"))

    def test_warn_without_constraint_keeps_top_k(self):
        mask = _dirty_mask()
        cleaned, report = prepare_mask_for_skeleton_large(mask)
        self.assertIsNotNone(report.get("warning"))
        self.assertGreater(np.count_nonzero(cleaned), 0)
        # should not keep all noise — cleaned should be smaller than raw
        self.assertLess(np.count_nonzero(cleaned), np.count_nonzero(mask))


class GenerateSkeletonTests(unittest.TestCase):
    def test_outputs_and_artifacts(self):
        mask = _dirty_mask()
        seeds = [[(30, 200), (570, 200)]]
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_cleaned_skeleton_large(
                mask,
                main_road_seed_strokes=seeds,
                output_dir=tmp,
                config={"auto_accept_bridges": True, "line_close_length_preview": 11},
            )
            cleaned = result["cleaned_skeleton"]
            raw = result["raw_skeleton"]
            self.assertEqual(cleaned.shape, mask.shape)
            self.assertGreater(np.count_nonzero(cleaned), 0)
            # cleaned should not have more pixels than a noisy raw typically
            # (at least both exist)
            self.assertGreater(np.count_nonzero(raw), 0)

            out = Path(tmp)
            for name in (
                "skeleton_input_mask_preview.png",
                "skeleton_cleaned_mask_preview.png",
                "raw_skeleton_preview.png",
                "pruned_skeleton_preview.png",
                "bridge_candidates_overlay.png",
                "graph_edge_score_overlay.png",
                "large_skeleton_preprocess_report.json",
                "optimized_skeleton.png",
                "optimized_skeleton_preview.png",
            ):
                self.assertTrue((out / name).is_file(), f"missing {name}")


class PreviewRegionRegression(unittest.TestCase):
    """Brush dirty-rect must not wipe the rest of the preview."""

    def test_update_preview_region_bootstraps_full_preview(self):
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        from gui.layer_manager import LayerManager
        lm = LayerManager()
        lm._large_image_mode = True
        lm._preview_width = 60
        lm._preview_height = 40
        lm._original_width = 600
        lm._original_height = 400
        lm._image_size = (60, 40)
        full = _dirty_mask()
        lm.set_layer_data("mask", full, preview_data=None)
        # clear preview to simulate missing preview after edit path
        layer = lm.layers()["layer_road_mask"]
        layer.preview_data = None
        lm.update_layer_preview_region("layer_road_mask", (20, 190, 40, 210))
        prev = layer.preview_data
        self.assertIsNotNone(prev)
        self.assertEqual(prev.shape[:2], (40, 60))
        # full road should be present, not only the dirty patch
        self.assertGreater(np.count_nonzero(prev), 50)
        _ = app  # keep reference


if __name__ == "__main__":
    unittest.main()
