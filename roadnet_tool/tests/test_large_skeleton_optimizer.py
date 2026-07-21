"""Tests for large-image clean skeleton optimizer (large_image_mode only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from roadnet.large_skeleton_optimizer import (
    distance_transform_centerline_filter,
    generate_large_clean_skeleton,
    mask_preclean,
)


def _dirty_mask(h=400, w=600):
    mask = np.zeros((h, w), dtype=np.uint8)
    # thick main road
    mask[185:215, 20:580] = 255
    mask[185:215, 290:305] = 0  # small gap
    # noise blobs / building texture
    mask[40:70, 40:80] = 255
    mask[300:340, 400:460] = 255
    mask[100:108, 500:520] = 255
    # edge spur attached to road
    mask[170:185, 100:106] = 255
    return mask


class MaskPrecleanTests(unittest.TestCase):
    def test_seed_removes_noise(self):
        mask = _dirty_mask()
        seeds = [[(30, 200), (570, 200)]]
        cleaned, preview, report = mask_preclean(
            mask, main_road_seed_strokes=seeds,
        )
        self.assertEqual(cleaned.shape, mask.shape)
        self.assertGreater(np.count_nonzero(cleaned), 0)
        self.assertLess(np.count_nonzero(cleaned[40:70, 40:80]), 50)
        self.assertEqual(report["seed_stroke_count"], 1)

    def test_no_constraint_warns(self):
        mask = _dirty_mask()
        cleaned, _, report = mask_preclean(mask)
        self.assertIsNotNone(report.get("warning"))
        self.assertLess(np.count_nonzero(cleaned), np.count_nonzero(mask))


class CenterlineFilterTests(unittest.TestCase):
    def test_removes_low_distance_points(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:60, 10:190] = 255
        from roadnet.optimized_skeleton import skeletonize_thin
        raw = skeletonize_thin(mask)
        # inject artificial edge spur pixels with low distance
        raw[40, 50] = 255
        raw[59, 80] = 255
        filtered, dist, min_c, info = distance_transform_centerline_filter(mask, raw)
        self.assertGreater(info["raw_pixels"], 0)
        self.assertLessEqual(info["kept_pixels"], info["raw_pixels"])
        self.assertGreaterEqual(min_c, 2.0)
        # edge pixels should be gone
        self.assertEqual(int(filtered[40, 50]), 0)
        self.assertEqual(int(filtered[59, 80]), 0)


class GenerateCleanSkeletonTests(unittest.TestCase):
    def test_pipeline_artifacts_and_noise_reduction(self):
        mask = _dirty_mask()
        seeds = [[(30, 200), (570, 200)]]
        with tempfile.TemporaryDirectory() as tmp:
            cleaned, graph, pack = generate_large_clean_skeleton(
                mask,
                main_road_seed_strokes=seeds,
                output_dir=tmp,
                input_meta={
                    "selected_mask_path": "final_edited_mask.png",
                    "mask_source": "final_edited_mask",
                    "checksum": "abc",
                    "file_modified_time": "2026-01-01T00:00:00",
                },
            )
            self.assertEqual(cleaned.shape, mask.shape)
            self.assertGreater(np.count_nonzero(cleaned), 0)
            raw = pack["raw_skeleton"]
            # cleaned should be sparser than raw (noise pruned)
            self.assertLessEqual(
                np.count_nonzero(cleaned), np.count_nonzero(raw) + 50
            )
            report = pack["report"]
            self.assertEqual(report["pipeline"], "generate_large_clean_skeleton")
            self.assertIn("center_filtered_skeleton_pixels", report)
            self.assertIn("pruned_graph_edges", report)

            out = Path(tmp)
            for name in (
                "raw_skeleton_preview.png",
                "center_filtered_skeleton_preview.png",
                "pruned_skeleton_preview.png",
                "skeleton_graph_raw.json",
                "skeleton_graph_pruned.json",
                "large_skeleton_report.json",
            ):
                self.assertTrue((out / name).is_file(), msg=name)

            with (out / "large_skeleton_report.json").open(encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["mask_source"], "final_edited_mask")
            self.assertIn("accepted_bridge_count", data)

            nodes = graph.get("nodes") or []
            edges = graph.get("edges") or []
            self.assertIsInstance(nodes, list)
            self.assertIsInstance(edges, list)


if __name__ == "__main__":
    unittest.main()
