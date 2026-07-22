"""Tests for judge final_graph view helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.judge_final_graph_view import (  # noqa: E402
    validate_final_graph_for_judge,
    render_judge_overlay_bgr,
    export_judge_overlay_png,
    edge_polyline,
)


class JudgeFinalGraphViewTests(unittest.TestCase):
    def test_validate_ok_and_range_warn(self):
        nodes = [{"id": 1, "x": 10, "y": 10}, {"id": 2, "x": 90, "y": 90}]
        edges = [{"id": 1, "start": 1, "end": 2, "points_pixel": [[10, 10], [90, 90]]}]
        r = validate_final_graph_for_judge(
            has_image=True, image_width=100, image_height=100,
            nodes=nodes, edges=edges,
        )
        self.assertTrue(r["ok"])
        self.assertTrue(r["range_ok"])

        bad_nodes = [{"id": 1, "x": -500, "y": 0}, {"id": 2, "x": 50, "y": 50}]
        bad_edges = [{"id": 1, "points_pixel": [[-500, 0], [50, 50]]}]
        r2 = validate_final_graph_for_judge(
            has_image=True, image_width=100, image_height=100,
            nodes=bad_nodes, edges=bad_edges,
        )
        self.assertTrue(r2["ok"])
        self.assertFalse(r2["range_ok"])
        self.assertTrue(r2["warnings"])

    def test_export_only_image_and_graph(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        img[:] = (40, 60, 80)
        nodes = [{"id": 1, "x": 10, "y": 10}, {"id": 2, "x": 100, "y": 60}]
        edges = [{
            "id": 1, "enabled": True,
            "points_pixel": [[10, 10], [50, 30], [100, 60]],
        }]
        out = render_judge_overlay_bgr(img, nodes, edges, assume_rgb=True)
        self.assertEqual(out.shape[:2], (80, 120))
        # yellow stroke should appear somewhere
        self.assertGreater(int((out[:, :, 0] > 200).sum() + (out[:, :, 1] > 200).sum()), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "judge_final_graph_overlay.png")
            export_judge_overlay_png(path, img, nodes, edges, assume_rgb=True)
            self.assertTrue(os.path.isfile(path))
            loaded = cv2.imread(path)
            self.assertIsNotNone(loaded)

    def test_edge_polyline_aliases(self):
        self.assertEqual(
            edge_polyline({"polyline": [[1, 2], [3, 4]]}),
            [[1.0, 2.0], [3.0, 4.0]],
        )


if __name__ == "__main__":
    unittest.main()
