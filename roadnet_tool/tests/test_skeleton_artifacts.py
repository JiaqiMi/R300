import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from roadnet.optimized_skeleton import optimize_skeleton
from roadnet.skeleton_artifacts import (
    build_skeleton_optimize_report, save_skeleton_artifacts,
)


class SkeletonArtifactTests(unittest.TestCase):
    def _sample(self):
        mask = np.zeros((160, 160), dtype=np.uint8)
        cv2.line(mask, (15, 80), (145, 80), 255, 11)
        cv2.line(mask, (80, 15), (80, 145), 255, 11)
        raw = np.zeros_like(mask)
        cv2.line(raw, (15, 80), (145, 80), 255, 1)
        cv2.line(raw, (80, 15), (80, 145), 255, 1)
        cv2.line(raw, (120, 80), (126, 74), 255, 1)
        return mask, raw

    def test_reoptimizing_from_same_raw_is_idempotent(self):
        mask, raw = self._sample()
        kwargs = dict(
            min_center_dist=2.0, border_margin=0, min_branch_length=20,
            max_connect_dist=25, junction_cluster_radius=10,
        )
        first = optimize_skeleton(mask, raw, **kwargs)["optimized_skeleton"]
        second = optimize_skeleton(mask, raw, **kwargs)["optimized_skeleton"]
        np.testing.assert_array_equal(first, second)

    def test_required_artifacts_and_connectivity_report(self):
        mask, raw = self._sample()
        optimized = optimize_skeleton(
            mask, raw, min_center_dist=2.0, border_margin=0,
            min_branch_length=20, max_connect_dist=25,
        )["optimized_skeleton"]
        report = build_skeleton_optimize_report(
            mask, raw, optimized, min_branch_length=20,
            min_center_dist=2.0, endpoint_connect_distance=25,
            skeleton_state_input="raw",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_skeleton_artifacts(temp_dir, mask, raw, optimized, report)
            self.assertEqual(len(paths), 5)
            for path in paths.values():
                self.assertTrue(Path(path).is_file(), path)
            saved = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
            self.assertIn("removed_ratio", saved)
            self.assertIn("connected_components_before", saved)
            self.assertIn("connected_components_after", saved)
            self.assertEqual(saved["skeleton_state_output"], "optimized")


if __name__ == "__main__":
    unittest.main()
