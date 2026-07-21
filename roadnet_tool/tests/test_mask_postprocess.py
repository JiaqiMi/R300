"""Tests for SAM-Road mask postprocess defaults and uint8 CC safety."""

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

from roadnet.mask_postprocess import (  # noqa: E402
    MaskPostprocessConfig,
    COMPETITION_MASK_POSTPROCESS_DEFAULTS,
    ensure_binary_uint8_mask,
    postprocess_samroad_mask,
)


class MaskPostprocessTests(unittest.TestCase):
    def test_competition_defaults(self):
        cfg = MaskPostprocessConfig.competition_defaults()
        self.assertEqual(cfg.threshold, 240)
        self.assertEqual(cfg.blur_kernel, 3)
        self.assertEqual(cfg.close_kernel, 5)
        self.assertEqual(cfg.open_kernel, 3)
        self.assertEqual(cfg.min_area, 500)
        self.assertFalse(cfg.fill_holes)
        self.assertTrue(cfg.use_roi)
        self.assertTrue(cfg.use_ignore)
        self.assertEqual(COMPETITION_MASK_POSTPROCESS_DEFAULTS["threshold"], 240)

    def test_ensure_binary_uint8_mask_types(self):
        with self.assertRaises(ValueError):
            ensure_binary_uint8_mask(None)

        bool_m = np.array([[True, False], [False, True]])
        out = ensure_binary_uint8_mask(bool_m)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(set(np.unique(out)).issubset({0, 255}))

        float_m = np.array([[0.0, 0.8], [0.2, 0.0]], dtype=np.float32)
        out = ensure_binary_uint8_mask(float_m)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(set(np.unique(out)).issubset({0, 255}))

        u16 = np.array([[0, 1000], [0, 50]], dtype=np.uint16)
        out = ensure_binary_uint8_mask(u16)
        self.assertEqual(out.dtype, np.uint8)

        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        rgb[1:3, 1:3, :] = 200
        out = ensure_binary_uint8_mask(rgb)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.dtype, np.uint8)

        # Must be accepted by connectedComponentsWithStats
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        self.assertGreaterEqual(n, 1)

    def test_postprocess_float_score_no_cc_crash(self):
        # float score map historically caused iDepth assertion
        score = np.zeros((64, 64), dtype=np.float32)
        score[10:50, 10:50] = 250.0
        score[20:30, 20:30] = 100.0  # below threshold 240
        out, steps = postprocess_samroad_mask(score, MaskPostprocessConfig.competition_defaults())
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.ndim, 2)
        self.assertTrue(set(np.unique(out)).issubset({0, 255}))
        self.assertGreater(int((out > 0).sum()), 0)
        self.assertFalse(any("fill_holes" in name for name, _ in steps))

    def test_fill_holes_off_by_default(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:35, 5:35] = 255
        mask[15:25, 15:25] = 0  # hole
        out, steps = postprocess_samroad_mask(mask, MaskPostprocessConfig.competition_defaults())
        # hole should remain (fill_holes=False)
        self.assertEqual(int(out[20, 20]), 0)
        self.assertFalse(any(name.startswith("09_fill_holes") for name, _ in steps))


if __name__ == "__main__":
    unittest.main()
