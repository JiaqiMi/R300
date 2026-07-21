"""Tests for low-res formal working road mask generation."""

import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.lowres_formal_mask import (  # noqa: E402
    DEFAULT_LOWRES_MAX_SIDE,
    LowresFormalMaskConfig,
    build_lowres_work_image,
    generate_lowres_formal_mask,
    upscale_mask_nearest,
)


class UpscaleNearestTests(unittest.TestCase):
    def test_nearest_keeps_binary(self):
        low = np.zeros((10, 20), dtype=np.uint8)
        low[2:8, 5:15] = 255
        up = upscale_mask_nearest(low, 40, 20)
        self.assertEqual(up.shape, (20, 40))
        self.assertEqual(up.dtype, np.uint8)
        vals = set(np.unique(up).tolist())
        self.assertTrue(vals.issubset({0, 255}))
        self.assertGreater(int(np.count_nonzero(up)), 0)


class LowresFormalMaskPipelineTests(unittest.TestCase):
    def test_default_max_side_2500(self):
        self.assertEqual(DEFAULT_LOWRES_MAX_SIDE, 2500)
        self.assertEqual(LowresFormalMaskConfig().max_side, 2500)

    def test_build_work_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((800, 1200, 3), 180, dtype=np.uint8)
            path = os.path.join(tmp, "big.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            work, scale = build_lowres_work_image(path, max_side=400)
            self.assertLessEqual(max(work.shape[0], work.shape[1]), 400)
            self.assertEqual(scale["original_width"], 1200)
            self.assertEqual(scale["original_height"], 800)

    def test_end_to_end_formal_mask_no_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((600, 900, 3), 200, dtype=np.uint8)
            img[280:320, 50:850] = (70, 70, 70)
            img[0:30, :] = 0
            path = os.path.join(tmp, "map.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            pos = np.array([[70, 70, 70], [72, 72, 72]], dtype=np.uint8)
            neg = np.array([[200, 200, 200], [195, 195, 195]], dtype=np.uint8)
            out_dir = os.path.join(tmp, "lowres_run")
            cfg = LowresFormalMaskConfig(
                max_side=450,
                min_component_area=20,
            )
            result = generate_lowres_formal_mask(
                path,
                out_dir,
                max_side=450,
                positive_samples=pos,
                negative_samples=neg,
                config=cfg,
            )
            self.assertTrue(result.ok, result.error or result.warning)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "lowres_work_image.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "lowres_road_mask_raw.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "lowres_road_mask_cleaned.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "working_road_mask.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "working_road_mask_preview.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "lowres_formal_mask_report.json")))
            # must NOT produce graph artifacts
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "final_graph.json")))
            mask = cv2.imread(result.working_mask_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            self.assertEqual(mask.shape, (600, 900))
            vals = set(np.unique(mask).tolist())
            self.assertTrue(vals.issubset({0, 255}))
            with open(os.path.join(out_dir, "lowres_formal_mask_report.json"), encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(report["mask_source"], "lowres_formal_mask")
            self.assertTrue(report["formal_ready"])
            self.assertFalse(report["preview_only"])
            self.assertEqual(report["interpolation"], "INTER_NEAREST")

    def test_ui_wiring(self):
        panel = os.path.join(ROOT, "gui", "parameter_panel.py")
        main = os.path.join(ROOT, "gui", "main_window.py")
        core = os.path.join(ROOT, "roadnet", "lowres_formal_mask.py")
        worker = os.path.join(ROOT, "roadnet", "lowres_formal_mask_worker.py")
        self.assertTrue(os.path.isfile(core))
        self.assertTrue(os.path.isfile(worker))
        with open(panel, encoding="utf-8") as stream:
            panel_src = stream.read()
        with open(main, encoding="utf-8") as stream:
            main_src = stream.read()
        with open(worker, encoding="utf-8") as stream:
            worker_src = stream.read()
        with open(core, encoding="utf-8") as stream:
            core_src = stream.read()

        self.assertIn("低像素快速生成正式 Mask", panel_src)
        self.assertIn("lowres_formal_mask", panel_src)
        self.assertIn("2500.0", panel_src)
        self.assertIn("_on_lowres_formal_mask", main_src)
        self.assertIn("LowresFormalMaskWorker", main_src)
        self.assertIn("return  # ★ 立即返回", main_src)
        self.assertIn("_persist_working_mask", main_src)

        # callback must not run heavy ops
        start = main_src.find("def _on_lowres_formal_mask")
        end = main_src.find("\n    def _on_cancel_lowres_formal_mask")
        body = main_src[start:end]
        for forbidden in (
            "cv2.imread",
            "generate_large_clean_skeleton",
            "final_graph.json",
            "thread.wait",
            "worker.run(",
            "self.setEnabled(False)",
        ):
            self.assertNotIn(forbidden, body)

        self.assertIn("INTER_NEAREST", core_src)
        self.assertIn("segment_road_by_samples", core_src)
        self.assertNotIn("skeleton_to_graph", core_src)
        self.assertNotIn("generate_large_clean_skeleton", core_src)
        self.assertIn("class LowresFormalMaskWorker", worker_src)
        self.assertNotIn("QMessageBox", worker_src)
        self.assertNotIn("QPixmap(", worker_src)


if __name__ == "__main__":
    unittest.main()
