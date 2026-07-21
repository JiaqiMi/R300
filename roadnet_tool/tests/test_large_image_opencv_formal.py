"""Integration tests for the large-image OpenCV formal extraction worker."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from roadnet.large_image_worker import LargeImageSegmentationWorker


def _write_road_image(path):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (30, 120, 40)            # 草地背景
    img[180:220, :] = (185, 185, 185)  # 横向道路
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class LargeImageOpenCVFormalTests(unittest.TestCase):
    def test_roi_formal_generates_mask_and_tile_status(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            image_path = os.path.join(tmp, "big.png")
            _write_road_image(image_path)
            output_dir = os.path.join(tmp, "out")
            roi = [[[0, 150], [400, 150], [400, 260], [0, 260]]]
            worker = LargeImageSegmentationWorker(
                image_path=image_path,
                positive_samples_rgb=np.array([[185, 185, 185]], np.uint8),
                negative_samples_rgb=np.array([[30, 120, 40]], np.uint8),
                config={"mode": "combined", "use_negative_samples": True,
                        "blur_kernel": 3, "open_kernel": 3, "close_kernel": 5,
                        "min_area": 100, "fill_holes": False},
                output_dir=output_dir,
                roi_polygons=roi, ignore_polygons=[],
                tile_size=256, overlap=64,
                extraction_label="opencv_roi",
                roi_required=True, mask_type="formal_opencv",
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()

            self.assertEqual(len(results), 1)
            result = results[0]
            # 正式产物齐全
            for name in ("global_road_mask.png", "global_road_mask_preview.png",
                         "formal_extraction_report.json", "tile_status_report.json",
                         "tile_status_overlay.png"):
                self.assertTrue(os.path.exists(os.path.join(output_dir, name)), name)

            report = result.report
            self.assertEqual(report["mask_type"], "formal_opencv")
            self.assertTrue(report["formal_ready"])
            self.assertFalse(report["preview_only"])
            self.assertEqual(report["coordinate_system"], "original_image_pixel")

            with open(os.path.join(output_dir, "tile_status_report.json"),
                      encoding="utf-8") as fh:
                status = json.load(fh)
            self.assertGreater(len(status["tiles"]), 0)
            for record in status["tiles"]:
                for key in ("tile_id", "x0", "y0", "x1", "y1", "intersects_roi",
                            "skipped_black", "cache_hit", "processed", "success",
                            "failed", "mask_nonzero_ratio", "error_message",
                            "output_mask_path"):
                    self.assertIn(key, record)

            # 道路带应被检出
            self.assertGreater(np.count_nonzero(result.processed_mask[185:215, :]), 0)

    def test_roi_required_without_roi_fails(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            image_path = os.path.join(tmp, "big.png")
            _write_road_image(image_path)
            worker = LargeImageSegmentationWorker(
                image_path=image_path,
                positive_samples_rgb=np.array([[185, 185, 185]], np.uint8),
                negative_samples_rgb=np.empty((0, 3), np.uint8),
                config={"mode": "hsv", "use_negative_samples": False},
                output_dir=os.path.join(tmp, "out"),
                roi_polygons=[], ignore_polygons=[],
                tile_size=256, overlap=64,
                extraction_label="opencv_roi",
                roi_required=True, mask_type="formal_opencv",
            )
            finished, failed = [], []
            worker.finished.connect(finished.append)
            worker.failed.connect(lambda *a: failed.append(a))
            worker.run()
            # ROI 必需但缺失：不得生成 mask，必须失败（绝不退化为全图）。
            self.assertEqual(len(finished), 0)
            self.assertEqual(len(failed), 1)

    def test_full_mode_no_roi_processes_all(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            image_path = os.path.join(tmp, "big.png")
            _write_road_image(image_path)
            output_dir = os.path.join(tmp, "out")
            worker = LargeImageSegmentationWorker(
                image_path=image_path,
                positive_samples_rgb=np.array([[185, 185, 185]], np.uint8),
                negative_samples_rgb=np.empty((0, 3), np.uint8),
                config={"mode": "hsv", "use_negative_samples": False},
                output_dir=output_dir,
                roi_polygons=[], ignore_polygons=[],
                tile_size=256, overlap=64,
                extraction_label="opencv_full",
                roi_required=False, mask_type="formal_opencv",
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()
            self.assertEqual(len(results), 1)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "global_road_mask.png")))
            self.assertFalse(results[0].report["roi_used"])


if __name__ == "__main__":
    unittest.main()
