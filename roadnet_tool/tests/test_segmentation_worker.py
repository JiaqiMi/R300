import json
import os
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


try:
    import PySide6.QtCore  # noqa: F401
except ImportError:
    class _Signal:
        def __init__(self, *args):
            self.calls = []

        def connect(self, callback):
            self._callback = callback

        def emit(self, *args):
            self.calls.append(args)
            callback = getattr(self, "_callback", None)
            if callback:
                callback(*args)

    class _QObject:
        def __init__(self, parent=None):
            pass

    def _slot(*args, **kwargs):
        return lambda function: function

    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.QObject = _QObject
    qtcore.Signal = _Signal
    qtcore.Slot = _slot
    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore
    sys.modules.setdefault("PySide6", pyside)
    sys.modules.setdefault("PySide6.QtCore", qtcore)


from roadnet.segmentation_worker import (  # noqa: E402
    SegmentationWorker, generate_tile_grid,
)


class SegmentationWorkerTests(unittest.TestCase):
    def test_grid_covers_large_image_without_duplicate_tiles(self):
        tiles = generate_tile_grid(5000, 4300, 1024, 128)
        self.assertEqual(len(tiles), len(set(tiles)))
        self.assertEqual(min(t[0] for t in tiles), 0)
        self.assertEqual(min(t[1] for t in tiles), 0)
        self.assertEqual(max(t[2] for t in tiles), 5000)
        self.assertEqual(max(t[3] for t in tiles), 4300)

    def test_roi_limits_tiles_ignore_is_zero_and_outputs_are_saved(self):
        image = np.zeros((1400, 1600, 3), dtype=np.uint8)
        image[:] = (100, 100, 100)
        image[100:700, 100:700] = (180, 180, 180)
        roi = [[[100, 100], [700, 100], [700, 700], [100, 700]]]
        ignore = [[[300, 300], [450, 300], [450, 450], [300, 450]]]
        with tempfile.TemporaryDirectory() as output_dir:
            worker = SegmentationWorker(
                image_rgb=image,
                positive_samples_rgb=np.array([[180, 180, 180]], np.uint8),
                negative_samples_rgb=np.array([[100, 100, 100]], np.uint8),
                config={"mode": "hsv", "use_negative_samples": False},
                output_dir=output_dir,
                roi_polygons=roi,
                ignore_polygons=ignore,
                tile_size=512,
                overlap=64,
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(np.all(result.processed_mask[:90] == 0))
            self.assertTrue(np.all(result.processed_mask[320:430, 320:430] == 0))
            self.assertGreater(np.count_nonzero(result.processed_mask[120:280, 120:280]), 0)
            for name in (
                "road_mask_raw.png", "road_mask_processed.png",
                "valid_image_mask.png", "valid_mask_report.json",
                "segmentation_report.json", "segmentation_log.txt",
            ):
                self.assertTrue(os.path.exists(os.path.join(output_dir, name)), name)
            with open(os.path.join(output_dir, "segmentation_report.json"),
                      encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertTrue(report["roi_used"])
            self.assertTrue(report["ignore_used"])
            self.assertLess(report["tile_count"], len(generate_tile_grid(1600, 1400, 512, 64)))

    def test_black_tiles_are_skipped_and_cleared(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[0:256, 0:256] = 180
        with tempfile.TemporaryDirectory() as output_dir:
            worker = SegmentationWorker(
                image,
                np.array([[180, 180, 180]], np.uint8),
                np.empty((0, 3), np.uint8),
                {"mode": "hsv", "use_negative_samples": False},
                output_dir,
                tile_size=256,
                overlap=0,
                skip_black_area=True,
                black_threshold=10,
                valid_pixel_ratio_threshold=0.1,
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()
            result = results[0]
            self.assertEqual(result.report["candidate_tile_count"], 4)
            self.assertEqual(result.report["skipped_black_tile_count"], 3)
            self.assertEqual(result.report["tile_count"], 1)
            self.assertTrue(np.all(result.valid_image_mask[300:, 300:] == 0))
            self.assertTrue(np.all(result.processed_mask[300:, 300:] == 0))

    def test_quick_preview_returns_full_resolution_mask(self):
        image = np.full((800, 1000, 3), 140, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as output_dir:
            worker = SegmentationWorker(
                image,
                np.array([[140, 140, 140]], np.uint8),
                np.empty((0, 3), np.uint8),
                {"mode": "hsv", "use_negative_samples": False},
                output_dir,
                tile_size=256,
                overlap=64,
                preview_scale=0.25,
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()
            result = results[0]
            self.assertEqual(result.processed_mask.shape, (800, 1000))
            self.assertEqual(result.report["preview_scale"], 0.25)

    def test_cancel_does_not_write_mask(self):
        with tempfile.TemporaryDirectory() as output_dir:
            worker = SegmentationWorker(
                np.zeros((64, 64, 3), np.uint8),
                np.array([[0, 0, 0]], np.uint8),
                np.empty((0, 3), np.uint8),
                {"mode": "hsv", "use_negative_samples": False},
                output_dir,
                tile_size=32,
                overlap=4,
            )
            worker.cancel()
            worker.run()
            self.assertFalse(os.path.exists(os.path.join(output_dir, "road_mask_raw.png")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "segmentation_log.txt")))

    def test_over_4096_image_completes_with_roi_tiles(self):
        image = np.full((4100, 4100, 3), 150, dtype=np.uint8)
        roi = [[[50, 50], [500, 50], [500, 500], [50, 500]]]
        with tempfile.TemporaryDirectory() as output_dir:
            worker = SegmentationWorker(
                image,
                np.array([[150, 150, 150]], np.uint8),
                np.empty((0, 3), np.uint8),
                {"mode": "hsv", "use_negative_samples": False},
                output_dir,
                roi_polygons=roi,
                tile_size=1024,
                overlap=128,
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.processed_mask.shape, (4100, 4100))
            self.assertEqual(result.report["tile_count"], 1)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "road_mask_processed.png")))

    def test_failure_writes_error_log_without_mask(self):
        with tempfile.TemporaryDirectory() as output_dir:
            worker = SegmentationWorker(
                np.empty((0, 0, 3), np.uint8),
                np.array([[1, 1, 1]], np.uint8),
                np.empty((0, 3), np.uint8),
                {"mode": "hsv", "use_negative_samples": False},
                output_dir,
            )
            failures = []
            worker.failed.connect(lambda *args: failures.append(args))
            worker.run()
            self.assertEqual(failures[0][0], "validate_input")
            self.assertTrue(os.path.exists(os.path.join(output_dir, "segmentation_error.log")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "road_mask_raw.png")))


if __name__ == "__main__":
    unittest.main()
