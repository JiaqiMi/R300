import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.samroad_single_runner import SAMRoadSingleRunConfig  # noqa: E402
from roadnet.samroad_tile_worker import SAMRoadTileWorker  # noqa: E402


class MockSAMRoadTileWorker(SAMRoadTileWorker):
    def _run_process(self, command, env):
        image_path = command[command.index("--image") + 1]
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        tile = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray > 0, 200, 0).astype(np.uint8)
        cv2.imwrite(str(output_dir / "road_mask.png"), mask)
        return 0, "mock tile ok", ""


class SAMRoadTileWorkerTests(unittest.TestCase):
    def test_tile_inference_skips_black_and_merges_to_original_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((384, 384, 3), dtype=np.uint8)
            image[:192, :192] = (160, 160, 160)
            image_path = root / "large.png"
            cv2.imwrite(str(image_path), image)
            output_dir = root / "output"

            config = SAMRoadSingleRunConfig(
                project_dir=root,
                python_executable=Path("python"),
                infer_script=root / "infer_single.py",
                config_path=root / "config.yaml",
                samroad_model_ckpt_path=root / "model.ckpt",
                input_image=image_path,
            )
            worker = MockSAMRoadTileWorker(
                config, str(output_dir), tile_size=128, overlap=0,
                skip_black_tile=True, black_threshold=10,
                valid_pixel_ratio_threshold=0.1, merge_method="max",
            )
            results = []
            worker.finished.connect(results.append)
            worker.run()
            self.assertEqual(len(results), 1)
            merged = cv2.imread(str(output_dir / "road_mask.png"), cv2.IMREAD_GRAYSCALE)
            valid = cv2.imread(str(output_dir / "valid_image_mask.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(merged.shape, (384, 384))
            self.assertGreater(np.count_nonzero(merged[:192, :192]), 0)
            self.assertEqual(np.count_nonzero(merged[220:, 220:]), 0)
            self.assertEqual(np.count_nonzero(valid[220:, 220:]), 0)
            self.assertTrue((output_dir / "valid_mask_report.json").is_file())
            with open(output_dir / "samroad_tile_report.json", encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(report["candidate_tile_count"], 9)
            self.assertGreater(report["skipped_black_tile_count"], 0)
            self.assertEqual(report["merge_method"], "max")


if __name__ == "__main__":
    unittest.main()
