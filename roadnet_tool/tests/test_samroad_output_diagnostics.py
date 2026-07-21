import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.samroad_output_diagnostics import diagnose_and_standardize_samroad_outputs  # noqa: E402
from roadnet.samroad_runner import SAMRoadRunResult  # noqa: E402
from roadnet.samroad_single_runner import SAMRoadSingleRunResult  # noqa: E402


class SamRoadOutputDiagnosticsTests(unittest.TestCase):
    def _mask(self, path, value=255):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((32, 48), np.uint8)
        image[8:24, 10:38] = value
        self.assertTrue(cv2.imwrite(str(path), image))
        return image

    def test_recursively_maps_mask_viz_and_graph_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            raw = output / "nested" / "prediction"
            expected = self._mask(raw / "pred_mask.png")
            self._mask(raw / "overlay.png", 120)
            (raw / "pred_graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
            report = diagnose_and_standardize_samroad_outputs(output)
            self.assertTrue(report["road_mask_exists"])
            np.testing.assert_array_equal(cv2.imread(str(output / "road_mask.png"), 0), expected)
            self.assertTrue((output / "viz.png").is_file())
            self.assertTrue((output / "graph.json").is_file())
            self.assertIn("nested/prediction/pred_mask.png", report["files_found"])
            self.assertEqual(Path(report["mask_source"]).name, "pred_mask.png")
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            for key in ("output_dir", "files_found", "road_mask_exists",
                        "candidate_mask_files", "candidate_viz_files", "candidate_graph_files"):
                self.assertIn(key, metadata)

    def test_recent_project_default_output_is_recovered_but_stale_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "portable"
            output = root / "studio_run"
            recent = project / "runs" / "latest" / "mask.png"
            self._mask(recent)
            started = time.time() - 2.0
            report = diagnose_and_standardize_samroad_outputs(
                output, project_dir=project, started_at=started
            )
            self.assertTrue(report["road_mask_exists"])
            self.assertEqual(Path(report["mask_source"]), recent)
            self.assertTrue(any(path.endswith("portable\\runs") or path.endswith("portable/runs")
                                for path in report["project_output_dirs_scanned"]))

            output2 = root / "studio_run_stale"
            stale = project / "save" / "old" / "road_pred.png"
            self._mask(stale)
            old = time.time() - 7200
            os.utime(stale, (old, old))
            # The recent mask in runs is also now older than this run's tolerance.
            old_recent = time.time() - 7200
            os.utime(recent, (old_recent, old_recent))
            report2 = diagnose_and_standardize_samroad_outputs(
                output2, project_dir=project, started_at=time.time()
            )
            self.assertFalse(report2["road_mask_exists"])

    def test_missing_mask_is_explicit_and_process_code_alone_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "result.json").write_text("{}", encoding="utf-8")
            report = diagnose_and_standardize_samroad_outputs(output)
            self.assertFalse(report["road_mask_exists"])
            self.assertFalse(report["output_validation_success"])
            self.assertEqual(report["candidate_mask_files"], [])
            single = SAMRoadSingleRunResult.from_process_result(0, output, "", "")
            legacy = SAMRoadRunResult.from_process_result(0, output, "", "")
            self.assertFalse(single.success)
            self.assertFalse(legacy.success)

    def test_all_requested_mask_aliases_are_recognized(self):
        aliases = ["mask.png", "pred_mask.png", "road_pred.png", "road_prediction.png",
                   "seg.png", "segmentation.png", "binary_mask.png", "output_mask.png"]
        for alias in aliases:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self._mask(output / "sub" / alias)
                report = diagnose_and_standardize_samroad_outputs(output)
                self.assertTrue(report["road_mask_exists"])

    def test_gui_requires_mask_and_exposes_diagnostic_actions(self):
        dialog_source = (ROOT / "gui" / "samroad_single_run_dialog.py").read_text(encoding="utf-8")
        main_source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        for text in (
            "diagnose_and_standardize_samroad_outputs", "road_mask_exists",
            "打开输出目录", "打开 stdout", "打开 stderr", "打开 SAM-RoadPlus 工程目录",
            "candidate_mask_files",
        ):
            self.assertIn(text, dialog_source)
        self.assertIn("config.output_dir = config.output_dir.expanduser().resolve()", dialog_source)
        self.assertIn("_samroad_pipeline_mask_imported", main_source)
        self.assertIn("if self._samroad_pipeline_mask_imported", main_source)


if __name__ == "__main__":
    unittest.main()
