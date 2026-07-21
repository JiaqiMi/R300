"""Tests for Competition Fast Roadnet Mode."""

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

from roadnet.competition_fast_roadnet import (  # noqa: E402
    CompetitionFastConfig,
    DEFAULT_COMPETITION_PREVIEW_MAX_SIDE,
    build_competition_work_image,
    run_competition_fast_roadnet,
    upscale_graph_to_original,
)


class UpscaleGraphTests(unittest.TestCase):
    def test_upscale_maps_nodes_and_polylines_to_original(self):
        graph = {
            "nodes": [{"id": 1, "x": 10, "y": 20, "type": "endpoint"}],
            "edges": [{
                "id": 1, "from": 1, "to": 1,
                "path": [[20, 10], [40, 30]],  # y,x
            }],
        }
        out = upscale_graph_to_original(
            graph, scale_x=2.0, scale_y=3.0,
            original_width=1000, original_height=2000,
        )
        self.assertEqual(out["coordinate_system"], "original_image_pixel")
        self.assertEqual(out["source_mode"], "competition_fast_lowres")
        self.assertEqual(out["nodes"][0]["x"], 20)
        self.assertEqual(out["nodes"][0]["y"], 60)
        self.assertEqual(out["edges"][0]["points_pixel"][0], [20, 60])
        self.assertEqual(out["edges"][0]["points_pixel"][1], [60, 120])


class CompetitionFastPipelineTests(unittest.TestCase):
    def test_default_preview_max_side_is_1500(self):
        self.assertEqual(DEFAULT_COMPETITION_PREVIEW_MAX_SIDE, 1500)
        self.assertEqual(CompetitionFastConfig().competition_preview_max_side, 1500)
        self.assertFalse(CompetitionFastConfig().debug_mode)

    def test_work_image_scale_json_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            # synthetic "large" image saved to disk
            img = np.zeros((800, 1200, 3), dtype=np.uint8)
            img[200:600, 100:1100] = (40, 40, 40)  # dark road-like band
            img[0:40, :] = 0  # black border
            path = os.path.join(tmp, "big.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            work, scale = build_competition_work_image(path, max_side=400)
            self.assertLessEqual(max(work.shape[0], work.shape[1]), 400)
            self.assertEqual(scale["original_width"], 1200)
            self.assertEqual(scale["original_height"], 800)
            self.assertAlmostEqual(
                scale["scale_x"] * scale["work_width"], 1200, delta=2.0
            )

    def test_end_to_end_produces_original_pixel_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Bright background + gray road corridor
            img = np.full((600, 900, 3), 200, dtype=np.uint8)
            img[280:320, 50:850] = (70, 70, 70)
            img[0:30, :] = 0
            path = os.path.join(tmp, "map.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            pos = np.array([[70, 70, 70], [72, 72, 72]], dtype=np.uint8)
            neg = np.array([[200, 200, 200], [195, 195, 195]], dtype=np.uint8)
            out_dir = os.path.join(tmp, "fast_run")
            cfg = CompetitionFastConfig(
                competition_preview_max_side=450,
                min_component_area=20,
                max_total_seconds=90,
                max_segmentation_seconds=60,
                max_skeleton_graph_seconds=60,
            )
            result = run_competition_fast_roadnet(
                path, pos, neg, out_dir, config=cfg,
            )
            self.assertTrue(result.ok, result.error or result.warning)
            self.assertTrue(os.path.isfile(result.final_graph_path))
            with open(result.final_graph_path, encoding="utf-8") as stream:
                graph = json.load(stream)
            self.assertEqual(graph["coordinate_system"], "original_image_pixel")
            self.assertEqual(graph["source_mode"], "competition_fast_lowres")
            self.assertIn("scale_x", graph)
            self.assertGreater(len(graph["nodes"]), 0)
            self.assertGreater(len(graph["edges"]), 0)
            # node coords must be in original range
            for n in graph["nodes"]:
                self.assertGreaterEqual(n["x_pixel"], 0)
                self.assertLess(n["x_pixel"], 900)
                self.assertGreaterEqual(n["y_pixel"], 0)
                self.assertLess(n["y_pixel"], 600)
            report_path = os.path.join(out_dir, "competition_fast_roadnet_report.json")
            self.assertTrue(os.path.isfile(report_path))
            with open(report_path, encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(report["source_mode"], "competition_fast_lowres")
            self.assertFalse(report.get("preview_only", True))
            self.assertTrue(report.get("formal_ready", False))
            # must not depend on task corridor
            self.assertNotIn("task_corridor", report)

    def test_ui_wiring_nonblocking_competition_fast(self):
        panel = os.path.join(ROOT, "gui", "parameter_panel.py")
        main = os.path.join(ROOT, "gui", "main_window.py")
        worker = os.path.join(ROOT, "roadnet", "competition_fast_roadnet_worker.py")
        with open(panel, encoding="utf-8") as stream:
            panel_src = stream.read()
        with open(main, encoding="utf-8") as stream:
            main_src = stream.read()
        self.assertTrue(os.path.isfile(worker))
        with open(worker, encoding="utf-8") as stream:
            worker_src = stream.read()

        self.assertIn("大图比赛快速路网生成", panel_src)
        self.assertIn("取消快速路网生成", panel_src)
        self.assertIn("1500.0", panel_src)
        self.assertIn("应急手绘中心线（可能扣分）", panel_src)

        self.assertIn("competition_fast_roadnet", main_src)
        self.assertIn("_on_competition_fast_roadnet", main_src)
        self.assertIn("CompetitionFastRoadnetWorker", main_src)
        self.assertIn("competition_fast_roadnet_worker", main_src)
        self.assertIn("thread.start()", main_src)
        self.assertIn("return  # ★ 立即返回", main_src)
        # button callback must NOT read pixels / wait on main thread
        start = main_src.find("def _on_competition_fast_roadnet")
        end = main_src.find("\n    def _on_cancel_competition_fast_roadnet")
        self.assertGreater(start, 0)
        self.assertGreater(end, start)
        body = main_src[start:end]
        for forbidden in (
            "read_pixels",
            "cv2.imread",
            "thread.wait",
            "worker.run(",
            "future.result",
            "subprocess.run",
            "self.setEnabled(False)",
            "working_road_mask",
        ):
            self.assertNotIn(forbidden, body)

        self.assertIn("class CompetitionFastRoadnetWorker", worker_src)
        self.assertIn("heartbeat", worker_src)
        self.assertIn("cancel_requested", worker_src)
        self.assertIn("competition_fast_runtime.log", worker_src)
        self.assertIn("run_competition_fast_roadnet", worker_src)
        for forbidden_ui in (
            "QMessageBox",
            "QPixmap",
            "QImage",
            "QWidget",
            "processEvents",
        ):
            self.assertNotIn(f"{forbidden_ui}.", worker_src)
            self.assertNotIn(f"{forbidden_ui}(", worker_src)
        self.assertNotIn("from PySide6.QtWidgets", worker_src)
        self.assertNotIn("from PySide6.QtGui", worker_src)



if __name__ == "__main__":
    unittest.main()
