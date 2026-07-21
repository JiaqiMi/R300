"""Tests for corridor-constrained main road refinement (seed / ROI / task)."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from roadnet.main_road_postprocess import (
    refine_main_road_mask,
    build_main_road_corridor,
    directional_close,
    estimate_road_radius,
    DEFAULT_MAIN_ROAD_CONFIG,
)
from roadnet.large_image_worker import MainRoadRefineWorker


def _broken_road_mask(size=400, gap=12):
    """一条中间断开的主路 + 一个孤立方块（模拟无 seed 的建筑误检）。"""
    m = np.zeros((size, size), np.uint8)
    cv2.line(m, (20, 200), (190, 200), 255, 7)
    cv2.line(m, (190 + gap, 200), (size - 20, 200), 255, 7)
    cv2.rectangle(m, (60, 60), (110, 110), 255, -1)   # 孤立误检块（无 seed）
    return m


def _seed_along_road(size=400):
    return [[(30, 200), (size - 30, 200)]]


class BuildCorridorTests(unittest.TestCase):
    def test_corridor_from_seed(self):
        corridor, info = build_main_road_corridor(
            (400, 400), seed_strokes=_seed_along_road(), config=None)
        self.assertTrue(info["has_seed"])
        self.assertTrue(corridor[200, 200] > 0)      # 种子附近在 corridor 内
        self.assertEqual(corridor[60, 60], 0)        # 远离种子不在 corridor

    def test_corridor_from_roi_and_task(self):
        roi = [[[0, 150], [400, 150], [400, 260], [0, 260]]]
        corridor, info = build_main_road_corridor(
            (400, 400), roi_polygons=roi,
            task_points=[[50, 50]], config=None)
        self.assertTrue(info["has_roi"])
        self.assertTrue(info["has_task"])
        self.assertTrue(corridor[200, 200] > 0)
        self.assertTrue(corridor[50, 50] > 0)


class RefineMainRoadMaskTests(unittest.TestCase):
    def test_refuse_without_constraint(self):
        m = _broken_road_mask()
        refined, report = refine_main_road_mask(m)
        self.assertTrue(report["refused"])
        self.assertEqual(np.count_nonzero(refined), 0)
        self.assertIn("未提供主路约束", " ".join(report["warnings"]))

    def test_seed_removes_unseeded_and_keeps_road(self):
        m = _broken_road_mask()
        stages = {}
        refined, report = refine_main_road_mask(
            m, seed_strokes=_seed_along_road(),
            config={"auto_accept_bridges": True}, stages_out=stages)
        self.assertFalse(report["refused"])
        self.assertTrue(report["used_corridor"])
        # 无 seed 的建筑块被删除
        self.assertGreaterEqual(report["removed_unseeded_components"], 1)
        self.assertEqual(int((refined[60:110, 60:110] > 0).sum()), 0)
        # 主路保留
        self.assertGreater(np.count_nonzero(refined[195:205, :]), 0)
        for key in ("raw_mask_preview", "main_road_corridor_mask",
                    "seed_connected_components", "component_filtered_mask",
                    "skeleton_raw_preview", "pruned_skeleton_preview",
                    "main_road_mask_preview"):
            self.assertIn(key, stages)

    def test_bridge_is_limited_and_recorded(self):
        m = _broken_road_mask(gap=12)
        # 关闭方向闭运算，强制走桥接路径验证桥接约束与上限。
        refined, report = refine_main_road_mask(
            m, seed_strokes=_seed_along_road(),
            config={"auto_accept_bridges": True, "bridge_count_limit": 20,
                    "line_close_length_preview": 1})
        self.assertGreaterEqual(report["bridge_candidate_count"], 1)
        self.assertLessEqual(report["accepted_bridge_count"], 20)
        self.assertIn("bridge_candidates", report)
        # 小间隙 + 有支持 → 应接受桥接并连通
        self.assertGreaterEqual(report["accepted_bridge_count"], 1)
        self.assertGreater(np.count_nonzero(refined[195:205, 190:205]), 0)

    def test_bridge_not_auto_accepted_by_default(self):
        m = _broken_road_mask(gap=12)
        _, report = refine_main_road_mask(
            m, seed_strokes=_seed_along_road(),
            config={"auto_accept_bridges": False, "line_close_length_preview": 1})
        # 默认不自动接受：高置信桥接标为 pending，等待人工确认
        self.assertEqual(report["accepted_bridge_count"], 0)
        self.assertGreaterEqual(report["pending_bridge_count"], 0)

    def test_ignore_applied_last(self):
        m = _broken_road_mask(gap=12)
        ignore = [[[300, 150], [400, 150], [400, 260], [300, 260]]]
        refined, _ = refine_main_road_mask(
            m, seed_strokes=_seed_along_road(), ignore_polygons=ignore,
            config={"auto_accept_bridges": True})
        self.assertEqual(int((refined[150:260, 310:400] > 0).sum()), 0)

    def test_report_fields_present(self):
        m = _broken_road_mask()
        _, report = refine_main_road_mask(m, seed_strokes=_seed_along_road())
        for key in ("seed_stroke_count", "roi_count", "task_point_count",
                    "used_corridor", "component_count_before", "component_count_after",
                    "removed_unseeded_components", "endpoint_count_before",
                    "endpoint_count_after", "bridge_candidate_count",
                    "accepted_bridge_count", "rejected_bridge_count",
                    "edge_count_before", "edge_count_after",
                    "removed_short_branch_count",
                    "mask_nonzero_ratio_before", "mask_nonzero_ratio_after",
                    "warnings"):
            self.assertIn(key, report)

    def test_road_radius_auto_and_explicit(self):
        m = _broken_road_mask()
        self.assertEqual(estimate_road_radius(
            m, {"road_radius_preview": 6, "road_radius_min": 5, "road_radius_max": 8}), 6)
        auto = estimate_road_radius(
            m, {"road_radius_preview": "auto", "road_radius_min": 5,
                "road_radius_max": 8, "fallback_road_radius_preview": 6})
        self.assertTrue(5 <= auto <= 8)

    def test_directional_close_respects_allowed_region(self):
        m = np.zeros((100, 100), np.uint8)
        cv2.line(m, (10, 50), (90, 50), 255, 3)
        allowed = np.zeros_like(m)
        out = directional_close(m, 15, allowed)
        self.assertEqual(np.count_nonzero(out), np.count_nonzero(m > 0))


class MainRoadRefineWorkerTests(unittest.TestCase):
    def test_worker_refuses_without_constraint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            m = _broken_road_mask(size=300)
            worker = MainRoadRefineWorker(m, os.path.join(tmp, "run"))
            finished, failed = [], []
            worker.finished.connect(finished.append)
            worker.failed.connect(lambda *a: failed.append(a))
            worker.run()
            self.assertEqual(len(finished), 0)
            self.assertEqual(len(failed), 1)
            self.assertIn("种子", failed[0][0])

    def test_worker_outputs_and_registers_formal_mask(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            m = _broken_road_mask(size=600, gap=12)
            out_dir = os.path.join(tmp, "main_road_refine", "run_test")
            worker = MainRoadRefineWorker(
                m, out_dir,
                config={"auto_accept_bridges": True, "use_preview_level": True,
                        "preview_max_side": 600},
                seed_strokes=[[(30, 200), (570, 200)]],
            )
            results, failed = [], []
            worker.finished.connect(results.append)
            worker.failed.connect(lambda *a: failed.append(a))
            worker.run()

            self.assertEqual(failed, [])
            self.assertEqual(len(results), 1)
            result = results[0]
            for name in ("raw_mask_preview.png", "main_road_corridor_mask.png",
                         "seed_connected_components.png", "component_filtered_mask.png",
                         "skeleton_raw_preview.png", "graph_edge_score_overlay.png",
                         "bridge_candidates_overlay.png", "accepted_bridges_overlay.png",
                         "pruned_skeleton_preview.png", "main_road_mask_preview.png",
                         "main_road_mask.png", "main_road_refine_report.json"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), name)

            report = result.report
            self.assertEqual(report["mask_type"], "formal_opencv_mainroad_refined")
            self.assertTrue(report["formal_ready"])
            self.assertFalse(report["preview_only"])
            self.assertEqual(report["coordinate_system"], "original_image_pixel")

            with open(os.path.join(out_dir, "main_road_refine_report.json"),
                      encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["input_mask_shape"], [600, 600])

            full = cv2.imread(result.mask_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(full)
            self.assertEqual(full.shape, (600, 600))
            self.assertGreater(np.count_nonzero(full), 0)


if __name__ == "__main__":
    unittest.main()
