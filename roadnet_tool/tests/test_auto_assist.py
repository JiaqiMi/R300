import copy
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

from roadnet.graph_auto_repair import (  # noqa: E402
    apply_repair_candidates,
    generate_repair_candidates,
    save_auto_repair_bundle,
    suggest_component_bridge,
)
from roadnet.graph_diagnostics import analyze_graph  # noqa: E402
from roadnet.mask_quality_filter import (  # noqa: E402
    candidate_mask_from_runs,
    filter_mask_quality,
)


class MaskQualityFilterTests(unittest.TestCase):
    def test_candidates_are_exact_road_components_and_large_block_is_manual(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (10, 40), (220, 49), 255, -1)
        cv2.rectangle(mask, (260, 10), (264, 14), 255, -1)
        cv2.rectangle(mask, (150, 150), (270, 270), 255, -1)
        result = filter_mask_quality(mask, valid_image_mask=np.full_like(mask, 255))
        self.assertGreater(np.count_nonzero(result.cleaned_mask[40:50, 10:221]), 1800)
        self.assertEqual(np.count_nonzero(result.cleaned_mask[10:15, 260:265]), 0)
        # A >1% component is warning-only, so the hypothetical cleaned mask keeps it.
        self.assertGreater(np.count_nonzero(result.cleaned_mask[150:271, 150:271]), 14000)
        block = max(result.candidate_ignore_regions, key=lambda item: item["area"])
        self.assertFalse(block["auto_apply_eligible"])
        self.assertGreater(block["area_ratio"], 0.01)
        for candidate in result.candidate_ignore_regions:
            exact = candidate_mask_from_runs(candidate, mask.shape)
            self.assertEqual(np.count_nonzero(exact & (mask == 0)), 0)
            self.assertEqual(np.count_nonzero(exact), candidate["area"])

    def test_invalid_area_and_protected_graph_never_auto_apply(self):
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[0:5, 0:5] = 255
        mask[50:55, 50:55] = 255
        valid = np.full_like(mask, 255)
        valid[0:10, 0:10] = 0
        graph = {
            "nodes": [{"id": 1, "x": 45, "y": 52}, {"id": 2, "x": 60, "y": 52}],
            "edges": [{"id": 1, "start": 1, "end": 2,
                       "points_pixel": [[45, 52], [60, 52]], "enabled": True}],
        }
        result = filter_mask_quality(mask, valid_image_mask=valid, final_graph=graph)
        self.assertEqual(result.report["invalid_area_road_pixels_skipped"], 25)
        self.assertEqual(len(result.candidate_ignore_regions), 1)
        candidate = result.candidate_ignore_regions[0]
        self.assertTrue(candidate["near_final_graph"])
        self.assertFalse(candidate["auto_apply_eligible"])

    def test_total_high_confidence_area_over_eight_percent_is_blocked(self):
        mask = np.zeros((140, 140), dtype=np.uint8)
        for y in range(0, 140, 7):
            for x in range(0, 140, 7):
                mask[y:y + 3, x:x + 3] = 255
        result = filter_mask_quality(mask)
        self.assertGreater(result.report["total_ignore_area_ratio"], 0.08)
        self.assertTrue(result.report["auto_apply_blocked"])
        # A blocked batch is a preview only: it must never silently feed an
        # already-erased mask into skeleton generation.
        np.testing.assert_array_equal(result.cleaned_mask, mask)

    def test_writes_all_preview_and_report_outputs(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[5:8, 5:8] = 255
        with tempfile.TemporaryDirectory() as output_dir:
            result = filter_mask_quality(mask, output_dir=output_dir)
            for name in (
                    "cleaned_mask.png", "candidate_ignore_regions.json",
                    "mask_filter_report.json", "mask_before_ignore.png",
                    "mask_after_ignore.png", "ignore_candidates_overlay.png"):
                self.assertTrue(os.path.isfile(os.path.join(output_dir, name)), name)
            with open(os.path.join(output_dir, "mask_filter_report.json"), encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(report["candidate_count"], len(result.candidate_ignore_regions))
            self.assertEqual(report["candidate_source"], "road_mask_foreground_components_only")
            self.assertFalse(report["roi_used_for_candidate_generation"])


class GraphAutoRepairTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            {"id": 1, "x": 10, "y": 20}, {"id": 2, "x": 50, "y": 20},
            {"id": 3, "x": 60, "y": 20}, {"id": 4, "x": 100, "y": 20},
        ]
        self.edges = [
            {"id": 1, "start": 1, "end": 2, "length_pixel": 40,
             "points_pixel": [[10, 20], [50, 20]], "enabled": True},
            {"id": 2, "start": 3, "end": 4, "length_pixel": 40,
             "points_pixel": [[60, 20], [100, 20]], "enabled": True},
        ]

    def test_diagnostics_and_high_confidence_bridge(self):
        diagnostics = analyze_graph(self.nodes, self.edges)
        self.assertEqual(diagnostics["connected_components"], 2)
        candidates = generate_repair_candidates(self.nodes, self.edges, diagnostics)
        self.assertTrue(any(
            item["type"] == "connect_endpoints"
            and {item["node_a"], item["node_b"]} == {2, 3}
            for item in candidates
        ))
        before_nodes, before_edges = copy.deepcopy(self.nodes), copy.deepcopy(self.edges)
        new_nodes, new_edges, report = apply_repair_candidates(
            self.nodes, self.edges, candidates, confidence_threshold=0.80
        )
        self.assertEqual(self.nodes, before_nodes)
        self.assertEqual(self.edges, before_edges)
        self.assertGreaterEqual(report["added_edge_count"], 1)
        self.assertEqual(analyze_graph(new_nodes, new_edges)["connected_components"], 1)

    def test_planning_failure_bridge_and_output_bundle(self):
        candidate = suggest_component_bridge(self.nodes, self.edges, 1, 4, max_distance=50)
        self.assertIsNotNone(candidate)
        new_nodes, new_edges, apply_report = apply_repair_candidates(
            self.nodes, self.edges, [candidate], selected_ids={candidate["id"]}
        )
        before, after = analyze_graph(self.nodes, self.edges), analyze_graph(new_nodes, new_edges)
        with tempfile.TemporaryDirectory() as output_dir:
            save_auto_repair_bundle(
                output_dir, self.nodes, self.edges, new_nodes, new_edges,
                before, after, [candidate], apply_report,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
            )
            for name in (
                    "graph_diagnostics_report.json", "repair_candidates.json",
                    "final_graph_before_repair.json", "final_graph_after_repair.json",
                    "repair_overlay.png", "auto_repair_report.json"):
                self.assertTrue(os.path.isfile(os.path.join(output_dir, name)), name)


class AutoAssistUiWiringTests(unittest.TestCase):
    def test_buttons_history_preflight_and_planning_bridge_are_wired(self):
        with open(os.path.join(ROOT, "gui", "parameter_panel.py"), encoding="utf-8") as stream:
            panel = stream.read()
        for action in (
                "analyze_mask_quality", "apply_mask_candidates", "view_mask_candidates",
                "diagnose_graph", "apply_graph_repairs", "view_graph_repairs"):
            self.assertIn(f'apply_requested.emit("{action}")', panel)
        self.assertIn("confidence >= 0.90", panel)
        with open(os.path.join(ROOT, "gui", "main_window.py"), encoding="utf-8") as stream:
            main = stream.read()
        self.assertIn('self._history.push_state("auto_graph_repair")', main)
        self.assertIn('self._history.push_state("apply_auto_ignore_candidates")', main)
        self.assertIn("add_candidate_runs_to_mask", main)
        self.assertIn("应用高置信 Ignore - 预检", main)
        self.assertIn("max_total_ignore_area_ratio", main)
        self.assertIn("def _suggest_planning_bridge", main)
        self.assertIn("是否应用该修复并重新规划", main)
        self.assertIn('self.skeleton_state: str = "none"', main)
        self.assertIn('if self.skeleton_state == "optimized"', main)


if __name__ == "__main__":
    unittest.main()
