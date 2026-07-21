"""Tests for large-image local graph repair (polyline / snap / split / ROI)."""

import math
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.graph_editor_qt import GraphEditorQt  # noqa: E402
from roadnet.local_graph_repair import rebuild_graph_in_roi  # noqa: E402
from roadnet.global_planner import expand_edge_path_to_polyline  # noqa: E402


class PolylineRepairTests(unittest.TestCase):
    def test_add_manual_edge_saves_polyline_and_snaps(self):
        ge = GraphEditorQt()
        ge.configure_large_repair_snaps(node_snap=25, endpoint_snap=25)
        n1 = ge.add_node(100, 100, node_type="junction")
        n2 = ge.add_node(300, 100, node_type="junction")
        eid = ge.add_manual_edge(
            [(102, 101), (200, 120), (298, 99)],
            source="manual_repair",
        )
        self.assertIsNotNone(eid)
        edge = next(e for e in ge.edges if e["id"] == eid)
        self.assertEqual(edge["source"], "manual_repair")
        self.assertIn("points_pixel", edge)
        self.assertIn("polyline", edge)
        self.assertGreaterEqual(len(edge["points_pixel"]), 3)
        self.assertEqual(edge["start"], n1)
        self.assertEqual(edge["end"], n2)
        self.assertGreater(edge["length_pixel"], 0)

    def test_split_edge_inserts_node(self):
        ge = GraphEditorQt()
        a = ge.add_node(0, 0)
        b = ge.add_node(100, 0)
        ge.add_edge(a, b)
        eid = ge.edges[0]["id"]
        mid = ge.split_edge_at_point(eid, 50, 0)
        self.assertIsNotNone(mid)
        self.assertEqual(len(ge.edges), 2)
        self.assertTrue(any(e["start"] == mid or e["end"] == mid for e in ge.edges))

    def test_merge_nodes_updates_polyline_endpoints(self):
        ge = GraphEditorQt()
        a = ge.add_node(0, 0, node_type="junction")
        b = ge.add_node(10, 0, node_type="junction")
        c = ge.add_node(100, 0)
        ge.add_edge(a, c)
        ge.merge_nodes(a, b)
        self.assertEqual(len(ge.nodes), 2)
        for e in ge.edges:
            pts = e["points_pixel"]
            self.assertEqual(pts[0][0], e["start"] and pts[0][0])  # smoke
            nmap = {n["id"]: n for n in ge.nodes}
            self.assertEqual(pts[0], [nmap[e["start"]]["x"], nmap[e["start"]]["y"]])
            self.assertEqual(pts[-1], [nmap[e["end"]]["x"], nmap[e["end"]]["y"]])

    def test_planner_requires_polyline(self):
        nodes = [{"id": 1, "x": 0, "y": 0}, {"id": 2, "x": 10, "y": 0}]
        edges = [{
            "id": 1, "start": 1, "end": 2, "length_pixel": 10,
            "points_pixel": [[0, 0], [5, 1], [10, 0]], "enabled": True,
        }]
        poly = expand_edge_path_to_polyline([1, 2], [1], nodes, edges)
        self.assertGreaterEqual(len(poly), 3)


class LocalRoiRebuildTests(unittest.TestCase):
    def test_roi_rebuild_rejects_huge_roi(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, :] = 255
        nodes = [{"id": 1, "x": 10, "y": 50, "type": "endpoint", "source": "auto"}]
        edges = []
        # huge polygon relative to max_roi_side=50
        roi = [[0, 0], [99, 0], [99, 99], [0, 99]]
        _, _, report = rebuild_graph_in_roi(
            mask, nodes, edges, roi, max_roi_side=50, margin_px=0,
        )
        self.assertFalse(report.get("ok"))
        self.assertIn("过大", report.get("error", ""))

    def test_roi_rebuild_produces_local_edges(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[95:105, 20:180] = 255
        nodes = [
            {"id": 1, "x": 10, "y": 100, "type": "endpoint", "source": "auto"},
            {"id": 2, "x": 190, "y": 100, "type": "endpoint", "source": "auto"},
        ]
        edges = [{
            "id": 1, "start": 1, "end": 2, "length_pixel": 180,
            "points_pixel": [[10, 100], [190, 100]],
            "polyline": [[10, 100], [190, 100]],
            "source": "auto", "enabled": True,
        }]
        roi = [[40, 70], [160, 70], [160, 130], [40, 130]]
        new_nodes, new_edges, report = rebuild_graph_in_roi(
            mask, nodes, edges, roi, margin_px=5, max_roi_side=2500,
        )
        self.assertTrue(report.get("ok"), report.get("error"))
        self.assertTrue(any(e.get("source") == "local_repair" for e in new_edges))
        for e in new_edges:
            if e.get("source") == "local_repair":
                self.assertIn("polyline", e)
                self.assertGreaterEqual(len(e["points_pixel"]), 2)


class UiWiringTests(unittest.TestCase):
    def test_ui_mentions_local_repair(self):
        panel = os.path.join(ROOT, "gui", "parameter_panel.py")
        tools = os.path.join(ROOT, "gui", "tool_panel.py")
        main = os.path.join(ROOT, "gui", "main_window.py")
        with open(panel, encoding="utf-8") as f:
            p = f.read()
        with open(tools, encoding="utf-8") as f:
            t = f.read()
        with open(main, encoding="utf-8") as f:
            m = f.read()
        self.assertIn("折线补路", p)
        self.assertIn("局部重建路网", p)
        self.assertIn("定位异常跳边", p)
        self.assertIn("折线补路", t)
        self.assertIn("_on_graph_polyline_repair", m)
        self.assertIn("_on_graph_local_rebuild", m)
        self.assertIn("_on_graph_locate_jump", m)
        self.assertIn("local_graph_repair", m)


if __name__ == "__main__":
    unittest.main()
