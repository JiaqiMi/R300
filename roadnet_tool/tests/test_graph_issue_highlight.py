"""Tests for graph issue highlight (display-only diagnostics)."""

from __future__ import annotations

import copy
import json
import os
import tempfile

import numpy as np

from roadnet.graph_issue_highlight import (
    GraphIssueConfig,
    detect_graph_issues,
    export_graph_issue_reports,
    render_graph_issue_overlay_png,
)


def _nodes_edges_sample():
    # Main component: 0-1-2, plus degree-1 spur, isolated, missing poly, OOB, second component
    nodes = [
        {"id": 0, "x": 50.0, "y": 50.0},
        {"id": 1, "x": 80.0, "y": 50.0},
        {"id": 2, "x": 110.0, "y": 50.0},
        {"id": 3, "x": 85.0, "y": 55.0},  # short spur tip
        {"id": 4, "x": 10.0, "y": 10.0},   # isolated
        {"id": 5, "x": 200.0, "y": 200.0}, # second component
        {"id": 6, "x": 220.0, "y": 200.0},
        {"id": 7, "x": 50.0, "y": 80.0},   # complex junction hub later
        {"id": 8, "x": 30.0, "y": 80.0},
        {"id": 9, "x": 70.0, "y": 80.0},
        {"id": 10, "x": 50.0, "y": 100.0},
        {"id": 11, "x": 50.0, "y": 60.0},
    ]
    edges = [
        {
            "id": 1, "start": 0, "end": 1, "enabled": True,
            "points_pixel": [[50, 50], [80, 50]],
        },
        {
            "id": 2, "start": 1, "end": 2, "enabled": True,
            "points_pixel": [[80, 50], [110, 50]],
        },
        {
            "id": 3, "start": 1, "end": 3, "enabled": True,
            "points_pixel": [[80, 50], [85, 55]],  # short spur
        },
        {
            "id": 4, "start": 0, "end": 2, "enabled": True,
            # missing polyline
        },
        {
            "id": 5, "start": 5, "end": 6, "enabled": True,
            "points_pixel": [[200, 200], [220, 200]],
        },
        {
            "id": 6, "start": 0, "end": 7, "enabled": True,
            "points_pixel": [[50, 50], [50, 80]],
        },
        # long straight few points (stay in main component)
        {
            "id": 7, "start": 2, "end": 10, "enabled": True,
            "points_pixel": [[110, 50], [50, 100]],
        },
        # out of bounds (does not merge components)
        {
            "id": 8, "start": 2, "end": 11, "enabled": True,
            "points_pixel": [[110, 50], [50, 60], [9999, 9999]],
        },
        # complex junction edges around node 7
        {
            "id": 9, "start": 7, "end": 8, "enabled": True,
            "points_pixel": [[50, 80], [30, 80]],
        },
        {
            "id": 10, "start": 7, "end": 9, "enabled": True,
            "points_pixel": [[50, 80], [70, 80]],
        },
        {
            "id": 11, "start": 7, "end": 10, "enabled": True,
            "points_pixel": [[50, 80], [50, 100]],
        },
        {
            "id": 12, "start": 7, "end": 11, "enabled": True,
            "points_pixel": [[50, 80], [50, 60]],
        },
    ]
    return nodes, edges


def test_detect_core_issue_types():
    nodes, edges = _nodes_edges_sample()
    before_n = copy.deepcopy(nodes)
    before_e = copy.deepcopy(edges)
    report = detect_graph_issues(
        nodes, edges,
        image_width=256, image_height=256,
        config=GraphIssueConfig(short_spur_px=10.0, long_straight_px=100.0),
    )
    assert nodes == before_n
    assert edges == before_e

    types = {i["issue_type"] for i in report["issues"]}
    assert "isolated_node" in types
    assert "degree1_endpoint" in types
    assert "short_spur_edge" in types
    assert "edge_missing_polyline" in types
    assert "edge_out_of_bounds" in types
    assert "non_main_component" in types
    assert "complex_junction" in types
    assert report["component_count"] >= 2
    assert report["serious_issue_count"] >= 1


def test_low_road_support_edge():
    nodes = [
        {"id": 0, "x": 10.0, "y": 10.0},
        {"id": 1, "x": 90.0, "y": 10.0},
    ]
    edges = [{
        "id": 1, "start": 0, "end": 1, "enabled": True,
        "points_pixel": [[10, 10], [90, 10]],
    }]
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[8:13, 0:20] = 255  # only left part is road
    report = detect_graph_issues(
        nodes, edges,
        image_width=100, image_height=40,
        road_mask=mask,
        config=GraphIssueConfig(road_support_ratio_min=0.6),
    )
    types = {i["issue_type"] for i in report["issues"]}
    assert "low_road_support_edge" in types


def test_export_reports_and_overlay():
    nodes, edges = _nodes_edges_sample()
    report = detect_graph_issues(
        nodes, edges, image_width=256, image_height=256,
    )
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_graph_issue_reports(report, tmp)
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["csv"])
        with open(paths["json"], encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["issue_count"] == report["issue_count"]
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = os.path.join(tmp, "graph_issue_overlay.png")
        render_graph_issue_overlay_png(img, nodes, edges, report, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


def test_graph_bbox_mismatch():
    nodes = [{"id": 0, "x": -500.0, "y": -500.0}, {"id": 1, "x": -400.0, "y": -500.0}]
    edges = [{
        "id": 1, "start": 0, "end": 1, "enabled": True,
        "points_pixel": [[-500, -500], [-400, -500]],
    }]
    report = detect_graph_issues(nodes, edges, image_width=100, image_height=100)
    types = {i["issue_type"] for i in report["issues"]}
    assert "graph_bbox_mismatch" in types
