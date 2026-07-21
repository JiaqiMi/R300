"""Tests for layered path diagnostics (planned_segments → dense_path)."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from roadnet.path_layer_diagnostics import (
    check_edge_polyline_endpoints,
    classify_aba_source,
    expand_segment_edges_debug,
    orient_polyline_for_travel,
    run_layered_path_diagnostics,
    validate_dense_path,
    validate_virtual_node_splits,
)
from roadnet.global_planner import expand_edge_path_to_polyline, EdgeGeometryMissingError
from roadnet.task_snapping import SnappedTaskPoint, insert_virtual_nodes


def _nodes_edges_line():
    nodes = [
        {"id": 1, "x": 0.0, "y": 0.0},
        {"id": 2, "x": 100.0, "y": 0.0},
        {"id": 3, "x": 100.0, "y": 100.0},
    ]
    edges = [
        {
            "id": 10,
            "start": 1,
            "end": 2,
            "enabled": True,
            "points_pixel": [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            "polyline": [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
        },
        {
            "id": 11,
            "start": 2,
            "end": 3,
            "enabled": True,
            "points_pixel": [[100.0, 0.0], [100.0, 50.0], [100.0, 100.0]],
            "polyline": [[100.0, 0.0], [100.0, 50.0], [100.0, 100.0]],
        },
    ]
    return nodes, edges


def test_endpoint_check_forward_and_reverse():
    nodes, edges = _nodes_edges_line()
    nodes_by_id = {n["id"]: n for n in nodes}
    ok, orient, reason = check_edge_polyline_endpoints(edges[0], nodes_by_id)
    assert ok and orient == "forward" and reason == "ok"

    # reverse-stored polyline
    rev = dict(edges[0])
    rev["points_pixel"] = list(reversed(rev["points_pixel"]))
    rev["polyline"] = list(rev["points_pixel"])
    ok, orient, reason = check_edge_polyline_endpoints(rev, nodes_by_id)
    assert ok and orient == "reverse"

    bad = dict(edges[0])
    bad["points_pixel"] = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
    bad["polyline"] = bad["points_pixel"]
    ok, orient, reason = check_edge_polyline_endpoints(bad, nodes_by_id, tolerance_px=2.0)
    assert not ok and reason == "edge_polyline_endpoint_mismatch"


def test_orient_polyline_forward_reverse_and_str_ids():
    nodes, edges = _nodes_edges_line()
    nodes_by_id = {n["id"]: n for n in nodes}
    e = edges[0]

    pts, direction, err = orient_polyline_for_travel(e, 1, 2, nodes_by_id)
    assert err is None and direction == "forward"
    assert pts[0] == [0.0, 0.0] and pts[-1] == [100.0, 0.0]

    pts, direction, err = orient_polyline_for_travel(e, 2, 1, nodes_by_id)
    assert err is None and direction == "reverse"
    assert pts[0] == [100.0, 0.0] and pts[-1] == [0.0, 0.0]

    # str/int id mismatch from planning serialization
    pts, direction, err = orient_polyline_for_travel(e, "1", "2", nodes_by_id)
    assert err is None and direction == "forward"


def test_expand_rejects_missing_polyline():
    nodes, edges = _nodes_edges_line()
    edges[0] = dict(edges[0])
    edges[0]["points_pixel"] = []
    edges[0]["polyline"] = []
    with pytest.raises(EdgeGeometryMissingError):
        expand_edge_path_to_polyline([1, 2], [10], nodes, edges)


def test_expand_segment_debug_and_validate_ok():
    nodes, edges = _nodes_edges_line()
    dense_rows, edge_rows, dense_pts, fatal = expand_segment_edges_debug(
        [1, 2, 3], [10, 11], nodes, edges,
        segment_index=1, task_from_seq=1, task_to_seq=2,
        metres_per_pixel=0.5,
    )
    assert fatal is None
    assert all(r["edge_valid"] for r in edge_rows)
    assert edge_rows[0]["polyline_direction"] == "forward"
    assert len(dense_pts) >= 4
    # s_m monotonic
    s_vals = [r["s_m"] for r in dense_rows]
    assert s_vals == sorted(s_vals)
    ok, report, bad = validate_dense_path(dense_rows)
    assert ok and report["aba_count"] == 0 and not bad


def test_validate_dense_path_detects_aba_and_s_regression():
    rows = [
        {"global_index": 0, "segment_index": 1, "x_pixel": 0, "y_pixel": 0,
         "s_m": 0.0, "step_distance_m": 0.0, "edge_id": 1},
        {"global_index": 1, "segment_index": 1, "x_pixel": 10, "y_pixel": 0,
         "s_m": 5.0, "step_distance_m": 5.0, "edge_id": 1},
        {"global_index": 2, "segment_index": 1, "x_pixel": 0.1, "y_pixel": 0,
         "s_m": 10.0, "step_distance_m": 5.0, "edge_id": 1},
    ]
    ok, report, bad = validate_dense_path(rows, config=None)
    assert not ok
    assert report["aba_count"] >= 1
    assert any(b["reason"] == "aba_backtrack" for b in bad)

    rows2 = [
        {"global_index": 0, "segment_index": 1, "x_pixel": 0, "y_pixel": 0,
         "s_m": 10.0, "step_distance_m": 0.0, "edge_id": 1},
        {"global_index": 1, "segment_index": 1, "x_pixel": 10, "y_pixel": 0,
         "s_m": 5.0, "step_distance_m": 5.0, "edge_id": 1},
    ]
    ok2, report2, bad2 = validate_dense_path(rows2)
    assert not ok2
    assert report2["s_m_monotonic_valid"] is False
    assert any(b["reason"] == "from_s_m>to_s_m" for b in bad2)


def test_virtual_node_split_inherits_polyline():
    nodes, edges = _nodes_edges_line()
    sp = SnappedTaskPoint(
        seq=1,
        point_type=2,
        original_x=50.0,
        original_y=0.0,
        snapped_x=50.0,
        snapped_y=0.0,
        snap_distance=0.0,
        edge_id=10,
        node_id=None,
        virtual_node_id=-1001,
        snap_method="edge_projection",
        status="ok",
    )
    new_nodes, new_edges = insert_virtual_nodes(nodes, edges, [sp])
    split = [e for e in new_edges if e.get("source") == "task_split"]
    assert len(split) == 2
    left = next(e for e in split if e["end"] == -1001)
    right = next(e for e in split if e["start"] == -1001)
    assert len(left["points_pixel"]) >= 2
    assert len(right["points_pixel"]) >= 2
    assert left["points_pixel"][-1] == [50.0, 0.0]
    assert right["points_pixel"][0] == [50.0, 0.0]

    rows, all_ok = validate_virtual_node_splits(nodes, edges, [sp])
    assert all_ok
    assert rows and rows[0]["left_valid"] and rows[0]["right_valid"]


def test_run_layered_diagnostics_writes_artifacts():
    nodes, edges = _nodes_edges_line()
    seg = SimpleNamespace(
        from_seq=1, to_seq=2, status="ok",
        node_path=[1, 2, 3], edge_path=[10, 11],
        error=None, unexpected_task_virtual_nodes=[],
    )
    planning = SimpleNamespace(success=True, segments=[seg], task_sequence=[1, 2])
    with tempfile.TemporaryDirectory() as td:
        result = run_layered_path_diagnostics(
            planning, nodes, edges,
            snapped_task_points=[],
            output_dir=td,
            image_width=200, image_height=200,
        )
        assert result.planned_segments_valid
        assert result.dense_path_valid
        for name in (
            "planned_segments_debug.csv",
            "dense_path_debug.csv",
            "dense_path_validation_report.json",
            "dense_path_bad_segments.csv",
            "dense_path_validation_overlay.png",
            "virtual_node_split_debug.csv",
        ):
            assert os.path.isfile(os.path.join(td, name)), name
        with open(os.path.join(td, "dense_path_validation_report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["dense_path_valid"] is True


def test_classify_aba_source():
    assert classify_aba_source({"aba_count": 2}, {"aba_backtrack_count": 1}) == "dense_path"
    assert classify_aba_source({"aba_count": 0}, {"aba_backtrack_count": 1}) == "vehicle_waypoints"
    assert classify_aba_source({"aba_count": 0}, {"aba_backtrack_count": 0}) is None


def test_expand_uses_str_edge_ids_from_planning():
    nodes, edges = _nodes_edges_line()
    poly = expand_edge_path_to_polyline(["1", "2", "3"], ["10", "11"], nodes, edges)
    assert len(poly) >= 4
    assert poly[0] == [0.0, 0.0]
    assert poly[-1] == [100.0, 100.0]
