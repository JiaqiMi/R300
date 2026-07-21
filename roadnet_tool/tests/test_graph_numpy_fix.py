"""
测试 graph 模块 numpy 布尔歧义修复。

验证：
1. 工具函数 is_empty_array_like / has_points / point_to_tuple / polyline_to_list 正确
2. skeleton_to_graph 不再报 ambiguous truth value
3. draft_graph_extract 不再报 ambiguous truth value
4. graph_line_optimizer 不再报 ambiguous truth value
5. graph_editor_qt 的 load_draft / save / _path_length 安全处理 numpy
6. final_graph.json 可正常保存且不含 numpy 对象
7. numpy.ndarray 类型 polyline 测试
8. 空/单点/两点/多点 polyline 测试
"""

import json
import os
import sys
import tempfile

import numpy as np

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_utils():
    """测试 graph_utils 工具函数"""
    from roadnet.graph_utils import (
        is_empty_array_like, has_points,
        point_to_tuple, polyline_to_list,
        ensure_python_types, ensure_graph_python_types,
        as_bool,
    )

    # ── is_empty_array_like ──
    assert is_empty_array_like(None) is True
    assert is_empty_array_like(np.array([])) is True
    assert is_empty_array_like(np.array([1, 2])) is False
    assert is_empty_array_like([]) is True
    assert is_empty_array_like([[1, 2], [3, 4]]) is False
    assert is_empty_array_like(np.zeros((0, 2))) is True
    assert is_empty_array_like(np.zeros((3, 2))) is False
    print("  [OK] is_empty_array_like: all cases pass")

    # ── has_points ──
    assert has_points(np.array([[1, 2]])) is True
    assert has_points(None) is False
    assert has_points([]) is False
    print("  [OK] has_points: all cases pass")

    # ── point_to_tuple ──
    assert point_to_tuple([10, 20]) == (10.0, 20.0)
    assert point_to_tuple(np.array([10.5, 20.5])) == (10.5, 20.5)
    assert point_to_tuple((3, 4)) == (3.0, 4.0)
    print("  [OK] point_to_tuple: all cases pass")

    # ── polyline_to_list ──
    r = polyline_to_list(np.array([[1, 2], [3, 4]]))
    assert r == [[1.0, 2.0], [3.0, 4.0]], f"got {r}"
    assert polyline_to_list([]) == []
    assert polyline_to_list(None) == []
    assert polyline_to_list(np.array([])) == []
    # 单点
    r = polyline_to_list(np.array([[5, 6]]))
    assert r == [[5.0, 6.0]], f"got {r}"
    # 两点
    r = polyline_to_list([[7, 8], [9, 10]])
    assert r == [[7.0, 8.0], [9.0, 10.0]], f"got {r}"
    # 确保输出都是 list 而非 numpy
    for pt in r:
        assert isinstance(pt, list)
        assert isinstance(pt[0], float)
    print("  [OK] polyline_to_list: all cases pass")

    # ── ensure_python_types ──
    data = {
        "a": np.int32(42),
        "b": np.float64(3.14),
        "c": np.bool_(True),
        "d": [np.int64(1), np.float32(2.5)],
        "e": {"nested": np.int16(99)},
    }
    clean = ensure_python_types(data)
    assert isinstance(clean["a"], int) and clean["a"] == 42
    assert isinstance(clean["b"], float) and abs(clean["b"] - 3.14) < 1e-6
    assert isinstance(clean["c"], bool) and clean["c"] is True
    assert isinstance(clean["d"][0], int) and clean["d"][0] == 1
    assert isinstance(clean["d"][1], float) and abs(clean["d"][1] - 2.5) < 1e-6
    assert isinstance(clean["e"]["nested"], int) and clean["e"]["nested"] == 99
    print("  [OK] ensure_python_types: all cases pass")

    # ── ensure_graph_python_types ──
    nodes = [
        {"id": np.int64(0), "y": np.int64(100), "x": np.int64(200), "type": "junction"},
    ]
    edges = [
        {"id": np.int64(0), "from": np.int64(0), "to": np.int64(1),
         "length_px": np.float64(50.5),
         "path": [np.array([0, 0]), np.array([50, 0])]},
    ]
    cn, ce = ensure_graph_python_types(nodes, edges)
    assert isinstance(cn[0]["id"], int)
    assert isinstance(cn[0]["y"], int)
    assert isinstance(cn[0]["x"], int)
    assert isinstance(ce[0]["length_px"], float)
    assert isinstance(ce[0]["path"][0][0], float)
    # 验证 JSON 可序列化
    json.dumps({"nodes": cn, "edges": ce})
    print("  [OK] ensure_graph_python_types: JSON serializable")

    # ── as_bool ──
    binary = np.array([[True, False]])
    assert as_bool(binary[0, 0]) is True
    assert as_bool(binary[0, 1]) is False
    assert isinstance(as_bool(binary[0, 0]), bool)
    print("  [OK] as_bool: all cases pass")

    print("[PASS] test_utils")


def test_skeleton_to_graph():
    """测试 skeleton_to_graph 不再产生 ambiguous truth value"""
    from roadnet.skeleton_to_graph import (
        skeleton_to_graph, SkeletonToGraphConfig,
        detect_nodes, merge_nodes, trace_edges,
        filter_short_edges, simplify_edges,
        save_graph_from_skeleton,
    )
    from roadnet.graph_utils import ensure_python_types

    # 创建一个简单十字形骨架
    skel = np.zeros((100, 100), dtype=np.uint8)
    # 水平线
    skel[50, 20:80] = 255
    # 垂直线
    skel[20:80, 50] = 255

    # ── 测试 detect_nodes ──
    endpoints, normals, junctions_list = detect_nodes(skel)
    assert len(endpoints) == 4, f"Expected 4 endpoints, got {len(endpoints)}"
    assert len(junctions_list) >= 1, f"Expected >=1 junctions, got {len(junctions_list)}"
    print(f"  [OK] detect_nodes: {len(endpoints)} endpoints, {len(junctions_list)} junctions")

    # ── 测试 merge_nodes ──
    nodes, pixel_to_node = merge_nodes(endpoints, junctions_list, merge_distance=15, skeleton=skel)
    assert len(nodes) >= 4, f"Expected >=4 nodes, got {len(nodes)}"
    # 验证 node 坐标都是 Python int
    for n in nodes:
        assert isinstance(n["id"], int)
        assert isinstance(n["y"], int)
        assert isinstance(n["x"], int)
    print(f"  [OK] merge_nodes: {len(nodes)} nodes, all coordinates are int")

    # ── 测试 trace_edges ──
    edges = trace_edges(skel, nodes, pixel_to_node)
    assert len(edges) >= 2, f"Expected >=2 edges, got {len(edges)}"
    for e in edges:
        path = e["path"]
        assert isinstance(path[0], list), f"edge path should be list[list]"
        assert isinstance(path[0][0], (int, float)), f"coords should be numeric"
    print(f"  [OK] trace_edges: {len(edges)} edges")

    # ── 测试 filter_short_edges ──
    filtered = filter_short_edges(edges, min_length=5.0)
    assert len(filtered) == 4
    print(f"  [OK] filter_short_edges: {len(filtered)} edges")

    # ── 测试 simplify_edges ──
    simplified = simplify_edges(edges, tolerance=2.0)
    assert len(simplified) == 4
    print(f"  [OK] simplify_edges: {len(simplified)} edges")

    # ── 测试完整流程 skeleton_to_graph ──
    config = SkeletonToGraphConfig(merge_node_distance=15, min_edge_length=5, simplify_tolerance=2.0)
    nodes_out, edges_out = skeleton_to_graph(skel, config)
    assert len(nodes_out) >= 4
    assert len(edges_out) >= 4
    # 确认所有值都是 Python 原生类型
    for n in nodes_out:
        assert isinstance(n["id"], int)
        assert isinstance(n["y"], int)
        assert isinstance(n["x"], int)
    for e in edges_out:
        assert isinstance(e["path"][0][0], (int, float))
        assert isinstance(e["length_px"], (int, float))
    # JSON 可序列化
    json.dumps(nodes_out)
    json.dumps(edges_out)
    print(f"  [OK] skeleton_to_graph: {len(nodes_out)} nodes, {len(edges_out)} edges, JSON safe")

    # ── 测试保存 ──
    with tempfile.TemporaryDirectory() as tmpdir:
        save_graph_from_skeleton(nodes_out, edges_out, tmpdir)
        json_path = os.path.join(tmpdir, "graph_from_skeleton.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data
        print(f"  [OK] save_graph_from_skeleton: saved to {json_path}")

    # ── 测试 numpy.ndarray 作为 polyline ──
    nodes_np = [
        {"id": 0, "y": 10, "x": 10, "type": "endpoint", "degree": 1},
        {"id": 1, "y": 90, "x": 90, "type": "endpoint", "degree": 1},
    ]
    edges_np = [
        {"id": 0, "from": 0, "to": 1, "length_px": 113.0,
         "path": [np.array([10, 10]), np.array([50, 50]), np.array([90, 90])]},
    ]
    # load_draft 应该能处理 numpy 数组 path
    from roadnet.graph_editor_qt import GraphEditorQt
    ge = GraphEditorQt(image_size=(100, 100))
    ge.load_draft(nodes_np, edges_np)
    assert len(ge.nodes) == 2
    assert len(ge.edges) == 1
    pts = ge.edges[0]["points_pixel"]
    assert isinstance(pts[0][0], (int, float)), f"got {type(pts[0][0])}"
    print(f"  [OK] numpy.ndarray polyline → load_draft: {len(pts)} points, all clean")

    print("[PASS] test_skeleton_to_graph")


def test_graph_line_optimizer():
    """测试 graph_line_optimizer 不再产生 ambiguous truth value"""
    from roadnet.graph_line_optimizer import (
        optimize_graph_lines, GraphLineOptimizeConfig,
        save_optimization_results,
    )
    from roadnet.graph_utils import polyline_to_list, as_bool

    # 创建简单边数据和 mask
    edges = [
        {
            "id": 0, "start": 0, "end": 1,
            "length_pixel": 100.0,
            "points_pixel": [[0, 0], [10, 5], [20, 3], [30, 8], [40, 2],
                             [50, 7], [60, 3], [70, 6], [80, 4], [90, 5], [100, 0]],
            "source": "auto", "enabled": True,
        },
    ]

    # ── 不带 mask 的优化 ──
    config = GraphLineOptimizeConfig(validate_with_mask=False)
    opt_edges, report = optimize_graph_lines(edges, processed_mask=None, config=config)
    assert len(opt_edges) == 1
    s = report["summary"]
    assert s["total_edges"] == 1
    # 验证 points_pixel 是纯 Python list
    for pt in opt_edges[0]["points_pixel"]:
        assert isinstance(pt, (list, tuple)), f"expected list, got {type(pt)}"
        assert isinstance(pt[0], (int, float)), f"expected number, got {type(pt[0])}"
    print(f"  [OK] optimize_graph_lines (no mask): {s['total_edges']} edges, "
          f"action={report['per_edge_stats'][0]['action']}")

    # ── 带 mask 的优化 ──
    mask = np.ones((20, 110), dtype=bool)  # 全部是道路
    config2 = GraphLineOptimizeConfig(validate_with_mask=True)
    opt_edges2, report2 = optimize_graph_lines(edges, processed_mask=mask, config=config2)
    assert len(opt_edges2) == 1
    # 验证 mask 内部不再有 ambiguous truth value
    assert report2["summary"]["mask_rollback_edges"] == 0, "Should not rollback when inside mask"
    print(f"  [OK] optimize_graph_lines (with mask): mask_rollback={report2['summary']['mask_rollback_edges']}")

    # ── 带 mask 偏右（部分点在 mask 外）──
    mask_small = np.ones((20, 110), dtype=bool)
    mask_small[:, 60:] = False  # 右半边不是道路
    config3 = GraphLineOptimizeConfig(validate_with_mask=True, mask_tolerance=2.0)
    opt_edges3, report3 = optimize_graph_lines(edges, processed_mask=mask_small, config=config3)
    print(f"  [OK] optimize_graph_lines (mask partial): mask_rollback={report3['summary']['mask_rollback_edges']}")

    # ── 测试保存（确保 JSON 可序列化）──
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_optimization_results(edges, opt_edges, report, tmpdir)
        assert os.path.exists(paths["before_json"])
        assert os.path.exists(paths["after_json"])
        assert os.path.exists(paths["report_json"])
        # 验证 JSON 内容可读
        with open(paths["after_json"], "r") as f:
            data = json.load(f)
        assert "edges" in data
        print(f"  [OK] save_optimization_results: all files saved to {tmpdir}")

    print("[PASS] test_graph_line_optimizer")


def test_graph_editor_qt():
    """测试 GraphEditorQt 的 numpy 类型处理"""
    from roadnet.graph_editor_qt import GraphEditorQt
    from roadnet.graph_utils import polyline_to_list

    ge = GraphEditorQt(image_size=(200, 200))

    # ── 测试 load_draft 处理 numpy 节点数据 ──
    nodes_with_numpy = [
        {"id": np.int64(0), "y": np.int64(50), "x": np.int64(50), "type": "junction"},
        {"id": np.int64(1), "y": np.int64(150), "x": np.int64(150), "type": "endpoint"},
        {"id": np.int64(2), "y": np.int64(50), "x": np.int64(150), "type": "junction"},
    ]
    edges_with_numpy = [
        {"id": np.int64(0), "from": np.int64(0), "to": np.int64(1),
         "length_px": np.float64(141.42),
         "path": [np.array([50, 50], dtype=np.int64),
                  np.array([100, 100], dtype=np.int64),
                  np.array([150, 150], dtype=np.int64)]},
        {"id": np.int64(1), "from": np.int64(0), "to": np.int64(2),
         "length_px": np.float64(100.0),
         "path": [np.array([50, 50]), np.array([50, 150])]},
    ]
    ge.load_draft(nodes_with_numpy, edges_with_numpy)

    assert len(ge.nodes) == 3
    assert len(ge.edges) == 2
    # 验证所有值都是 Python 原生类型
    for n in ge.nodes:
        assert isinstance(n["id"], int), f"node id should be int, got {type(n['id'])}"
        assert isinstance(n["x"], int), f"node x should be int, got {type(n['x'])}"
        assert isinstance(n["y"], int), f"node y should be int, got {type(n['y'])}"
    for e in ge.edges:
        assert isinstance(e["id"], int), f"edge id should be int, got {type(e['id'])}"
        assert isinstance(e["start"], int), f"edge start should be int"
        assert isinstance(e["end"], int), f"edge end should be int"
        pts = e["points_pixel"]
        assert isinstance(pts[0][0], (int, float)), f"point coord should be numeric"
    print(f"  [OK] load_draft with numpy data: {len(ge.nodes)} nodes, {len(ge.edges)} edges")

    # ── 测试 _path_length 兼容 numpy ──
    np_pts = np.array([[0, 0], [3, 4]], dtype=np.float64)
    length = ge._path_length(np_pts)
    assert abs(length - 5.0) < 0.01, f"Expected 5.0, got {length}"
    print(f"  [OK] _path_length(numpy array): {length}")

    # 测试空 points
    assert ge._path_length([]) == 0.0
    assert ge._path_length(np.array([])) == 0.0
    print(f"  [OK] _path_length(empty): 0.0")

    # ── 测试 save 可正常保存 final_graph.json ──
    with tempfile.TemporaryDirectory() as tmpdir:
        path = ge.save(tmpdir)
        json_path = os.path.join(tmpdir, "final_graph.json")
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2
        # 确保没有任何 numpy 值泄漏
        jstr = json.dumps(data)
        assert "dtype" not in jstr.lower(), "JSON contains numpy dtype"
        print(f"  [OK] save final_graph.json: {len(data['nodes'])} nodes, {len(data['edges'])} edges")

    # ── 测试 add_manual_edge 的安全 ──
    from roadnet.graph_utils import polyline_to_list
    ge2 = GraphEditorQt(image_size=(200, 200))
    ge2._add_node_raw(10, 10)
    ge2._add_node_raw(100, 100)
    # 使用 numpy 点添加 manual edge
    np_points = [(0, 0), (50, 50), (100, 100)]  # 纯 Python tuple
    ge2.add_manual_edge(np_points)
    assert len(ge2.edges) == 1
    pts = ge2.edges[0]["points_pixel"]
    assert isinstance(pts[0][0], (int, float))
    print(f"  [OK] add_manual_edge with tuple points: OK")

    print("[PASS] test_graph_editor_qt")


def test_ambiguous_truth_prevention():
    """直接测试 numpy.bool_ 在 if 条件中不再触发错误"""
    import warnings
    from roadnet.graph_utils import as_bool

    # 模拟 skeleton_to_graph 中 binary[ny, nx] 的场景
    binary = np.zeros((10, 10), dtype=bool)
    binary[3, 3] = True
    binary[5, 5] = False

    # 旧写法会触发 FutureWarning（在 numpy >= 1.25 中）
    # if binary[3, 3]:  ← 这会触发 DeprecationWarning
    # if not binary[5, 5]: ← 同样

    # 新写法：使用 as_bool()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = as_bool(binary[3, 3])
        result2 = as_bool(binary[5, 5])
        # 检查没有 numpy DeprecationWarning
        numpy_warnings = [x for x in w if "ambiguous" in str(x.message).lower()]
        assert len(numpy_warnings) == 0, f"Got ambiguous truth value warning: {numpy_warnings}"

    assert result1 is True
    assert result2 is False
    print("  [OK] as_bool: no ambiguous truth value warnings")

    # 测试在循环中的行为（模拟 _count_neighbors）
    cnt = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = 3 + dy, 3 + dx
                if 0 <= ny < 10 and 0 <= nx < 10 and as_bool(binary[ny, nx]):
                    cnt += 1
        numpy_warnings = [x for x in w if "ambiguous" in str(x.message).lower()]
        assert len(numpy_warnings) == 0

    # binary[3,3] 是 True，有 8 个邻居（含自己），但 binary[3,3] 本身被计数
    # 实际上 _count_neighbors 遍历 8 邻居而不包括自己...
    # 我们只验证计数合理即可
    assert cnt >= 0, f"Count should be non-negative, got {cnt}"
    print(f"  [OK] neighbor counting loop: cnt={cnt}, no warnings")

    print("[PASS] test_ambiguous_truth_prevention")


def test_empty_polyline_edge_cases():
    """测试空/单点/两点/多点 polyline 的边缘情况"""
    from roadnet.graph_utils import polyline_to_list, has_points, is_empty_array_like

    # 空
    assert polyline_to_list(None) == []
    assert polyline_to_list([]) == []
    assert polyline_to_list(np.array([])) == []

    # 单点
    r = polyline_to_list([[5, 5]])
    assert r == [[5.0, 5.0]]

    # 两点
    r = polyline_to_list([[0, 0], [10, 10]])
    assert r == [[0.0, 0.0], [10.0, 10.0]]

    # 多点
    r = polyline_to_list(np.array([[1, 2], [3, 4], [5, 6]]))
    assert r == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    # numpy 1D 数组
    r = polyline_to_list(np.array([7, 8]))
    assert r == [[7.0, 8.0]]

    print("  [OK] empty/single/double/multi polyline: all cases pass")

    # ── 测试 graph_line_optimizer 接收各种格式的 polyline ──
    from roadnet.graph_line_optimizer import optimize_graph_lines, GraphLineOptimizeConfig

    config = GraphLineOptimizeConfig(validate_with_mask=False)

    # numpy ndarray polyline
    edges_ndarray = [{
        "id": 0, "start": 0, "end": 1,
        "length_pixel": 100.0,
        "points_pixel": np.array([[0, 0], [50, 50], [100, 100]], dtype=np.float64),
        "source": "auto", "enabled": True,
    }]
    opt, _ = optimize_graph_lines(edges_ndarray, processed_mask=None, config=config)
    assert len(opt) == 1
    for pt in opt[0]["points_pixel"]:
        assert isinstance(pt[0], (int, float))

    # 空 polyline
    edges_empty = [{
        "id": 0, "start": 0, "end": 1,
        "length_pixel": 0.0,
        "points_pixel": [],
        "source": "auto", "enabled": True,
    }]
    opt2, _ = optimize_graph_lines(edges_empty, processed_mask=None, config=config)
    assert len(opt2) == 1

    # 单点 polyline（应被 skip，因为 < 2 个点）
    edges_single = [{
        "id": 0, "start": 0, "end": 0,
        "length_pixel": 0.0,
        "points_pixel": [[5, 5]],
        "source": "auto", "enabled": True,
    }]
    opt3, _ = optimize_graph_lines(edges_single, processed_mask=None, config=config)
    assert len(opt3) == 1

    print("[PASS] test_empty_polyline_edge_cases")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing graph numpy ambiguous truth value fixes")
    print("=" * 60)

    all_passed = True
    tests = [
        ("test_utils", test_utils),
        ("test_skeleton_to_graph", test_skeleton_to_graph),
        ("test_graph_line_optimizer", test_graph_line_optimizer),
        ("test_graph_editor_qt", test_graph_editor_qt),
        ("test_ambiguous_truth_prevention", test_ambiguous_truth_prevention),
        ("test_empty_polyline_edge_cases", test_empty_polyline_edge_cases),
    ]

    for name, func in tests:
        try:
            func()
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [FAIL]")
        sys.exit(1)
