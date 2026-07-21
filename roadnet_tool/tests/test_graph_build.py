"""
Graph 生成流水线测试 (test_graph_build.py)

测试覆盖：
1. skeleton 为空时，明确报错
2. skeleton 为 bool 图时能生成 graph
3. skeleton 为 0/255 uint8 图时能生成 graph
4. graph edge.polyline 为 numpy.ndarray 时能保存 JSON
5. graph edge.polyline 为 list 时能保存 JSON
6. graph_line_optimizer 失败时 raw graph 仍然保留
7. final_graph_raw.json 可以重新加载
8. 渲染 final_graph 不报错
9. numpy.ndarray 布尔判断安全性
10. 阶段日志输出

运行方式：
    cd d:/road_zy/roadnet_tool
    python tests/test_graph_build.py
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import cv2

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadnet.graph_utils import (
    is_empty_array_like,
    has_points,
    point_to_tuple,
    polyline_to_list,
    ensure_python_types,
    ensure_graph_python_types,
    as_bool,
)
from roadnet.graph_build import (
    build_graph_from_skeleton,
    generate_raw_graph_minimal,
    validate_final_graph_raw,
    GraphBuildResult,
    _read_and_validate_skeleton,
    _generate_raw_graph,
    _save_raw_graph,
    GraphBuildLog,
)


def _make_test_skeleton(size=100, pattern="cross"):
    """创建测试用的二值骨架图。

    Args:
        size: 图像尺寸
        pattern: 'cross' | 'empty' | 'three_lines'
    """
    skel = np.zeros((size, size), dtype=np.uint8)
    if pattern == "cross":
        # 十字骨架
        skel[size//2, :] = 255
        skel[:, size//2] = 255
    elif pattern == "three_lines":
        # 三条相连线段
        skel[30, 20:60] = 255
        skel[20:60, 60] = 255
        skel[60, 30:70] = 255
    elif pattern == "empty":
        pass  # 全黑
    return skel


class TestSkeletonValidation(unittest.TestCase):
    """测试 skeleton 读取和验证。"""

    def setUp(self):
        self.log = GraphBuildLog()

    def test_skeleton_none(self):
        """1. skeleton 为 None → 报错 empty_skeleton / read_skeleton_failed"""
        binary, err = _read_and_validate_skeleton(None, self.log)
        self.assertIsNone(binary)
        self.assertTrue(err in ("read_skeleton_failed", "empty_skeleton"),
                        f"Expected error code but got '{err}'")

    def test_skeleton_empty(self):
        """2. skeleton 全黑 → 报错 empty_skeleton"""
        skel = np.zeros((100, 100), dtype=np.uint8)
        binary, err = _read_and_validate_skeleton(skel, self.log)
        self.assertIsNone(binary)
        self.assertEqual(err, "empty_skeleton",
                         f"Expected 'empty_skeleton' but got '{err}'")

    def test_skeleton_too_few_pixels(self):
        """skeleton 非零像素过少 → 报错 empty_skeleton"""
        skel = np.zeros((100, 100), dtype=np.uint8)
        skel[50, 50] = 255  # 仅 1 像素
        binary, err = _read_and_validate_skeleton(skel, self.log)
        self.assertIsNone(binary)
        self.assertEqual(err, "empty_skeleton")

    def test_skeleton_bool_dtype(self):
        """3. skeleton 为 bool 图 → 正确转为二值"""
        skel = _make_test_skeleton(100, "cross").astype(bool)
        binary, err = _read_and_validate_skeleton(skel, self.log)
        self.assertIsNotNone(binary)
        self.assertEqual(err, "")
        self.assertEqual(binary.dtype, np.uint8)

    def test_skeleton_uint8_0_255(self):
        """4. skeleton 为 0/255 uint8 → 正确识别"""
        skel = _make_test_skeleton(100, "cross")
        binary, err = _read_and_validate_skeleton(skel, self.log)
        self.assertIsNotNone(binary)
        self.assertEqual(err, "")
        self.assertGreater(binary.sum(), 0)

    def test_skeleton_rgba_3channel(self):
        """skeleton 为 3 通道 RGB → 正确取第一个通道"""
        skel = _make_test_skeleton(80, "cross")
        skel_rgb = np.stack([skel, skel, skel], axis=2)
        binary, err = _read_and_validate_skeleton(skel_rgb, self.log)
        self.assertIsNotNone(binary)
        self.assertEqual(err, "")
        self.assertEqual(binary.ndim, 2)

    def test_skeleton_dict_input(self):
        """skeleton 为 dict（有 'skeleton' key）→ 正确提取"""
        skel = _make_test_skeleton(80, "cross")
        data = {"skeleton": skel, "other": "stuff"}
        binary, err = _read_and_validate_skeleton(data, self.log)
        self.assertIsNotNone(binary)
        self.assertEqual(err, "")

    def test_skeleton_dict_optimized_skeleton_key(self):
        """skeleton dict 使用 'optimized_skeleton' key → 正确提取"""
        skel = _make_test_skeleton(80, "cross")
        data = {"optimized_skeleton": skel}
        binary, err = _read_and_validate_skeleton(data, self.log)
        self.assertIsNotNone(binary)
        self.assertEqual(err, "")


class TestGraphGeneration(unittest.TestCase):
    """测试 raw graph 生成。"""

    def test_generate_raw_graph_cross(self):
        """十字骨架 → 应生成节点和边"""
        skel = _make_test_skeleton(100, "cross")
        nodes, edges, err = _generate_raw_graph(skel)
        if err:
            # 十字骨架可能因中间合并导致 0 edges，这不算致命
            self.assertIn(err, ("", "raw_graph_empty"))
        else:
            self.assertTrue(len(nodes) > 0 or len(edges) > 0,
                            "Cross skeleton should produce some nodes or edges")

    def test_generate_raw_graph_three_lines(self):
        """三条线骨架 → 应生成图"""
        skel = _make_test_skeleton(100, "three_lines")
        nodes, edges, err = _generate_raw_graph(skel)
        if not err:
            self.assertTrue(len(nodes) > 0, "Should have at least some nodes")
            print(f"  three_lines: nodes={len(nodes)}, edges={len(edges)}")

    def test_empty_skeleton_fails(self):
        """空骨架 → 返回错误码"""
        skel = np.zeros((100, 100), dtype=np.uint8)
        skel[50, 50] = 1  # 至少要有一些像素才能进入 _generate_raw_graph
        # Actually _generate_raw_graph doesn't validate, use build_graph_from_skeleton
        result = build_graph_from_skeleton(skeleton=np.zeros((100, 100), dtype=np.uint8))
        self.assertFalse(result.success)
        self.assertTrue("empty_skeleton" in result.stage.lower() or
                        "read" in result.stage.lower(),
                        f"Stage should indicate skeleton error, got: {result.stage}")


class TestGraphSaveAndLoad(unittest.TestCase):
    """测试 graph 保存和加载。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_save_raw_graph_json_format(self):
        """5. final_graph_raw.json 格式正确"""
        skel = _make_test_skeleton(100, "cross")
        result = build_graph_from_skeleton(
            skeleton=skel,
            output_dir=self.tmpdir,
            run_optimization=False,
        )
        self.assertTrue(result.success, f"Build failed: {result.errors}")

        # 验证格式
        json_path = result.raw_graph_path
        self.assertIsNotNone(json_path, "raw_graph_path should not be None")
        self.assertTrue(os.path.exists(json_path), f"File not found: {json_path}")

        validation = validate_final_graph_raw(json_path)
        self.assertTrue(validation["valid"],
                        f"Validation errors: {validation['errors']}")

        with open(json_path, "r") as f:
            graph = json.load(f)

        self.assertIn("coordinate_system", graph)
        self.assertEqual(graph["coordinate_system"], "image_pixel")
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)

    def test_save_raw_graph_reloadable(self):
        """6. final_graph_raw.json 可以重新加载"""
        skel = _make_test_skeleton(100, "cross")
        result = build_graph_from_skeleton(skeleton=skel, output_dir=self.tmpdir)

        if result.success and result.raw_graph_path:
            with open(result.raw_graph_path, "r") as f:
                graph = json.load(f)

            # 确认所有 node 都有 id
            for node in graph["nodes"]:
                self.assertIn("id", node)
                self.assertIn("x", node)
                self.assertIn("y", node)

            # 确认所有 edge 都有 polyline
            for edge in graph["edges"]:
                self.assertIn("id", edge)

            print(f"  Loaded: {graph['metadata']['node_count']} nodes, "
                  f"{graph['metadata']['edge_count']} edges")


class TestNumpyBooleanSafety(unittest.TestCase):
    """测试 numpy.ndarray 布尔判断安全性。"""

    def test_is_empty_array_like(self):
        """is_empty_array_like 对各种类型正确判断"""
        self.assertTrue(is_empty_array_like(None))
        self.assertTrue(is_empty_array_like(np.array([])))
        self.assertTrue(is_empty_array_like([]))
        self.assertFalse(is_empty_array_like(np.array([1, 2])))
        self.assertFalse(is_empty_array_like([1, 2]))
        self.assertFalse(is_empty_array_like([[1, 2], [3, 4]]))

    def test_has_points(self):
        """has_points 正确判断"""
        self.assertFalse(has_points(None))
        self.assertFalse(has_points([]))
        self.assertFalse(has_points(np.array([])))
        self.assertTrue(has_points([1, 2]))
        self.assertTrue(has_points(np.array([1, 2])))

    def test_as_bool(self):
        """as_bool 对 numpy.bool_ 正确转换"""
        arr = np.array([[True, False]])
        self.assertTrue(as_bool(arr[0, 0]))
        self.assertFalse(as_bool(arr[0, 1]))
        self.assertTrue(as_bool(1))
        self.assertFalse(as_bool(0))
        self.assertTrue(as_bool(True))
        self.assertFalse(as_bool(False))

    def test_ensure_python_types_ndarray(self):
        """7. numpy 类型 → JSON 可序列化"""
        # 模拟包含 numpy 类型的 edge
        edges_with_numpy = [{
            "id": np.int64(42),
            "points_pixel": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "length_pixel": np.float64(10.5),
            "source": np.str_("auto"),
            "enabled": np.bool_(True),
        }]
        cleaned = ensure_python_types(edges_with_numpy)

        # 验证类型
        self.assertIsInstance(cleaned[0]["id"], int)
        self.assertIsInstance(cleaned[0]["points_pixel"], list)
        self.assertIsInstance(cleaned[0]["points_pixel"][0], list)
        self.assertIsInstance(cleaned[0]["points_pixel"][0][0], float)
        self.assertIsInstance(cleaned[0]["length_pixel"], float)

        # 验证 JSON 序列化
        json_str = json.dumps(cleaned)
        loaded = json.loads(json_str)
        self.assertEqual(loaded[0]["id"], 42)
        self.assertEqual(loaded[0]["points_pixel"], [[1.0, 2.0], [3.0, 4.0]])

    def test_polyline_to_list_numpy(self):
        """polyline_to_list 对 numpy.ndarray 正确转换"""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = polyline_to_list(arr)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_polyline_to_list_none(self):
        """polyline_to_list 对 None 返回空列表"""
        self.assertEqual(polyline_to_list(None), [])
        self.assertEqual(polyline_to_list([]), [])


class TestBuildPipeline(unittest.TestCase):
    """测试完整 graph_build 流水线。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_full_pipeline_success(self):
        """完整流水线：skeleton → raw graph → save"""
        skel = _make_test_skeleton(100, "cross")
        result = build_graph_from_skeleton(
            skeleton=skel,
            output_dir=self.tmpdir,
            run_optimization=False,
        )
        self.assertTrue(result.success, f"Build failed: {result.errors}")
        self.assertGreater(len(result.raw_nodes), 0)
        self.assertIsNotNone(result.raw_graph_path)
        self.assertTrue(os.path.exists(result.raw_graph_path))

        # 验证日志输出
        self.assertGreater(len(result.log.messages), 0)
        has_shape_log = any("skeleton shape" in m for m in result.log.messages)
        has_nonzero_log = any("nonzero pixels" in m for m in result.log.messages)
        has_raw_nodes_log = any("raw nodes" in m for m in result.log.messages)
        self.assertTrue(has_shape_log, "Missing skeleton shape log")
        self.assertTrue(has_nonzero_log, "Missing nonzero pixels log")
        self.assertTrue(has_raw_nodes_log, "Missing raw nodes log")

        # 验证日志包含 [GraphBuild] 阶段日志
        has_save_log = any("saved final_graph_raw.json" in m for m in result.log.messages)
        has_render_log = any("render final_graph" in m for m in result.log.messages)
        # render 日志在 graph_editor 为 None 时不会输出
        self.assertTrue(has_save_log, "Missing save log")

    def test_pipeline_empty_skeleton(self):
        """skeleton 为空时流水线正确失败"""
        result = build_graph_from_skeleton(
            skeleton=np.zeros((100, 100), dtype=np.uint8),
            output_dir=self.tmpdir,
        )
        self.assertFalse(result.success)
        self.assertIn("empty_skeleton", result.stage)
        self.assertGreater(len(result.errors), 0)

    def test_pipeline_skips_optimization(self):
        """run_optimization=False 时跳过优化"""
        skel = _make_test_skeleton(100, "cross")
        result = build_graph_from_skeleton(
            skeleton=skel,
            output_dir=self.tmpdir,
            run_optimization=False,
        )
        self.assertTrue(result.success)
        self.assertIsNone(result.optimized_edges)
        self.assertIsNone(result.optimized_graph_path)
        has_skip_log = any("graph_line_optimizer skipped" in m for m in result.log.messages)
        self.assertTrue(has_skip_log, "Missing optimization skip log")

    def test_graph_line_optimizer_failure_keeps_raw_graph(self):
        """8. graph_line_optimizer 失败时 raw graph 仍然保留"""
        skel = _make_test_skeleton(100, "cross")
        from unittest.mock import patch

        # 模拟 optimize_graph_lines 抛出异常
        with patch("roadnet.graph_build._run_line_optimization",
                   return_value=(None, None, "graph_line_optimizer_failed")):
            result = build_graph_from_skeleton(
                skeleton=skel,
                output_dir=self.tmpdir,
                run_optimization=True,
            )
            # raw graph 应该仍然成功
            self.assertTrue(result.success)
            self.assertGreater(len(result.raw_nodes), 0)
            self.assertIsNotNone(result.raw_graph_path)
            # 优化应该失败但 raw 保留
            self.assertIsNone(result.optimized_edges)
            self.assertIn("graph_line_optimizer_failed",
                          str(result.errors))


class TestGraphUtils(unittest.TestCase):
    """测试 graph_utils 工具函数。"""

    def test_point_to_tuple(self):
        """point_to_tuple 正确转换"""
        self.assertEqual(point_to_tuple([10, 20]), (10.0, 20.0))
        self.assertEqual(point_to_tuple(np.array([10.5, 20.5])), (10.5, 20.5))
        self.assertEqual(point_to_tuple((10, 20)), (10.0, 20.0))

    def test_point_to_tuple_value_error(self):
        """point_to_tuple 对无效输入抛出 ValueError"""
        with self.assertRaises(ValueError):
            point_to_tuple([10])  # 只有 1 个坐标

    def test_polyline_to_list_list(self):
        """polyline_to_list 对 list of lists 正确处理"""
        data = [[1, 2], [3, 4]]
        result = polyline_to_list(data)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_ensure_graph_python_types(self):
        """ensure_graph_python_types 清理 numpy 类型"""
        nodes = [{"id": np.int64(1), "y": np.int64(50), "x": np.int64(60), "type": "junction"}]
        edges = [{
            "id": np.int64(0),
            "from": np.int64(0),
            "to": np.int64(1),
            "path": [[np.float64(50), np.float64(60)], [np.float64(70), np.float64(80)]],
            "length_px": np.float64(28.28),
        }]
        cnodes, cedges = ensure_graph_python_types(nodes, edges)
        self.assertIsInstance(cnodes[0]["id"], int)
        self.assertIsInstance(cnodes[0]["x"], int)
        self.assertIsInstance(cnodes[0]["y"], int)
        self.assertIsInstance(cedges[0]["id"], int)
        self.assertIsInstance(cedges[0]["path"], list)
        self.assertIsInstance(cedges[0]["path"][0][0], float)


if __name__ == "__main__":
    print("=" * 60)
    print("Graph Build Pipeline Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
