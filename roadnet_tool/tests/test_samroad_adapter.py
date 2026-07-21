"""
SAM-Road 适配模块单元测试

测试内容：
1. detect SAM-Road output（文件探测）
2. load draft_graph（draft_graph.json 解析）
3. load mask（road_mask_raw.png 加载）
4. load skeleton（文件不存在时的容错）
5. validate image size（尺寸匹配验证）
6. graph node/edge 数量正确
7. path 坐标格式没有 x/y 反转
8. 文件名别名兼容性
9. load_graph_for_draft vs load_graph_only 格式差异
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from roadnet.samroad_adapter import (
    # 公共 API
    load_samroad_output,
    validate_samroad_output,
    detect_samroad_outputs,
    find_samroad_dirs,
    load_mask_only,
    load_mask_clean_only,
    load_skeleton_from_file,
    load_graph_only,
    load_graph_for_draft,
    repair_edge_endpoints,
    # 内部
    _resolve_filename,
    _convert_nodes,
    _convert_edges,
    SAMRoadOutput,
    FILENAME_ALIASES,
)


# ===================================================================
# 测试辅助
# ===================================================================

def _make_temp_dir(files: dict) -> str:
    """创建临时目录并写入文件。files: {filename: content}"""
    tmp = tempfile.mkdtemp(prefix="samroad_test_")
    for fname, content in files.items():
        fpath = os.path.join(tmp, fname)
        if isinstance(content, np.ndarray):
            cv2.imwrite(fpath, content)
        elif isinstance(content, (dict, list)):
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(content, f)
        else:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(str(content))
    return tmp


def _make_draft_graph(n_nodes: int = 10, n_edges: int = 15, img_w: int = 1024, img_h: int = 768):
    """生成一个合法的 draft_graph.json 字典。

    节点坐标在图像范围内随机分布。
    path 使用标准格式 [[y, x], ...]。
    """
    rng = np.random.RandomState(42)
    nodes = []
    for i in range(n_nodes):
        nodes.append({
            "id": i,
            "x": int(rng.randint(10, img_w - 10)),
            "y": int(rng.randint(10, img_h - 10)),
            "type": rng.choice(["endpoint", "junction", "normal"]),
            "degree": 0,
        })

    edges = []
    for i in range(n_edges):
        frm = rng.randint(0, n_nodes - 1)
        to = rng.randint(0, n_nodes - 1)
        if frm == to:
            to = (frm + 1) % n_nodes
        n_pts = rng.randint(2, 8)
        # 生成 path: [[y, x], ...] — 注意 y 在前！
        start_y, start_x = nodes[frm]["y"], nodes[frm]["x"]
        end_y, end_x = nodes[to]["y"], nodes[to]["x"]
        path = []
        for j in range(n_pts):
            t = j / (n_pts - 1)
            py = int(start_y + (end_y - start_y) * t)
            px = int(start_x + (end_x - start_x) * t)
            path.append([py, px])  # [y, x] 格式

        edges.append({
            "id": i,
            "from": frm,
            "to": to,
            "length_px": round(float(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)), 2),
            "path": path,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "image_size": {"width": img_w, "height": img_h},
        },
    }


# ===================================================================
# 测试类
# ===================================================================

class TestDetectSAMRoadOutputs(unittest.TestCase):
    """测试文件探测功能"""

    def test_empty_dir(self):
        tmp = tempfile.mkdtemp(prefix="samroad_empty_")
        result = detect_samroad_outputs(tmp)
        self.assertFalse(result["is_samroad"])
        os.rmdir(tmp)

    def test_dir_with_mask_only(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        tmp = _make_temp_dir({"road_mask_raw.png": mask})
        try:
            result = detect_samroad_outputs(tmp)
            self.assertTrue(result["is_samroad"])
            self.assertTrue(result["has_mask_raw"])
            self.assertFalse(result["has_mask_clean"])
            self.assertFalse(result["has_graph"])
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_dir_with_graph_only(self):
        graph = _make_draft_graph(5, 5)
        tmp = _make_temp_dir({"draft_graph.json": graph})
        try:
            result = detect_samroad_outputs(tmp)
            self.assertTrue(result["is_samroad"])
            self.assertTrue(result["has_graph"])
            self.assertFalse(result["has_mask_raw"])
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_dir_with_all_files(self):
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        skeleton = np.eye(64, dtype=np.uint8) * 255
        graph = _make_draft_graph(3, 4, img_w=64, img_h=64)
        tmp = _make_temp_dir({
            "road_mask_raw.png": mask,
            "road_mask.png": mask,
            "road_mask_samroad_score.png": mask,
            "keypoint_mask_samroad_score.png": mask,
            "skeleton.png": skeleton,
            "draft_graph.json": graph,
            "draft_graph_overlay.png": mask,
        })
        try:
            result = detect_samroad_outputs(tmp)
            self.assertTrue(result["is_samroad"])
            self.assertTrue(result["has_mask_raw"])
            self.assertTrue(result["has_mask_clean"])
            self.assertTrue(result["has_mask_score"])
            self.assertTrue(result["has_keypoint"])
            self.assertTrue(result["has_skeleton"])
            self.assertTrue(result["has_graph"])
            self.assertTrue(result["has_overlay"])
            self.assertEqual(len(result.get("found_files", [])), 7)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_skeleton_detection(self):
        """测试 skeleton.png 和 road_skeleton.png 的检测"""
        skeleton = np.eye(32, dtype=np.uint8) * 255
        # 只有 skeleton.png
        tmp = _make_temp_dir({"skeleton.png": skeleton, "road_mask_raw.png": np.zeros((32, 32), dtype=np.uint8)})
        try:
            result = detect_samroad_outputs(tmp)
            self.assertTrue(result["has_skeleton"])
            self.assertFalse(result.get("has_skeleton_road", False))
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

        # 只有 road_skeleton.png
        tmp = _make_temp_dir({"road_skeleton.png": skeleton, "road_mask_raw.png": np.zeros((32, 32), dtype=np.uint8)})
        try:
            result = detect_samroad_outputs(tmp)
            self.assertTrue(result["has_skeleton"])
            self.assertTrue(result.get("has_skeleton_road", False))
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


class TestLoadMask(unittest.TestCase):
    """测试 Mask 加载"""

    def test_load_mask_raw(self):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        tmp = _make_temp_dir({"road_mask_raw.png": mask})
        try:
            loaded = load_mask_only(tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (100, 100))
            self.assertEqual(loaded.dtype, np.uint8)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_load_mask_clean_preferred(self):
        """测试优先加载 road_mask.png"""
        mask_raw = np.ones((64, 64), dtype=np.uint8) * 128
        mask_clean = np.ones((64, 64), dtype=np.uint8) * 200
        tmp = _make_temp_dir({
            "road_mask_raw.png": mask_raw,
            "road_mask.png": mask_clean,
        })
        try:
            # load_mask_only 应优先返回 road_mask.png
            loaded = load_mask_only(tmp)
            self.assertIsNotNone(loaded)
            # 应返回 mask_clean（200），而不是 mask_raw（128）
            self.assertEqual(loaded[0, 0], 200)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_load_mask_clean_only(self):
        mask_clean = np.ones((50, 50), dtype=np.uint8) * 180
        tmp = _make_temp_dir({"road_mask.png": mask_clean})
        try:
            loaded = load_mask_clean_only(tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded[0, 0], 180)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_mask_not_found(self):
        tmp = tempfile.mkdtemp(prefix="samroad_nomask_")
        loaded = load_mask_only(tmp)
        self.assertIsNone(loaded)
        os.rmdir(tmp)


class TestLoadSkeleton(unittest.TestCase):
    """测试 Skeleton 加载"""

    def test_load_skeleton_png(self):
        skeleton = np.eye(64, dtype=np.uint8) * 255
        tmp = _make_temp_dir({"skeleton.png": skeleton})
        try:
            loaded = load_skeleton_from_file(tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (64, 64))
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_load_road_skeleton_png(self):
        skeleton = np.ones((32, 32), dtype=np.uint8) * 255
        tmp = _make_temp_dir({"road_skeleton.png": skeleton})
        try:
            loaded = load_skeleton_from_file(tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (32, 32))
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_road_skeleton_preferred_over_skeleton(self):
        """road_skeleton.png 优先级高于 skeleton.png"""
        sk1 = np.ones((32, 32), dtype=np.uint8) * 100   # road_skeleton
        sk2 = np.ones((32, 32), dtype=np.uint8) * 200   # skeleton
        tmp = _make_temp_dir({"road_skeleton.png": sk1, "skeleton.png": sk2})
        try:
            loaded = load_skeleton_from_file(tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded[0, 0], 100)  # 应返回 road_skeleton（100）
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_skeleton_not_found(self):
        tmp = tempfile.mkdtemp(prefix="samroad_nosk_")
        loaded = load_skeleton_from_file(tmp)
        self.assertIsNone(loaded)
        os.rmdir(tmp)


class TestLoadGraph(unittest.TestCase):
    """测试 Graph 加载"""

    def test_load_draft_graph(self):
        graph = _make_draft_graph(10, 15, img_w=1024, img_h=768)
        tmp = _make_temp_dir({"draft_graph.json": graph})
        try:
            # 测试 load_graph_for_draft（返回原始 draft 格式）
            nodes_raw, edges_raw = load_graph_for_draft(tmp)
            self.assertEqual(len(nodes_raw), 10)
            self.assertEqual(len(edges_raw), 15)

            # 检查原始格式特征
            self.assertIn("x", nodes_raw[0])
            self.assertIn("y", nodes_raw[0])
            self.assertIn("from", edges_raw[0])
            self.assertIn("to", edges_raw[0])
            self.assertIn("path", edges_raw[0])

            # 检查 path 格式是 [[y, x], ...]
            path_pt = edges_raw[0]["path"][0]
            self.assertEqual(len(path_pt), 2)
            # path 的第一个元素应该是 y（行坐标），第二个是 x（列坐标）
            # 注意：这里无法绝对区分 x/y，但我们知道生成时用了 [y, x]
            # 至少坐标值应在图像范围内
            self.assertGreaterEqual(path_pt[0], 0)  # y
            self.assertGreaterEqual(path_pt[1], 0)  # x
            self.assertLess(path_pt[0], 768)   # y < height
            self.assertLess(path_pt[1], 1024)   # x < width

            # 测试 load_graph_only（返回内部转换格式）
            nodes_int, edges_int = load_graph_only(tmp)
            self.assertEqual(len(nodes_int), 10)
            self.assertEqual(len(edges_int), 15)

            # 检查内部格式：应包含 source 字段
            self.assertEqual(nodes_int[0]["source"], "auto")
            self.assertEqual(edges_int[0]["source"], "auto")

            # 检查 points_pixel 格式是 [[x, y], ...]
            pts = edges_int[0]["points_pixel"]
            self.assertGreater(len(pts), 0)
            pt = pts[0]
            self.assertEqual(len(pt), 2)
            # 内部格式中 [x, y] — x 在 [0, width), y 在 [0, height)
            self.assertGreaterEqual(pt[0], 0)
            self.assertGreaterEqual(pt[1], 0)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_graph_not_found(self):
        tmp = tempfile.mkdtemp(prefix="samroad_nograph_")
        nodes, edges = load_graph_for_draft(tmp)
        self.assertEqual(nodes, [])
        self.assertEqual(edges, [])
        os.rmdir(tmp)


class TestPathCoordinateFormat(unittest.TestCase):
    """测试 path 坐标格式：确保没有 x/y 反转"""

    def test_path_to_points_pixel_conversion(self):
        """验证 _convert_edges 正确地将 [y,x] 转为 [x,y]"""
        # 构造原始 draft edge，path 使用 [[y, x], ...] 格式
        raw_edges = [{
            "id": 0,
            "from": 0,
            "to": 1,
            "length_px": 100.0,
            "path": [
                [10, 20],   # (y=10, x=20)
                [30, 80],   # (y=30, x=80)
                [60, 120],  # (y=60, x=120)
            ],
        }]

        converted = _convert_edges(raw_edges)
        self.assertEqual(len(converted), 1)

        pts = converted[0]["points_pixel"]
        # 应转换为 [[x, y], ...]
        self.assertEqual(pts[0], [20, 10])   # [x=20, y=10]
        self.assertEqual(pts[1], [80, 30])   # [x=80, y=30]
        self.assertEqual(pts[2], [120, 60])  # [x=120, y=60]

    def test_roundtrip_draft_format(self):
        """测试 draft 格式 → 内部格式 互不干扰"""
        graph = _make_draft_graph(20, 25, img_w=800, img_h=600)

        # 保存到临时文件
        tmp = _make_temp_dir({"draft_graph.json": graph})
        try:
            # load_graph_for_draft 返回原始 draft 格式
            nodes_raw, edges_raw = load_graph_for_draft(tmp)
            # load_graph_only 返回内部格式
            nodes_int, edges_int = load_graph_only(tmp)

            # 数量应一致
            self.assertEqual(len(nodes_raw), len(nodes_int))
            self.assertEqual(len(edges_raw), len(edges_int))

            # 原始格式节点有 x, y 直给
            for i, n in enumerate(nodes_raw):
                self.assertIn("x", n)
                self.assertIn("y", n)

            # 检查内部格式边的 start/end 与原始的 from/to 对应
            for i in range(len(edges_raw)):
                self.assertEqual(edges_int[i]["start"], edges_raw[i]["from"])
                self.assertEqual(edges_int[i]["end"], edges_raw[i]["to"])
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


class TestLoadSammroadOutputFull(unittest.TestCase):
    """测试完整加载流程"""

    def test_full_load_with_mask_and_graph(self):
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        graph = _make_draft_graph(5, 8, img_w=64, img_h=64)
        tmp = _make_temp_dir({
            "road_mask_raw.png": mask,
            "draft_graph.json": graph,
        })
        try:
            output = load_samroad_output(tmp)
            self.assertTrue(output.is_valid)
            self.assertTrue(output.has_mask)
            self.assertFalse(output.has_mask_clean)  # 没有 road_mask.png
            self.assertFalse(output.has_skeleton)    # 没有 skeleton.png
            self.assertTrue(output.has_graph)
            self.assertEqual(output.node_count, 5)
            self.assertEqual(output.edge_count, 8)
            self.assertEqual(len(output.nodes), 5)
            self.assertEqual(len(output.edges), 8)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_full_load_with_skeleton(self):
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        skeleton = np.eye(32, dtype=np.uint8) * 255
        graph = _make_draft_graph(3, 4, img_w=32, img_h=32)
        tmp = _make_temp_dir({
            "road_mask_raw.png": mask,
            "road_skeleton.png": skeleton,
            "draft_graph.json": graph,
        })
        try:
            output = load_samroad_output(tmp)
            self.assertTrue(output.is_valid)
            self.assertTrue(output.has_mask)
            self.assertTrue(output.has_skeleton)
            self.assertTrue(output.has_graph)
            self.assertEqual(output.skeleton.shape, (32, 32))
            # 找到的文件列表应包含 3 个
            self.assertGreaterEqual(len(output.found_files), 3)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_load_with_missing_files(self):
        """测试只有 graph 没有 mask 的情况"""
        graph = _make_draft_graph(3, 4)
        tmp = _make_temp_dir({"draft_graph.json": graph})
        try:
            output = load_samroad_output(tmp)
            self.assertTrue(output.is_valid)
            self.assertFalse(output.has_mask)
            self.assertTrue(output.has_graph)
            # 警告中应提到缺少 mask 和 skeleton
            warnings_text = " ".join(output.warnings)
            self.assertIn("road_mask_raw", warnings_text)
            self.assertIn("skeleton", warnings_text)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_load_empty_dir(self):
        """测试目录不存在时的错误处理"""
        tmp = tempfile.mkdtemp(prefix="samroad_empty_")
        try:
            # 目录存在但为空 — 应该有 warnings，但没有 errors
            output = load_samroad_output(tmp)
            self.assertTrue(output.is_valid)  # 空目录不是错误，只是缺少文件
            self.assertGreater(len(output.warnings), 0)
            self.assertFalse(output.has_mask)
            self.assertFalse(output.has_graph)
        finally:
            os.rmdir(tmp)

    def test_load_nonexistent_dir(self):
        """测试不存在的目录路径"""
        output = load_samroad_output("C:/nonexistent_path_xyz123/")
        self.assertFalse(output.is_valid)
        self.assertEqual(len(output.errors), 1)


class TestValidateSamroadOutput(unittest.TestCase):
    """测试验证功能"""

    def test_size_mismatch_warning(self):
        mask = np.ones((128, 128), dtype=np.uint8) * 255
        graph = _make_draft_graph(3, 4, img_w=256, img_h=256)
        tmp = _make_temp_dir({
            "road_mask_raw.png": mask,
            "draft_graph.json": graph,
        })
        try:
            output = load_samroad_output(tmp)
            output = validate_samroad_output(output, expected_size=(256, 256))
            # 应有尺寸不匹配的警告（mask 128x128 vs graph 256x256）
            has_size_warning = any(
                "不一致" in w for w in output.warnings
            )
            self.assertTrue(
                has_size_warning,
                f"预期有尺寸不匹配警告，但只有: {output.warnings}"
            )
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_edge_reference_validation(self):
        """测试边引用了不存在的节点"""
        graph = {
            "nodes": [{"id": 0, "x": 10, "y": 10, "type": "junction"}],
            "edges": [
                {"id": 0, "from": 0, "to": 99, "length_px": 100, "path": [[10, 10], [100, 100]]},
                {"id": 1, "from": 42, "to": 0, "length_px": 50, "path": [[200, 200], [10, 10]]},
            ],
            "metadata": {"node_count": 1, "edge_count": 2, "image_size": {"width": 512, "height": 512}},
        }
        tmp = _make_temp_dir({"draft_graph.json": graph})
        try:
            output = load_samroad_output(tmp)
            output = validate_samroad_output(output)
            self.assertFalse(output.is_valid)
            self.assertEqual(len(output.errors), 2)
            self.assertIn("不存在的终点节点 99", output.errors[0])
            self.assertIn("不存在的起点节点 42", output.errors[1])
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


class TestFilenameAlias(unittest.TestCase):
    """测试文件名别名兼容"""

    def test_alias_map_exists(self):
        """确认别名映射包含必需的键"""
        required_keys = [
            "road_mask_raw.png", "road_mask.png",
            "road_mask_samroad_score.png", "keypoint_mask_samroad_score.png",
            "road_skeleton.png", "skeleton.png", "draft_graph.json",
            "draft_graph_overlay.png",
        ]
        for key in required_keys:
            self.assertIn(key, FILENAME_ALIASES)

    def test_resolve_standard_name(self):
        """测试标准文件名解析"""
        tmp = _make_temp_dir({"road_mask_raw.png": np.zeros((8, 8), dtype=np.uint8)})
        try:
            result = _resolve_filename(tmp, "road_mask_raw.png")
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("road_mask_raw.png"))
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_missing_file(self):
        """测试不存在的文件"""
        tmp = tempfile.mkdtemp(prefix="samroad_alias_")
        result = _resolve_filename(tmp, "road_mask_raw.png")
        self.assertIsNone(result)
        os.rmdir(tmp)


class TestRepairEdgeEndpoints(unittest.TestCase):
    """测试边端点修复"""

    def test_repair_missing_points(self):
        nodes = [
            {"id": 0, "x": 10, "y": 20},
            {"id": 1, "x": 30, "y": 40},
        ]
        edges = [
            {"id": 0, "start": 0, "end": 1, "length_pixel": 0, "points_pixel": []},
        ]
        repaired = repair_edge_endpoints(edges, nodes)
        self.assertEqual(len(repaired[0]["points_pixel"]), 2)
        self.assertEqual(repaired[0]["points_pixel"][0], [10, 20])
        self.assertEqual(repaired[0]["points_pixel"][1], [30, 40])


class TestFindSamroadDirs(unittest.TestCase):
    """测试递归查找"""

    def test_find_in_nested(self):
        tmp = tempfile.mkdtemp(prefix="samroad_find_")
        os.makedirs(os.path.join(tmp, "sub", "deep"), exist_ok=True)
        # 在深层目录放一个 mask
        cv2.imwrite(
            os.path.join(tmp, "sub", "deep", "road_mask_raw.png"),
            np.zeros((8, 8), dtype=np.uint8),
        )
        try:
            dirs = find_samroad_dirs(tmp)
            self.assertEqual(len(dirs), 1)
            self.assertIn("deep", dirs[0])
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


class TestMaskCleanLoad(unittest.TestCase):
    """测试清理后 mask 的加载"""

    def test_load_mask_clean(self):
        mask_raw = np.ones((64, 64), dtype=np.uint8) * 255
        mask_clean = np.ones((64, 64), dtype=np.uint8) * 128  # 清理后不同
        tmp = _make_temp_dir({
            "road_mask_raw.png": mask_raw,
            "road_mask.png": mask_clean,
            "draft_graph.json": _make_draft_graph(2, 2, 64, 64),
        })
        try:
            output = load_samroad_output(tmp)
            self.assertTrue(output.has_mask)
            self.assertTrue(output.has_mask_clean)
            # mask_clean 应等于 128
            self.assertEqual(output.mask_clean[0, 0], 128)
            # mask_raw 应等于 255
            self.assertEqual(output.mask_raw[0, 0], 255)
        finally:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


# ===================================================================
# 集成测试：使用实际 SAM-Road 输出目录
# ===================================================================

class TestRealSamroadOutput(unittest.TestCase):
    """使用 outputs/samroad_004_bridge/ 进行集成测试"""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(PROJECT_ROOT, "outputs", "samroad_004_bridge")
        if not os.path.isdir(cls.data_dir):
            raise unittest.SkipTest(f"测试数据目录不存在: {cls.data_dir}")

    def test_detect_real_output(self):
        result = detect_samroad_outputs(self.data_dir)
        self.assertTrue(result["is_samroad"])
        self.assertTrue(result["has_graph"])
        self.assertTrue(result["has_mask_raw"])
        print(f"\n  [实际输出] 发现文件: {result.get('found_files', [])}")
        print(f"  [实际输出] has_skeleton={result.get('has_skeleton')}, "
              f"has_mask_score={result.get('has_mask_score')}, "
              f"has_keypoint={result.get('has_keypoint')}")

    def test_load_real_graph(self):
        nodes, edges = load_graph_for_draft(self.data_dir)
        self.assertGreater(len(nodes), 0)
        self.assertGreater(len(edges), 0)
        print(f"\n  [实际输出] Graph: {len(nodes)} 节点, {len(edges)} 边")

        # 验证节点格式
        for n in nodes[:3]:  # 检查前3个
            self.assertIn("id", n)
            self.assertIn("x", n)
            self.assertIn("y", n)

        # 验证边格式
        for e in edges[:3]:
            self.assertIn("id", e)
            self.assertIn("from", e)
            self.assertIn("to", e)
            if e.get("path"):
                pt = e["path"][0]
                self.assertEqual(len(pt), 2)
                # path 是 [y, x] 格式，y 在前
                # 这里我们只能验证两个值都是合理的数字
                self.assertIsInstance(pt[0], (int, float))
                self.assertIsInstance(pt[1], (int, float))

    def test_load_real_mask(self):
        mask = load_mask_only(self.data_dir)
        self.assertIsNotNone(mask)
        self.assertGreater(mask.shape[0], 0)
        self.assertGreater(mask.shape[1], 0)
        road_px = int((mask > 0).sum())
        total = mask.size
        print(f"\n  [实际输出] Mask: {mask.shape[1]}x{mask.shape[0]}, "
              f"道路占比 {road_px / total * 100:.1f}%")

    def test_skeleton_missing_graceful(self):
        """测试 skeleton 文件不存在时的容错"""
        skeleton = load_skeleton_from_file(self.data_dir)
        # 当前目录可能没有 skeleton，应返回 None 而不抛异常
        print(f"\n  [实际输出] Skeleton: {'有' if skeleton is not None else '无'}")


# ===================================================================
# 运行
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SAM-Road Adapter Unit Tests")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # 检查可选的真实数据目录
    real_dir = os.path.join(PROJECT_ROOT, "outputs", "samroad_004_bridge")
    if os.path.isdir(real_dir):
        print(f"[OK] Found real SAM-Road output directory: {real_dir}")
    else:
        print(f"[SKIP] No real SAM-Road output directory (integration tests will be skipped)")

    print()
    unittest.main(verbosity=2)
