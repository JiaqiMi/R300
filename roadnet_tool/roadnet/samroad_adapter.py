"""
SAM-Road 输出适配模块。

将 SAM-Road 模型的推理输出（mask、skeleton、graph JSON）
转换为 RoadNet Studio 内部统一格式，支持：

1. 加载并验证 SAM-Road 输出文件
2. Mask / 得分图 / 骨架图加载（原始图像分辨率）
3. draft_graph.json 解析 → GraphEditorQt 内部节点/边格式
4. 元数据提取（图像尺寸、节点数、边数）
5. 文件名兼容映射（非标准文件名自动识别）
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from roadnet.graph_utils import is_empty_array_like


# ===================================================================
# 文件名兼容映射：处理非标准命名
# ===================================================================

# 将常见变体文件名映射到标准名称
FILENAME_ALIASES: Dict[str, List[str]] = {
    "road_mask_raw.png":              ["road_mask_raw.png", "mask_raw.png", "raw_mask.png"],
    "road_mask.png":                  ["road_mask.png", "mask.png", "road_mask_clean.png",
                                       "clean_mask.png", "mask_clean.png"],
    "road_mask_samroad_score.png":    ["road_mask_samroad_score.png", "road_mask_score.png",
                                       "mask_score.png", "samroad_score.png"],
    "keypoint_mask_samroad_score.png":["keypoint_mask_samroad_score.png", "keypoint_mask_score.png",
                                       "keypoint_score.png"],
    "road_skeleton.png":              ["road_skeleton.png", "road_skeleton_raw.png"],
    "skeleton.png":                   ["skeleton.png", "skeleton_raw.png", "road_skeleton.png"],
    "draft_graph.json":               ["draft_graph.json", "draft_graph.json"],
    "draft_graph_overlay.png":        ["draft_graph_overlay.png", "draft_graph_overlay.png"],
}


def _resolve_filename(source_dir: str, canonical_name: str) -> Optional[str]:
    """在目录中查找文件，支持别名映射。

    优先精确匹配，其次按别名列表查找。
    返回实际文件名（含路径），找不到则返回 None。
    """
    src = Path(source_dir)
    if not src.is_dir():
        return None

    # 1. 精确匹配
    exact = src / canonical_name
    if exact.is_file():
        return str(exact)

    # 2. 别名列表查找
    aliases = FILENAME_ALIASES.get(canonical_name, [canonical_name])
    for alias in aliases:
        candidate = src / alias
        if candidate.is_file():
            return str(candidate)

    # 3. 模糊匹配：同文件名的不同变体（不区分大小写）
    stem_lower = Path(canonical_name).stem.lower()
    suffix = Path(canonical_name).suffix.lower()
    for f in src.iterdir():
        if f.is_file() and f.suffix.lower() == suffix and f.stem.lower() == stem_lower:
            return str(f)

    return None


# ===================================================================
# 数据结构
# ===================================================================

@dataclass
class SAMRoadOutput:
    """已加载的 SAM-Road 输出数据"""
    # 文件路径
    source_dir: str = ""

    # Mask（原始图像分辨率，uint8 0/255）
    mask_raw: Optional[np.ndarray] = None       # road_mask_raw.png — 原始二值 mask
    mask_clean: Optional[np.ndarray] = None      # road_mask.png — 后处理的干净 mask
    mask_score: Optional[np.ndarray] = None      # road_mask_samroad_score.png — 道路得分图
    keypoint_score: Optional[np.ndarray] = None  # keypoint_mask_samroad_score.png

    # Skeleton（原始图像分辨率，uint8 0/255）
    skeleton: Optional[np.ndarray] = None        # skeleton.png / road_skeleton.png

    # 已发现文件列表
    found_files: List[str] = field(default_factory=list)

    # Graph
    nodes: List[Dict[str, Any]] = field(default_factory=list)  # 内部格式
    edges: List[Dict[str, Any]] = field(default_factory=list)  # 内部格式

    # 元数据
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    node_count: int = 0
    edge_count: int = 0
    graph_loaded: bool = False

    # 验证信息
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def has_mask(self) -> bool:
        return self.mask_raw is not None

    @property
    def has_mask_clean(self) -> bool:
        return self.mask_clean is not None

    @property
    def has_skeleton(self) -> bool:
        return self.skeleton is not None

    @property
    def has_graph(self) -> bool:
        return self.graph_loaded and len(self.nodes) > 0

    @property
    def mask_shape(self) -> Optional[Tuple[int, int]]:
        if self.mask_raw is not None:
            h, w = self.mask_raw.shape[:2]
            return (w, h)
        return None


# ===================================================================
# 公共 API
# ===================================================================

def load_samroad_output(source_dir: str) -> SAMRoadOutput:
    """从 SAM-Road 输出目录加载所有结果。

    自动检测并加载以下文件（按需，支持别名）：
    - road_mask_raw.png              → mask_raw (0/255 二值图)
    - road_mask.png                  → mask_clean (后处理 mask)
    - road_mask_samroad_score.png    → mask_score (0-255 得分图)
    - keypoint_mask_samroad_score.png → keypoint_score
    - road_skeleton.png / skeleton.png → skeleton
    - draft_graph.json               → nodes + edges

    Args:
        source_dir: SAM-Road 输出目录路径

    Returns:
        SAMRoadOutput: 包含所有加载数据的结构体
    """
    result = SAMRoadOutput(source_dir=source_dir)
    src = Path(source_dir)

    if not src.is_dir():
        result.errors.append(f"目录不存在: {source_dir}")
        return result

    # ── 辅助：尝试读取灰度图 ──
    def _try_read_gray(filename: str):
        path = _resolve_filename(source_dir, filename)
        if path is None:
            return None
        result.found_files.append(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            result.warnings.append(f"无法读取图像: {os.path.basename(path)}")
        return img

    # ── 1. 加载原始 mask ──
    result.mask_raw = _try_read_gray("road_mask_raw.png")
    if result.mask_raw is None:
        result.warnings.append("未找到 road_mask_raw.png")

    # ── 2. 加载清理后 mask ──
    result.mask_clean = _try_read_gray("road_mask.png")

    # ── 3. 加载得分图 ──
    result.mask_score = _try_read_gray("road_mask_samroad_score.png")
    result.keypoint_score = _try_read_gray("keypoint_mask_samroad_score.png")

    # ── 4. 加载 skeleton ──
    # 优先 road_skeleton.png，其次 skeleton.png
    skeleton_img = _try_read_gray("road_skeleton.png")
    if skeleton_img is None:
        skeleton_img = _try_read_gray("skeleton.png")
    result.skeleton = skeleton_img
    if result.skeleton is None:
        result.warnings.append("未找到 skeleton.png 或 road_skeleton.png")

    # ── 5. 加载 graph ──
    graph_path = _resolve_filename(source_dir, "draft_graph.json")
    if graph_path is not None:
        result.found_files.append(os.path.basename(graph_path))
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph = json.load(f)

            raw_nodes = graph.get("nodes", [])
            raw_edges = graph.get("edges", [])
            meta = graph.get("metadata", {})

            # 提取图像尺寸
            img_meta = meta.get("image_size", {})
            result.image_size = (
                int(img_meta.get("width", 0)),
                int(img_meta.get("height", 0)),
            )
            result.node_count = meta.get("node_count", len(raw_nodes))
            result.edge_count = meta.get("edge_count", len(raw_edges))

            # 转换为内部格式（与 GraphEditorQt.load_draft 一致）
            result.nodes = _convert_nodes(raw_nodes)
            result.edges = _convert_edges(raw_edges)
            result.graph_loaded = True

        except json.JSONDecodeError:
            result.errors.append(f"draft_graph.json 格式错误，无法解析 JSON")
        except Exception as e:
            result.errors.append(f"加载 draft_graph.json 失败: {e}")
    else:
        result.warnings.append("未找到 draft_graph.json")

    return result


def validate_samroad_output(
    output: SAMRoadOutput,
    expected_size: Optional[Tuple[int, int]] = None,
) -> SAMRoadOutput:
    """验证 SAM-Road 输出数据的完整性和一致性。

    Args:
        output: 已加载的 SAM-Road 输出
        expected_size: 期望的图像尺寸 (width, height)，用于检查 mask 尺寸是否匹配

    Returns:
        更新了 errors/warnings 的同一对象
    """
    # 检查 mask 与 graph 的图像尺寸是否匹配
    if output.has_mask and output.graph_loaded:
        mask_w, mask_h = output.mask_raw.shape[1], output.mask_raw.shape[0]
        gw, gh = output.image_size
        if (mask_w != gw or mask_h != gh) and gw > 0:
            output.warnings.append(
                f"Mask 尺寸 ({mask_w}x{mask_h}) 与 draft_graph.json 中记录的不一致 ({gw}x{gh})"
            )

    # 检查 skeleton 与 mask 的尺寸
    if output.has_skeleton and output.has_mask:
        sk_h, sk_w = output.skeleton.shape[:2]
        mk_h, mk_w = output.mask_raw.shape[:2]
        if (sk_w != mk_w or sk_h != mk_h):
            output.warnings.append(
                f"Skeleton 尺寸 ({sk_w}x{sk_h}) 与 mask 尺寸 ({mk_w}x{mk_h}) 不一致"
            )

    # 检查与当前图像的尺寸是否匹配
    if expected_size is not None:
        ew, eh = expected_size
        if output.has_mask:
            mw, mh = output.mask_raw.shape[1], output.mask_raw.shape[0]
            if (mw != ew or mh != eh):
                output.warnings.append(
                    f"SAM-Road mask 尺寸 ({mw}x{mh}) 与当前图像 ({ew}x{eh}) 不一致"
                )
        if output.has_skeleton:
            sw, sh = output.skeleton.shape[1], output.skeleton.shape[0]
            if (sw != ew or sh != eh):
                output.warnings.append(
                    f"SAM-Road skeleton 尺寸 ({sw}x{sh}) 与当前图像 ({ew}x{eh}) 不一致"
                )

    # 检查 edge 引用的节点 ID 是否都存在
    if output.graph_loaded:
        node_ids = {n["id"] for n in output.nodes}
        for e in output.edges:
            if e["start"] not in node_ids:
                output.errors.append(f"边 {e['id']} 引用了不存在的起点节点 {e['start']}")
            if e["end"] not in node_ids:
                output.errors.append(f"边 {e['id']} 引用了不存在的终点节点 {e['end']}")

    return output


def load_mask_only(source_dir: str) -> Optional[np.ndarray]:
    """仅加载 SAM-Road mask（快速模式）。

    优先加载 road_mask.png（清理后），其次 road_mask_raw.png。
    """
    for fname in ["road_mask.png", "road_mask_raw.png"]:
        path = _resolve_filename(source_dir, fname)
        if path:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None


def load_mask_clean_only(source_dir: str) -> Optional[np.ndarray]:
    """仅加载清理后的 SAM-Road mask（road_mask.png）。"""
    path = _resolve_filename(source_dir, "road_mask.png")
    if path:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None


def load_skeleton_from_file(source_dir: str) -> Optional[np.ndarray]:
    """从 SAM-Road 输出目录加载 skeleton 图像。

    优先加载 road_skeleton.png，其次 skeleton.png。
    返回 (H, W) uint8 灰度图，0/255。
    尺寸为原始图像分辨率。
    """
    for fname in ["road_skeleton.png", "skeleton.png"]:
        path = _resolve_filename(source_dir, fname)
        if path is None:
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    return None


def load_graph_only(source_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """仅加载 SAM-Road draft graph（快速模式，返回内部格式）。

    Returns:
        (nodes, edges) 内部格式的节点和边列表
    """
    graph_path = _resolve_filename(source_dir, "draft_graph.json")
    if graph_path is None:
        return [], []

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = _convert_nodes(graph.get("nodes", []))
    edges = _convert_edges(graph.get("edges", []))
    return nodes, edges


def load_graph_for_draft(source_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """仅加载 SAM-Road draft graph（返回原始 draft 格式，可直接传给 load_draft）。

    GraphEditorQt.load_draft() 内部会自行处理格式转换（path[y,x]→points_pixel[x,y]），
    因此本函数返回原始格式的数据。

    Returns:
        (nodes, edges) 原始 draft 格式，节点 {id, x, y, type}，边 {id, from, to, path: [[y,x],...]}
    """
    graph_path = _resolve_filename(source_dir, "draft_graph.json")
    if graph_path is None:
        return [], []

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # 返回原始 draft 格式 —— load_draft 会自行转换
    raw_nodes = list(graph.get("nodes", []))
    raw_edges = list(graph.get("edges", []))
    return raw_nodes, raw_edges


# ===================================================================
# 内部转换函数
# ===================================================================

def _convert_nodes(raw_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 SAM-Road draft graph 的节点格式转换为内部格式。

    Draft 节点格式: {id, x, y, type, [degree]}
    内部节点格式:   {id, x, y, type, source}

    注意：draft_graph.json 中节点使用 x/y 字段（图像列/行），
    与 GraphEditorQt 内部格式一致，无需转换坐标轴。
    """
    converted = []
    for n in raw_nodes:
        converted.append({
            "id": n["id"],
            "x": float(n.get("x", 0)),
            "y": float(n.get("y", 0)),
            "type": n.get("type", "junction"),
            "source": "auto",
        })
    return converted


def _convert_edges(raw_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 SAM-Road draft graph 的边格式转换为内部格式。

    Draft 边格式: {id, from, to, length_px, path: [[y, x], ...]}
    内部边格式:   {id, start, end, length_pixel, points_pixel: [[x, y], ...], source, enabled}

    关键转换：
    1. from/to → start/end
    2. path 中的 [y, x] → [x, y]（坐标轴交换）
    """
    converted = []

    def _path_length(points: List[List[float]]) -> float:
        total = 0.0
        for p0, p1 in zip(points, points[1:]):
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            total += (dx * dx + dy * dy) ** 0.5
        return total

    for e in raw_edges:
        path = e.get("path", [])
        # path 是 [[y, x], ...] 格式，转换为 [[x, y], ...]
        points_pixel = [[float(p[1]), float(p[0])] for p in path] if path else []

        if len(points_pixel) == 0:
            # 没有 path 时，根据 from/to 节点的坐标构建简单直线
            # 这种情况需要在调用方已经有节点数据时处理
            points_pixel = []

        length = float(e.get("length_px", e.get("length_pixel", 0)))
        if length <= 0 and len(points_pixel) >= 2:
            length = round(_path_length(points_pixel), 2)

        converted.append({
            "id": e["id"],
            "start": e.get("from", e.get("start", -1)),
            "end": e.get("to", e.get("end", -1)),
            "length_pixel": length,
            "points_pixel": points_pixel,
            "source": "auto",
            "enabled": True,
        })

    return converted


def repair_edge_endpoints(
    edges: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """修复缺失端点的边：为没有 points_pixel 的边补充端点坐标。

    当边只有 start/end 节点 ID 但没有 path 时，从节点坐标构造直线路径。

    Args:
        edges: 内部格式的边列表
        nodes: 内部格式的节点列表

    Returns:
        修复后的边列表
    """
    node_map = {n["id"]: n for n in nodes}
    repaired = []

    for e in edges:
        edge = dict(e)
        if not edge.get("points_pixel") or len(edge["points_pixel"]) < 2:
            start_node = node_map.get(edge["start"])
            end_node = node_map.get(edge["end"])
            if start_node and end_node:
                edge["points_pixel"] = [
                    [start_node["x"], start_node["y"]],
                    [end_node["x"], end_node["y"]],
                ]
            else:
                edge["points_pixel"] = [[0, 0], [0, 0]]
        repaired.append(edge)

    return repaired


# ===================================================================
# 文件探测
# ===================================================================

def detect_samroad_outputs(source_dir: str) -> Dict[str, bool]:
    """探测目录中的 SAM-Road 输出文件。

    Returns:
        {
            "is_samroad": bool,          # 是否为 SAM-Road 输出目录
            "has_mask_raw": bool,         # road_mask_raw.png
            "has_mask_clean": bool,       # road_mask.png
            "has_mask_score": bool,       # road_mask_samroad_score.png
            "has_graph": bool,            # draft_graph.json
            "has_skeleton": bool,         # skeleton.png 或 road_skeleton.png
            "has_skeleton_road": bool,    # road_skeleton.png
            "has_keypoint": bool,         # keypoint_mask_samroad_score.png
            "has_overlay": bool,          # draft_graph_overlay.png
            "found_files": [str],         # 实际发现的文件名列表
        }
    """
    src = Path(source_dir)
    if not src.is_dir():
        return {"is_samroad": False}

    def _exists(canonical: str) -> bool:
        return _resolve_filename(source_dir, canonical) is not None

    has_mask_raw = _exists("road_mask_raw.png")
    has_mask_clean = _exists("road_mask.png")
    has_graph = _exists("draft_graph.json")
    has_keypoint = _exists("keypoint_mask_samroad_score.png")

    found_files = []
    for fname in [
        "road_mask_raw.png", "road_mask.png", "road_mask_samroad_score.png",
        "keypoint_mask_samroad_score.png", "road_skeleton.png", "skeleton.png",
        "draft_graph.json", "draft_graph_overlay.png",
    ]:
        resolved = _resolve_filename(source_dir, fname)
        if resolved:
            found_files.append(os.path.basename(resolved))

    return {
        "is_samroad": has_mask_raw or has_graph,
        "has_mask_raw": has_mask_raw,
        "has_mask_clean": has_mask_clean,
        "has_mask_score": _exists("road_mask_samroad_score.png"),
        "has_graph": has_graph,
        "has_skeleton": _exists("road_skeleton.png") or _exists("skeleton.png"),
        "has_skeleton_road": _exists("road_skeleton.png"),
        "has_keypoint": has_keypoint,
        "has_overlay": _exists("draft_graph_overlay.png"),
        "found_files": found_files,
    }


def find_samroad_dirs(root_dir: str) -> List[str]:
    """递归查找包含 SAM-Road 输出文件的目录。"""
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        files = set(filenames)
        if "road_mask_raw.png" in files or "draft_graph.json" in files:
            results.append(dirpath)
    return results


# ===================================================================
# Mask → Skeleton 自动生成接口
# ===================================================================

def generate_skeleton_from_mask(
    mask: np.ndarray,
    method: str = "medial_axis",
    min_branch_length: int = 40,
    max_connect_dist: int = 45,
    max_connect_angle: float = 45.0,
    border_margin: int = 10,
) -> Optional[np.ndarray]:
    """从道路 mask 自动生成骨架线（skeleton）。

    这是一个便捷接口，内部调用 optimized_skeleton 模块的完整流水线。

    Args:
        mask: 二值 mask (H, W) uint8，0/255
        method: 骨架化方法 "medial_axis" 或 "thin"
        min_branch_length: 最小分支长度（像素），短于此值的分支将被剪除
        max_connect_dist: 最大断点连接距离
        max_connect_angle: 最大连接角度（度）
        border_margin: 边界冗余（像素）

    Returns:
        uint8 骨架图 (H, W) 0/255，或 None（生成失败时）
    """
    if mask is None or mask.size == 0:
        return None

    # 确保是二值 0/255
    mask_bin = (mask > 0).astype(np.uint8) * 255

    try:
        from roadnet.optimized_skeleton import skeletonize_medial_axis, optimize_skeleton

        # Step 1: 骨架化
        if method == "medial_axis":
            raw = skeletonize_medial_axis(mask_bin)
        else:
            from roadnet.optimized_skeleton import skeletonize_thin
            raw = skeletonize_thin(mask_bin)

        # Step 2: 优化（剪枝+平滑+断点连接）
        result = optimize_skeleton(
            mask_bin, raw,
            min_center_dist=3,
            border_margin=border_margin,
            min_branch_length=min_branch_length,
            max_connect_dist=max_connect_dist,
            max_connect_angle=max_connect_angle,
            min_line_mask_overlap=0.65,
        )

        optimized = result.get("optimized_skeleton")
        if optimized is not None:
            return (optimized > 0).astype(np.uint8) * 255

    except ImportError as e:
        # 降级：只做基本骨架化
        print(f"[SamroadAdapter] 优化模块不可用 ({e})，使用基础骨架化")
        try:
            from roadnet.optimized_skeleton import skeletonize_medial_axis
            return skeletonize_medial_axis(mask_bin)
        except ImportError:
            print(f"[SamroadAdapter] 骨架化模块也不可用")
            return None
    except Exception as e:
        print(f"[SamroadAdapter] 骨架生成失败: {e}")
        return None

    return None
