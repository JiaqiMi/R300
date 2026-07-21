"""
SAM-Road 单图推理包输出适配模块。

将 D:/sam_road_single_image_share/infer_single.py 的推理输出
转换为 RoadNet Studio 内部统一格式，支持：

1. 检测输出目录中的文件：
   - road_mask.png          → 道路分割图（得分图/二值图）
   - itsc_mask.png          → 路口/关键点分割图
   - viz.png                → 可视化叠加图
   - graph.p                → Sat2Graph 风格道路拓扑图 (pickle)
   - metadata.json          → 运行元数据

2. graph.p 转换：Sat2Graph dict → 统一 graph JSON 格式
   - GraphEditorQt 内部格式：节点 {id, x, y}, 边 {id, start, end, points_pixel: [[x,y],...]}

3. 参考图层策略：
   - graph.p 转换为 reference_graph → 仅作为参考图层
   - final_graph 仍然来自 road_mask → mask 后处理 → skeleton → graph

安全要求：
- pickle 仅读取本地可信文件
- graph.p 转换失败不影响 road_mask 导入
"""

from __future__ import annotations

import json
import os
import pickle
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ===================================================================
# 数据结构
# ===================================================================

@dataclass
class SAMRoadSingleOutput:
    """单图推理包已加载的输出数据。"""
    source_dir: str = ""

    # 图像层
    road_mask: Optional[np.ndarray] = None       # road_mask.png — 道路分割图
    itsc_mask: Optional[np.ndarray] = None       # itsc_mask.png — 路口分割图
    viz: Optional[np.ndarray] = None             # viz.png — 可视化叠加图

    # Graph — Sat2Graph 原始数据
    graph_raw: Optional[dict] = None             # graph.p 原始 dict {(r,c): [(r,c),...]}
    graph_nodes: List[Dict[str, Any]] = field(default_factory=list)  # 统一格式节点
    graph_edges: List[Dict[str, Any]] = field(default_factory=list)  # 统一格式边

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    node_count: int = 0
    edge_count: int = 0
    graph_loaded: bool = False
    graph_error: str = ""  # graph.p 转换失败时的错误信息

    # 已发现文件
    found_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def has_road_mask(self) -> bool:
        return self.road_mask is not None

    @property
    def has_itsc_mask(self) -> bool:
        return self.itsc_mask is not None

    @property
    def has_viz(self) -> bool:
        return self.viz is not None

    @property
    def has_graph(self) -> bool:
        return self.graph_loaded and len(self.graph_nodes) > 0


# ===================================================================
# 文件探测
# ===================================================================

_SINGLE_FILES = [
    "road_mask.png",
    "itsc_mask.png",
    "viz.png",
    "graph.p",
    "graph.json",
    "metadata.json",
]


def detect_single_outputs(source_dir: str) -> Dict[str, bool]:
    """探测单图推理包输出目录中的文件。

    Returns:
        {
            "is_samroad_single": bool,
            "has_road_mask": bool,
            "has_itsc_mask": bool,
            "has_viz": bool,
            "has_graph": bool,
            "has_metadata": bool,
            "found_files": [str],
        }
    """
    src = Path(source_dir)
    if not src.is_dir():
        return {"is_samroad_single": False}

    found = {}
    found_files = []
    for fname in _SINGLE_FILES:
        p = src / fname
        exists = p.is_file()
        key = f"has_{fname.replace('.', '_').replace('-', '_')}"
        if fname in ("graph.p", "graph.json"):
            key = "has_graph"
        elif fname == "metadata.json":
            key = "has_metadata"
        else:
            key = f"has_{fname.split('.')[0]}"
        found[key] = found.get(key, False) or exists
        if exists:
            found_files.append(fname)

    found["is_samroad_single"] = found.get("has_road_mask", False) or found.get("has_graph", False)
    found["found_files"] = found_files
    return found


# ===================================================================
# 主加载函数
# ===================================================================

def load_single_output(source_dir: str) -> SAMRoadSingleOutput:
    """从单图推理包输出目录加载所有结果。

    Args:
        source_dir: 输出目录路径（infer_single.py 的 save/<output_dir>/）

    Returns:
        SAMRoadSingleOutput: 包含所有加载数据的结构体
    """
    result = SAMRoadSingleOutput(source_dir=source_dir)
    src = Path(source_dir)

    if not src.is_dir():
        result.errors.append(f"目录不存在: {source_dir}")
        return result

    # ── 辅助：读取灰度图 ──
    def _read_gray(filename: str) -> Optional[np.ndarray]:
        path = src / filename
        if not path.is_file():
            return None
        result.found_files.append(filename)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # 尝试彩色读取
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            result.warnings.append(f"无法读取图像: {filename}")
        return img

    def _read_color(filename: str) -> Optional[np.ndarray]:
        path = src / filename
        if not path.is_file():
            return None
        result.found_files.append(filename)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            result.warnings.append(f"无法读取图像: {filename}")
        return img

    # ── 1. 加载 metadata ──
    meta_path = src / "metadata.json"
    if meta_path.is_file():
        result.found_files.append("metadata.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                result.metadata = json.load(f)
            result.image_size = (
                int(result.metadata.get("original_width", 0)),
                int(result.metadata.get("original_height", 0)),
            )
        except Exception as e:
            result.warnings.append(f"metadata.json 读取失败: {e}")

    # ── 2. 加载 road_mask ──
    result.road_mask = _read_gray("road_mask.png")
    if result.road_mask is None:
        result.errors.append("未找到 road_mask.png")

    # ── 3. 加载 itsc_mask ──
    result.itsc_mask = _read_gray("itsc_mask.png")
    if result.itsc_mask is None:
        result.warnings.append("未找到 itsc_mask.png")

    # ── 4. 加载 viz ──
    result.viz = _read_color("viz.png")
    if result.viz is None:
        result.warnings.append("未找到 viz.png")

    # ── 5. 加载 graph.p / graph.json（始终仅供 reference graph）──
    graph_path = src / "graph.p"
    graph_json_path = src / "graph.json"
    if graph_path.is_file() or graph_json_path.is_file():
        selected_graph = graph_path if graph_path.is_file() else graph_json_path
        result.found_files.append(selected_graph.name)
        try:
            if selected_graph.suffix.lower() == ".json":
                with selected_graph.open("r", encoding="utf-8") as stream:
                    result.graph_raw = json.load(stream)
                nodes, edges = _convert_graph_json_to_unified(result.graph_raw)
            else:
                result.graph_raw = _load_graph_p(str(selected_graph))
                nodes, edges = _convert_graph_p_to_unified(result.graph_raw)
            result.graph_nodes = nodes
            result.graph_edges = edges
            result.node_count = len(nodes)
            result.edge_count = len(edges)
            result.graph_loaded = True

            # 从 graph 推断图像尺寸（如果没有 metadata）
            if result.image_size == (0, 0) and nodes:
                max_x = max(n.get("x", 0) for n in nodes)
                max_y = max(n.get("y", 0) for n in nodes)
                result.image_size = (int(max_x + 1), int(max_y + 1))

        except Exception as e:
            result.graph_error = str(e)
            result.warnings.append(f"{selected_graph.name} 转换失败 (不影响 mask 导入): {e}")
    else:
        result.warnings.append("未找到 graph.p / graph.json")

    return result


# ===================================================================
# graph.p 加载与转换
# ===================================================================

def _load_graph_p(path: str) -> dict:
    """安全读取本地可信 graph.p pickle 文件。

    graph.p 格式：Sat2Graph dict
    {
        (row_0, col_0): [(row_nbr1, col_nbr1), ...],
        (row_1, col_1): [...],
        ...
    }
    键是 (int(row), int(col)) 元组，值是邻接节点坐标列表。
    无向图：自动包含反向边。
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"graph.p 格式异常：期望 dict，实际为 {type(data).__name__}")

    # 验证键值格式
    sample_key = next(iter(data.keys()), None)
    if sample_key is not None:
        if not isinstance(sample_key, tuple) or len(sample_key) != 2:
            raise ValueError(
                f"graph.p 键格式异常：期望 (row, col) 元组，"
                f"实际为 {type(sample_key).__name__} {sample_key}"
            )

    return data


def _convert_graph_p_to_unified(graph: dict) -> Tuple[List[Dict], List[Dict]]:
    """将 Sat2Graph dict 转换为统一 graph JSON 格式。

    Sat2Graph 格式：
        {(row, col): [(row_nbr, col_nbr), ...], ...}
        坐标是 (row, col) = (y, x) 图像像素坐标

    统一格式：
        节点：{id: "n_001", x: float, y: float, type: "junction"}
        边：{id: "e_001", start: "n_001", end: "n_002",
             points_pixel: [[x, y], ...], length_pixel: float, source: "auto", enabled: True}
    """
    # 步骤 1：收集所有唯一节点
    # Sat2Graph 用 (row, col) 元组表示节点
    all_coords = set()  # {(row, col)}
    for src, neighbors in graph.items():
        r, c = int(src[0]), int(src[1])
        all_coords.add((r, c))
        for nb in neighbors:
            nr, nc = int(nb[0]), int(nb[1])
            all_coords.add((nr, nc))

    # 步骤 2：创建节点 ID 映射
    # (row, col) → "n_XXX"
    coord_to_id: Dict[Tuple[int, int], str] = {}
    nodes: List[Dict[str, Any]] = []
    for i, (r, c) in enumerate(sorted(all_coords)):
        nid = f"n_{i:03d}"
        coord_to_id[(r, c)] = nid
        nodes.append({
            "id": nid,
            "x": float(c),   # col → x
            "y": float(r),   # row → y
            "type": "junction",
        })

    # 步骤 3：创建边
    edges: List[Dict[str, Any]] = []
    seen_pairs: set = set()  # 去重 (min_id, max_id)
    edge_idx = 0

    for src_coord, neighbors in graph.items():
        src_r, src_c = int(src_coord[0]), int(src_coord[1])
        src_id = coord_to_id.get((src_r, src_c))
        if src_id is None:
            continue

        for nb_coord in neighbors:
            nb_r, nb_c = int(nb_coord[0]), int(nb_coord[1])
            nb_id = coord_to_id.get((nb_r, nb_c))
            if nb_id is None:
                continue

            # 去重（无向图）
            pair = tuple(sorted([src_id, nb_id]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # 构建 polyline
            points_pixel = [[float(src_c), float(src_r)], [float(nb_c), float(nb_r)]]
            length = float(math.hypot(nb_c - src_c, nb_r - src_r))

            edges.append({
                "id": f"e_{edge_idx:03d}",
                "start": src_id,
                "end": nb_id,
                "points_pixel": points_pixel,
                "length_pixel": round(length, 2),
                "source": "auto",
                "enabled": True,
            })
            edge_idx += 1

    return nodes, edges


def _convert_graph_json_to_unified(graph: dict) -> Tuple[List[Dict], List[Dict]]:
    """Convert common nodes/edges JSON without treating it as final_graph."""
    if not isinstance(graph, dict):
        raise ValueError(f"graph.json 期望 object，实际为 {type(graph).__name__}")
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("graph.json 必须包含 nodes/edges 数组")
    nodes = []
    known_ids = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            continue
        node_id = raw.get("id", raw.get("node_id", f"n_{index:03d}"))
        x = raw.get("x", raw.get("x_pixel", raw.get("col")))
        y = raw.get("y", raw.get("y_pixel", raw.get("row")))
        if x is None or y is None:
            continue
        known_ids.add(node_id)
        nodes.append({
            "id": node_id, "x": float(x), "y": float(y),
            "type": str(raw.get("type", "junction")),
        })
    edges = []
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, dict):
            continue
        start = raw.get("start", raw.get("source", raw.get("u", raw.get("from"))))
        end = raw.get("end", raw.get("target", raw.get("v", raw.get("to"))))
        if start not in known_ids or end not in known_ids:
            continue
        points = raw.get("points_pixel", raw.get("polyline", []))
        edges.append({
            "id": raw.get("id", f"e_{index:03d}"),
            "start": start, "end": end,
            "points_pixel": points if isinstance(points, list) else [],
            "length_pixel": float(raw.get("length_pixel", 0.0)),
            "source": "samroadplus_reference",
            "enabled": bool(raw.get("enabled", True)),
        })
    return nodes, edges


# ===================================================================
# 便捷接口
# ===================================================================

def load_single_reference_graph(source_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """仅加载 graph.p 并转换为参考图层格式（不加载 mask）。

    Returns:
        (nodes, edges) 内部格式
    """
    graph_path = Path(source_dir) / "graph.p"
    if not graph_path.is_file():
        return [], []

    try:
        graph_raw = _load_graph_p(str(graph_path))
        return _convert_graph_p_to_unified(graph_raw)
    except Exception:
        return [], []


def load_road_mask_only(source_dir: str) -> Optional[np.ndarray]:
    """仅加载 road_mask.png。"""
    path = Path(source_dir) / "road_mask.png"
    if path.is_file():
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None
