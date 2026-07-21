"""
Graph 工具函数：numpy 安全类型判断与转换。

解决 numpy.ndarray 在布尔上下文中产生的 "ambiguous truth value" 错误。
统一所有 graph 模块的数据格式，确保输出的 node 坐标和 edge.polyline
都是普通 Python list/float 类型而非 numpy array。

用法：
    from roadnet.graph_utils import (
        is_empty_array_like, has_points,
        point_to_tuple, polyline_to_list,
        ensure_python_types, ensure_graph_python_types,
    )
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple, Optional


# ===========================================================================
# 核心判断函数
# ===========================================================================

def is_empty_array_like(x: Any) -> bool:
    """安全地判断变量是否为空（None、空 numpy array、空 list）。

    可用于替换危险的 `if not x:` 写法（当 x 可能是 numpy.ndarray 时）。

    Args:
        x: 任意值（可能是 None、numpy array、list、tuple 等）

    Returns:
        True 表示 x 是 None 或尺寸为 0

    Examples:
        >>> is_empty_array_like(None)
        True
        >>> is_empty_array_like(np.array([]))
        True
        >>> is_empty_array_like(np.array([1, 2]))
        False
        >>> is_empty_array_like([])
        True
        >>> is_empty_array_like([[1, 2], [3, 4]])
        False
    """
    if x is None:
        return True
    try:
        return np.asarray(x).size == 0
    except Exception:
        try:
            return len(x) == 0
        except Exception:
            return False


def has_points(x: Any) -> bool:
    """安全地判断变量是否包含有效数据。

    是 is_empty_array_like 的反义。

    Examples:
        >>> has_points(np.array([[1,2],[3,4]]))
        True
        >>> has_points([])
        False
        >>> has_points(None)
        False
    """
    return not is_empty_array_like(x)


# ===========================================================================
# 类型转换函数
# ===========================================================================

def point_to_tuple(p: Any) -> Tuple[float, float]:
    """将任意点表示转换为 (float, float) 元组。

    支持：tuple, list, numpy.ndarray, numpy scalar

    Args:
        p: 任意 2 坐标表示，如 [x, y], (x, y), np.array([x, y])

    Returns:
        (float, float)

    Raises:
        ValueError: 如果坐标维度不是 2

    Examples:
        >>> point_to_tuple([10, 20])
        (10.0, 20.0)
        >>> point_to_tuple(np.array([10.5, 20.5]))
        (10.5, 20.5)
    """
    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected at least 2 coordinates, got {arr.size}")
    return (float(arr[0]), float(arr[1]))


def polyline_to_list(polyline: Any) -> List[List[float]]:
    """将任意 polyline 表示转换为标准 Python list[list[float, float]]。

    支持：list of lists, list of tuples, numpy.ndarray (N,2), numpy.ndarray (2,)

    Args:
        polyline: 任意 polyline 表示

    Returns:
        [[float, float], ...] 格式的纯 Python 列表

    Examples:
        >>> polyline_to_list(np.array([[1,2],[3,4]]))
        [[1.0, 2.0], [3.0, 4.0]]
        >>> polyline_to_list([])
        []
        >>> polyline_to_list(None)
        []
    """
    if polyline is None:
        return []
    arr = np.asarray(polyline, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in arr]


def points_to_list_of_tuples(points: Any) -> List[Tuple[float, float]]:
    """将任意点集转换为 list of tuples。

    Args:
        points: 任意点集表示，形状为 (N, 2)

    Returns:
        [(float, float), ...]
    """
    if points is None:
        return []
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in arr]


# ===========================================================================
# 安全比较函数
# ===========================================================================

def points_equal(p1: Any, p2: Any) -> bool:
    """安全地比较两个点是否相等（容忍 numpy array 类型差异）。"""
    try:
        return np.array_equal(
            np.asarray(p1, dtype=np.float64).reshape(-1),
            np.asarray(p2, dtype=np.float64).reshape(-1),
        )
    except Exception:
        return False


# ===========================================================================
# JSON 序列化安全函数
# ===========================================================================

def ensure_python_types(obj: Any, max_depth: int = 20) -> Any:
    """递归地将 numpy 类型的值转换为 Python 原生类型，确保可 JSON 序列化。

    支持嵌套的 dict、list、tuple 结构。

    Args:
        obj: 任意 Python 对象（可能包含 numpy 类型）
        max_depth: 最大递归深度，防止无限递归

    Returns:
        等价的纯 Python 类型版本
    """
    if max_depth <= 0:
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return ensure_python_types(obj.tolist(), max_depth - 1)
    if isinstance(obj, dict):
        return {k: ensure_python_types(v, max_depth - 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_python_types(item, max_depth - 1) for item in obj]
    # numpy bool types
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def ensure_graph_python_types(nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """确保 graph 中的所有值都是 Python 原生类型，而非 numpy 类型。

    特别处理：
    - node["x"], node["y"] → int
    - edge["from"], edge["to"] → int
    - edge["path"] / edge["points_pixel"] → [[float, float], ...]
    - edge["length_px"] / edge["length_pixel"] → float

    Args:
        nodes: 节点列表
        edges: 边列表

    Returns:
        (cleaned_nodes, cleaned_edges)
    """
    cleaned_nodes = []
    for n in nodes:
        cn = {
            "id": int(n.get("id", 0)),
            "y": int(n.get("y", 0)),
            "x": int(n.get("x", 0)),
            "type": str(n.get("type", "endpoint")),
        }
        for k in ("degree",):
            if k in n:
                cn[k] = int(n[k])
        for k in ("source",):
            if k in n:
                cn[k] = str(n[k])
        cleaned_nodes.append(cn)

    cleaned_edges = []
    for e in edges:
        ce = {
            "id": int(e.get("id", 0)),
            "from": int(e.get("from", 0)),
            "to": int(e.get("to", 0)),
        }
        # path 可能是 "path" (draft) 或 "points_pixel" (final)
        for pk in ("path", "points_pixel"):
            if pk in e:
                ce[pk] = polyline_to_list(e[pk])
        for lk in ("length_px", "length_pixel"):
            if lk in e:
                ce[lk] = float(e[lk])
        for sk in ("source",):
            if sk in e:
                ce[sk] = str(e[sk])
        if "enabled" in e:
            ce["enabled"] = bool(e.get("enabled", True))
        cleaned_edges.append(ce)

    return cleaned_nodes, cleaned_edges


# ===========================================================================
# numpy.bool_ 安全包装
# ===========================================================================

def as_bool(x: Any) -> bool:
    """安全地将 numpy.bool_ 或标量转为 Python bool。

    用于替代直接 `if arr[idx]:` 的场景。

    Args:
        x: numpy.bool_ 或可转为 bool 的值

    Returns:
        Python bool

    Examples:
        >>> binary = np.array([[True, False]])
        >>> as_bool(binary[0, 0])
        True
        >>> as_bool(binary[0, 1])
        False
    """
    if isinstance(x, (np.bool_, np.integer)):
        return bool(x)
    return bool(x)
