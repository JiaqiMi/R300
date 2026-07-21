"""
任务点文件解析模块

比赛标准格式：
  序号;经度;纬度;高程;属性
  1;105.62300;39.29636;0;0

point_type：0=起点, 1=终点, 2=必经点

兼容旧格式第 6 列 reserve。手动点击与文件导入统一为 TaskPoint。
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskPoint:
    """任务点数据结构（全局像素坐标 + WGS84）"""
    seq: int
    longitude: Optional[float]
    latitude: Optional[float]
    altitude: float = 0.0
    point_type: int = 2          # 0=起点, 1=终点, 2=必经点
    reserve: str = ""
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    map_x: Optional[float] = None
    map_y: Optional[float] = None
    status: str = "pending"
    inside_image: Optional[bool] = None
    created_order: int = 0
    source: str = ""             # "file_import" / "manual_click"
    snap_status: str = ""
    snap_distance: Optional[float] = None

    @property
    def type_name(self) -> str:
        return {0: "start", 1: "goal", 2: "task"}.get(self.point_type, "task")

    @property
    def is_start(self) -> bool:
        return self.point_type == 0

    @property
    def is_goal(self) -> bool:
        return self.point_type == 1

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "altitude": self.altitude,
            "point_type": self.point_type,
            "type_name": self.type_name,
            "reserve": self.reserve,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "status": self.status,
            "inside_image": self.inside_image,
            "created_order": self.created_order,
            "source": self.source,
            "snap_status": self.snap_status,
            "snap_distance": self.snap_distance,
        }

    def __repr__(self):
        coords = ""
        if self.pixel_x is not None:
            coords = f", pixel=({self.pixel_x:.1f},{self.pixel_y:.1f})"
        return (f"TaskPoint(seq={self.seq}, type={self.type_name}"
                f", lon={self.longitude}, lat={self.latitude}{coords})")


# ===================================================================
# 解析辅助
# ===================================================================

def _strip_line_comment(line: str) -> str:
    for marker in ("//", "#"):
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def _detect_delimiter(line: str) -> str:
    """标准格式优先英文分号；再兼容中文分号、逗号、tab、空格。"""
    if ";" in line:
        return ";"
    if "\uff1b" in line:  # ；
        return "\uff1b"
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return " "


def _split_line(line: str, delim: str) -> List[str]:
    if delim in (" ", "\t"):
        return [p for p in re.split(r"\s+", line.strip()) if p]
    # 统一中文分号到英文后再切
    s = line.replace("\uff1b", ";") if delim in (";", "\uff1b") else line
    use = ";" if delim in (";", "\uff1b") else delim
    return [p.strip() for p in s.split(use) if p.strip() or p == "0"]


def _is_numeric(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_header_line(parts: List[str]) -> bool:
    if len(parts) < 2:
        return False
    joined = " ".join(parts).lower()
    header_keys = (
        "序号", "经度", "纬度", "高程", "属性", "seq", "longitude", "latitude",
        "altitude", "type", "point_type", "lon", "lat", "备用", "reserve",
    )
    if any(k in joined for k in header_keys):
        return True
    non_numeric = sum(1 for p in parts if not _is_numeric(p))
    return non_numeric >= 2


def _lon_lat_swap_suspect(lon: float, lat: float) -> bool:
    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        if (-180.0 <= lat <= 180.0) and (-90.0 <= lon <= 90.0):
            return True
        return False
    # 中国区域常见填反：lon 像纬度、lat 像经度
    if -90.0 <= lon <= 90.0 and 100.0 <= abs(lat) <= 180.0:
        return True
    return False


def _try_parse_row(parts: List[str]) -> Tuple[Optional[List], Optional[str], bool]:
    """解析一行。

    Returns:
        (parsed_list | None, error_message | None, swap_suspect)
        parsed_list = [seq, lon, lat, alt, ptype, reserve]
    """
    if len(parts) < 5:
        return None, "任务点文件格式错误，应为：序号;经度;纬度;高程;属性。", False
    try:
        seq = int(float(parts[0]))
        lon = float(parts[1])
        lat = float(parts[2])
        alt = float(parts[3])
        ptype = int(float(parts[4]))
        reserve = parts[5] if len(parts) >= 6 else ""
    except (ValueError, IndexError):
        return None, "任务点文件格式错误，应为：序号;经度;纬度;高程;属性。", False

    if ptype not in (0, 1, 2):
        return None, "任务点属性必须为 0 起点、1 终点、2 必经点。", False

    swap = _lon_lat_swap_suspect(lon, lat)
    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        if swap:
            # 保留原值，交给 UI 询问是否交换
            return [seq, lon, lat, alt, ptype, reserve], None, True
        return None, f"经纬度超出有效范围: lon={lon}, lat={lat}", False

    return [seq, lon, lat, alt, ptype, reserve], None, swap


# ===================================================================
# 公开 API
# ===================================================================

def parse_task_points_txt(file_path: str) -> Dict[str, Any]:
    """解析比赛任务点 txt。

    Returns:
        {
          ok, error, points, warnings, swap_suspect, invalid_rows,
          start_count, goal_count, waypoint_count, file_path
        }
    """
    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "points": [],
        "warnings": [],
        "swap_suspect": False,
        "invalid_rows": 0,
        "start_count": 0,
        "goal_count": 0,
        "waypoint_count": 0,
        "file_path": file_path,
    }
    if not os.path.exists(file_path):
        result["error"] = f"任务点文件不存在: {file_path}"
        return result

    try:
        raw_text = _read_with_fallback_encoding(file_path)
    except Exception as exc:
        result["error"] = f"无法读取任务点文件: {exc}"
        return result

    lines = [_strip_line_comment(l) for l in raw_text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines:
        result["error"] = "任务点文件为空"
        return result

    delim = _detect_delimiter(lines[0]) or ";"
    first_parts = _split_line(lines[0], delim)
    start_idx = 1 if _is_header_line(first_parts) else 0

    points: List[TaskPoint] = []
    for line_no, line in enumerate(lines[start_idx:], start=start_idx + 1):
        parts = _split_line(line, delim)
        parsed, err, swap = _try_parse_row(parts)
        if err:
            result["invalid_rows"] += 1
            result["error"] = f"第 {line_no} 行：{err}"
            return result
        if parsed is None:
            result["invalid_rows"] += 1
            result["warnings"].append(f"第 {line_no} 行无法解析，已跳过")
            continue
        if swap:
            result["swap_suspect"] = True
        seq, lon, lat, alt, ptype, reserve = parsed
        tp = TaskPoint(
            seq=seq,
            longitude=lon,
            latitude=lat,
            altitude=alt,
            point_type=ptype,
            reserve=reserve,
            pixel_x=None,
            pixel_y=None,
            status="pending",
            created_order=len(points),
            source="file_import",
        )
        points.append(tp)

    if not points:
        result["error"] = "未加载到任何任务点"
        return result

    points.sort(key=lambda tp: int(tp.seq))
    starts = [tp for tp in points if tp.point_type == 0]
    goals = [tp for tp in points if tp.point_type == 1]
    vias = [tp for tp in points if tp.point_type == 2]
    result["start_count"] = len(starts)
    result["goal_count"] = len(goals)
    result["waypoint_count"] = len(vias)

    if len(starts) != 1 or len(goals) != 1:
        result["error"] = "任务点文件中起点/终点数量异常，请检查 point_type。"
        result["points"] = points
        return result

    # type 与 seq 位置冲突警告（仍按 seq 规划）
    min_seq = min(tp.seq for tp in points)
    max_seq = max(tp.seq for tp in points)
    min_tp = next(tp for tp in points if tp.seq == min_seq)
    max_tp = next(tp for tp in points if tp.seq == max_seq)
    if min_tp.point_type != 0:
        result["warnings"].append(
            f"警告：seq 最小点 (seq={min_seq}) 的 point_type={min_tp.point_type}，"
            "通常应为 0（起点）；仍按 seq 顺序规划。"
        )
    if max_tp.point_type != 1:
        result["warnings"].append(
            f"警告：seq 最大点 (seq={max_seq}) 的 point_type={max_tp.point_type}，"
            "通常应为 1（终点）；仍按 seq 顺序规划。"
        )

    result["points"] = points
    result["ok"] = True
    return result


def load_task_points(path: str, coordinate_type: str = "geo") -> List[TaskPoint]:
    """加载任务点文件（兼容入口）。

    优先走 parse_task_points_txt；失败时抛出 ValueError。
    coordinate_type="pixel" 时将 lon/lat 字段当作像素（兼容旧用法）。
    """
    parsed = parse_task_points_txt(path)
    if not parsed.get("ok"):
        # 若仅因起点终点数量失败但仍有点，仍抛错让 UI 提示
        raise ValueError(parsed.get("error") or "任务点解析失败")

    points: List[TaskPoint] = list(parsed["points"])
    if coordinate_type == "pixel":
        for tp in points:
            tp.pixel_x = float(tp.longitude) if tp.longitude is not None else None
            tp.pixel_y = float(tp.latitude) if tp.latitude is not None else None
            tp.status = "pixel_only"

    for w in parsed.get("warnings") or []:
        warnings.warn(w)

    print(
        f"[TaskPoints] 加载 {len(points)} 个任务点 "
        f"(start={parsed['start_count']}, goal={parsed['goal_count']}, "
        f"via={parsed['waypoint_count']})"
    )
    return points


def apply_lon_lat_swap(points: List[TaskPoint]) -> None:
    """交换每个点的经纬度（用户确认后调用）。"""
    for tp in points:
        if tp.longitude is None or tp.latitude is None:
            continue
        tp.longitude, tp.latitude = float(tp.latitude), float(tp.longitude)


def _read_with_fallback_encoding(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be")
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def get_plan_sequence(points: List[TaskPoint]) -> List[TaskPoint]:
    """严格按 seq 升序；point_type / 空间距离不参与排序。"""
    return sorted(list(points), key=lambda point: int(point.seq))


def normalize_task_point_sequence(points: List[TaskPoint]) -> List[TaskPoint]:
    """手动任务点：稳定归一化为 START → vias → GOAL 连续 seq。

    文件导入点（source=file_import）默认跳过重排，仅按 seq 排序，保留文件顺序。
    """
    target = points if isinstance(points, list) else list(points or [])
    items = list(target)
    if not items:
        return target

    # 文件导入主导：不按 type 重排、不重编号
    if all(getattr(p, "source", "") == "file_import" for p in items):
        items.sort(key=lambda p: int(p.seq))
        target[:] = items
        return target

    starts = [point for point in items if int(point.point_type) == 0]
    goals = [point for point in items if int(point.point_type) == 1]
    vias = [point for point in items if int(point.point_type) == 2]
    if len(starts) > 1:
        starts.sort(key=lambda p: (int(getattr(p, "created_order", 0)), int(p.seq)))
        for point in starts[:-1]:
            point.point_type = 2
            vias.append(point)
        starts = starts[-1:]
    if len(goals) > 1:
        goals.sort(key=lambda p: (int(getattr(p, "created_order", 0)), int(p.seq)))
        for point in goals[:-1]:
            point.point_type = 2
            vias.append(point)
        goals = goals[-1:]
    vias.sort(key=lambda p: (int(p.seq), int(getattr(p, "created_order", 0))))
    ordered = starts + vias + goals
    for seq, point in enumerate(ordered, 1):
        point.seq = seq
    target[:] = ordered
    return target


def validate_task_points_for_planning(points: List[TaskPoint]) -> List[str]:
    """规划/导出阶段验证。"""
    points = list(points or [])
    errors = []
    if len(points) < 2:
        errors.append("至少需要两个任务点")
    starts = [point for point in points if int(point.point_type) == 0]
    goals = [point for point in points if int(point.point_type) == 1]
    if len(starts) != 1:
        errors.append(f"必须有且只有一个起点，当前为 {len(starts)} 个")
    if len(goals) != 1:
        errors.append(f"必须有且只有一个终点，当前为 {len(goals)} 个")
    seqs = [int(point.seq) for point in points]
    if len(seqs) != len(set(seqs)):
        errors.append("任务点 seq 存在重复值")
    # 文件导入允许非 1..N 连续；手动点仍建议连续
    file_only = bool(points) and all(
        getattr(p, "source", "") == "file_import" for p in points
    )
    if not file_only and sorted(seqs) != list(range(1, len(points) + 1)):
        errors.append("任务点 seq 不连续，可使用“重新编号”修复")
    return errors


# ===================================================================
# 保存
# ===================================================================

@dataclass
class SnappedTaskPoint:
    """吸附后的任务点"""
    seq: int
    point_type: int
    original_x: float
    original_y: float
    snapped_x: float
    snapped_y: float
    snap_distance: float
    edge_id: Optional[int] = None
    node_id: Optional[int] = None
    virtual_node_id: Optional[str] = None
    snap_method: str = "none"
    status: str = "pending"
    warning: str = ""

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "point_type": self.point_type,
            "original": [round(self.original_x, 2), round(self.original_y, 2)],
            "snapped": [round(self.snapped_x, 2), round(self.snapped_y, 2)],
            "snap_distance": round(self.snap_distance, 2),
            "edge_id": self.edge_id,
            "node_id": self.node_id,
            "virtual_node_id": self.virtual_node_id,
            "snap_method": self.snap_method,
            "status": self.status,
            "warning": self.warning,
        }


def save_snapped_results(snapped: List[SnappedTaskPoint], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "task_points_snapped.json")
    data = {"task_points": [s.to_dict() for s in snapped]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[TaskPoints] 吸附结果已保存: {path}")
    return path


def save_task_points_loaded(points: List[TaskPoint], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "task_points_loaded.json")
    data = {"task_points": [tp.to_dict() for tp in points]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[TaskPoints] 原始任务点已保存: {path}")
    return path


def save_task_points_import_report(report: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "task_points_import_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
