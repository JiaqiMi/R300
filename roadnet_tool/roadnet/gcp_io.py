"""
GCP 坐标文件导入模块

支持 TXT / CSV / JSON 多种格式的 GCP 控制点导入。
优先保证 TXT 格式解析稳定。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 角点名归一化 — 支持多种写法
# ============================================================================

_CORNER_NAME_ALIASES: Dict[str, str] = {
    # 左上
    "top_left":     "top_left",
    "left_top":     "top_left",
    "tl":           "top_left",
    "lt":           "top_left",
    "左上":          "top_left",
    "左上角":         "top_left",
    "topleft":      "top_left",
    "lefttop":      "top_left",
    "t_l":          "top_left",
    "l_t":          "top_left",
    # 右上
    "top_right":    "top_right",
    "right_top":    "top_right",
    "tr":           "top_right",
    "rt":           "top_right",
    "右上":          "top_right",
    "右上角":         "top_right",
    "topright":     "top_right",
    "righttop":     "top_right",
    "t_r":          "top_right",
    "r_t":          "top_right",
    # 左下
    "bottom_left":  "bottom_left",
    "left_bottom":  "bottom_left",
    "bl":           "bottom_left",
    "lb":           "bottom_left",
    "左下":          "bottom_left",
    "左下角":         "bottom_left",
    "bottomleft":   "bottom_left",
    "leftbottom":   "bottom_left",
    "b_l":          "bottom_left",
    "l_b":          "bottom_left",
    # 右下
    "bottom_right": "bottom_right",
    "right_bottom": "bottom_right",
    "br":           "bottom_right",
    "rb":           "bottom_right",
    "右下":          "bottom_right",
    "右下角":         "bottom_right",
    "bottomright":  "bottom_right",
    "rightbottom":  "bottom_right",
    "b_r":          "bottom_right",
    "r_b":          "bottom_right",
}


def normalize_corner_name(name: str) -> Optional[str]:
    """将各种写法归一化为标准角点名 (top_left/top_right/bottom_left/bottom_right)。

    不识别则返回 None。
    """
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    # 先查别名字典
    if cleaned in _CORNER_NAME_ALIASES:
        return _CORNER_NAME_ALIASES[cleaned]
    # 模糊去标点
    cleaned2 = re.sub(r"[^a-z0-9\u4e00-\u9fff_]", "", cleaned)
    if cleaned2 in _CORNER_NAME_ALIASES:
        return _CORNER_NAME_ALIASES[cleaned2]
    return None


def infer_pixel_from_corner_name(name: str, image_width: int, image_height: int) -> Optional[Tuple[int, int]]:
    """根据标准角点名自动推断像素坐标。

    Returns:
        (u, v) 像素坐标，或 None（名称不识别）
    """
    mapping = {
        "top_left":     (0, 0),
        "top_right":    (image_width - 1, 0),
        "bottom_left":  (0, image_height - 1),
        "bottom_right": (image_width - 1, image_height - 1),
    }
    return mapping.get(name)


# ============================================================================
# TXT 解析
# ============================================================================

def _is_header_line(line: str) -> bool:
    """判断是否为表头行。"""
    s = line.strip().lower()
    indicators = ["point_name", "lon", "lat", "longitude", "latitude",
                   "名称", "经度", "纬度", "name", "pixel"]
    return any(ind in s for ind in indicators)


def _parse_float_pair(line: str) -> Optional[Tuple[float, float]]:
    """尝试从一行中提取两个浮点数（经度/纬度）。"""
    # 匹配各种分隔：空格、逗号、制表符
    parts = re.split(r"[,\s\t]+", line.strip())
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def load_gcp_txt(file_path: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """从 TXT 文件加载 GCP 控制点。

    支持格式:
      1. top_left 105.123456 38.123456
      2. top_left,105.123456,38.123456
      3. 带表头: point_name,lon,lat
      4. 混合格式

    Args:
        file_path: TXT 文件路径
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        控制点列表 [{"name": "top_left", "pixel": [u,v], "lon": ..., "lat": ...}, ...]
        解析失败返回空列表
    """
    control_points: List[Dict[str, Any]] = []
    seen_corners: set = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[GCP_IO] 读取 TXT 失败: {e}")
        return []

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # 跳表头
        if _is_header_line(line):
            if not any(ch.isdigit() for ch in line):
                continue

        # 尝试拆分
        # 判断分隔符：优先逗号 > 制表符 > 空格
        if "\t" in line:
            parts = line.split("\t")
        elif "," in line:
            parts = line.split(",")
        else:
            parts = line.split()

        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 3:
            # 可能名字在前半部分
            lon_lat = _parse_float_pair(line)
            if lon_lat is None:
                print(f"[GCP_IO] 第 {line_num} 行格式错误（字段不足）: {line}")
                continue
            # 没有角点名，尝试从行中查找
            # 可能只是纯数字，跳过
            print(f"[GCP_IO] 第 {line_num} 行无法识别角点名，跳过: {line}")
            continue

        # 第一列为角点名
        raw_name = parts[0].strip()
        corner_name = normalize_corner_name(raw_name)
        if corner_name is None:
            print(f"[GCP_IO] 第 {line_num} 行角点名无法识别: '{raw_name}'，跳过")
            continue

        if corner_name in seen_corners:
            print(f"[GCP_IO] 第 {line_num} 行角点名重复: {corner_name}，跳过")
            continue

        # 提取 lon / lat
        lon, lat = None, None
        # 剩余字段中找两个数字
        for i in range(1, len(parts)):
            try:
                val = float(parts[i])
                if lon is None:
                    lon = val
                elif lat is None:
                    lat = val
                    break
            except ValueError:
                continue

        if lon is None or lat is None:
            print(f"[GCP_IO] 第 {line_num} 行经纬度解析失败: {line}")
            continue

        # 范围检查
        if abs(lon) > 180:
            print(f"[GCP_IO] 第 {line_num} 行经度异常 ({lon})，可能 lat/lon 顺序写反: {line}")
            # 交换试试
            if abs(lat) <= 180:
                print(f"[GCP_IO] 自动交换经纬度...")
                lon, lat = lat, lon
            else:
                continue
        if abs(lat) > 90:
            print(f"[GCP_IO] 第 {line_num} 行纬度异常 ({lat}): {line}")
            continue

        # 推断像素坐标
        pixel_coord = infer_pixel_from_corner_name(corner_name, image_width, image_height)
        if pixel_coord is None:
            print(f"[GCP_IO] 无法推断像素坐标: {corner_name}")
            pixel_coord = [0, 0]

        cp = {
            "name": corner_name,
            "pixel": [pixel_coord[0], pixel_coord[1]],
            "lon": round(lon, 8),
            "lat": round(lat, 8),
        }

        control_points.append(cp)
        seen_corners.add(corner_name)
        print(f"[GCP_IO] 解析控制点: {corner_name} pixel=({cp['pixel'][0]},{cp['pixel'][1]}) lon={lon} lat={lat}")

    return control_points


# ============================================================================
# CSV 解析
# ============================================================================

def load_gcp_csv(file_path: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """从 CSV 文件加载 GCP 控制点。

    表头需包含: point_name (或 name), lon (或 longitude), lat (或 latitude)
    可选: pixel_x, pixel_y
    """
    import csv

    control_points: List[Dict[str, Any]] = []
    seen_corners: set = set()

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []

            fn_lower = [h.strip().lower() for h in reader.fieldnames]

            for row_num, row in enumerate(reader, start=2):
                # 找角点名
                name = None
                for possible in ["point_name", "name", "corner", "corner_name", "id", "label"]:
                    if possible in fn_lower:
                        idx = fn_lower.index(possible)
                        name = list(row.values())[idx]
                        break

                if name is None:
                    # 尝试从第一列推断
                    name = list(row.values())[0]

                corner_name = normalize_corner_name(name.strip()) if name else None
                if corner_name is None:
                    print(f"[GCP_IO] CSV 第 {row_num} 行角点名无法识别: '{name}'")
                    continue
                if corner_name in seen_corners:
                    continue

                # 找 lon / lat
                lon, lat = None, None
                for possible_lon in ["lon", "longitude", "经度", "lng"]:
                    if possible_lon in fn_lower:
                        idx = fn_lower.index(possible_lon)
                        try:
                            lon = float(list(row.values())[idx])
                        except (ValueError, IndexError):
                            pass
                        break

                for possible_lat in ["lat", "latitude", "纬度"]:
                    if possible_lat in fn_lower:
                        idx = fn_lower.index(possible_lat)
                        try:
                            lat = float(list(row.values())[idx])
                        except (ValueError, IndexError):
                            pass
                        break

                if lon is None or lat is None:
                    print(f"[GCP_IO] CSV 第 {row_num} 行经纬度解析失败")
                    continue

                # 像素坐标（可选，从 CSV 读取）
                pixel_coord = infer_pixel_from_corner_name(corner_name, image_width, image_height)
                for px_col in ["pixel_x", "u", "x_pixel"]:
                    if px_col in fn_lower:
                        idx = fn_lower.index(px_col)
                        try:
                            px_x = int(float(list(row.values())[idx]))
                            pixel_coord = [px_x, pixel_coord[1] if pixel_coord else 0]
                        except Exception:
                            pass
                for py_col in ["pixel_y", "v", "y_pixel"]:
                    if py_col in fn_lower:
                        idx = fn_lower.index(py_col)
                        try:
                            py_y = int(float(list(row.values())[idx]))
                            pixel_coord = [pixel_coord[0] if pixel_coord else 0, py_y]
                        except Exception:
                            pass

                cp = {
                    "name": corner_name,
                    "pixel": list(pixel_coord) if pixel_coord else [0, 0],
                    "lon": round(lon, 8),
                    "lat": round(lat, 8),
                }
                control_points.append(cp)
                seen_corners.add(corner_name)

    except Exception as e:
        print(f"[GCP_IO] CSV 解析失败: {e}")
        return []

    return control_points


# ============================================================================
# JSON 解析
# ============================================================================

def load_gcp_json(file_path: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """从 JSON 文件加载 GCP 控制点。

    支持三种格式:

    格式 1 — control_points 列表:
    {
      "control_points": [
        {"name": "top_left", "pixel": [0, 0], "lon": 105.123, "lat": 38.123},
        ...
      ]
    }

    格式 2 — corners_wgs84 字典（键名 → [lon, lat]）:
    {
      "corners_wgs84": {
        "top_left": [117.123456, 31.123456],
        "top_right": [117.223456, 31.123456],
        "bottom_left": [117.123456, 31.023456],
        "bottom_right": [117.223456, 31.023456]
      }
    }

    格式 3 — 直接数组:
    [
        {"name": "top_left", "lon": 105.123, "lat": 38.123},
        ...
    ]
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[GCP_IO] JSON 解析失败: {e}")
        return []

    # ★ 格式 2: corners_wgs84 字典（{name: [lon, lat]}）
    if isinstance(data, dict) and "corners_wgs84" in data:
        corners = data["corners_wgs84"]
        if isinstance(corners, dict):
            control_points = []
            seen_corners = set()
            for raw_name, coords in corners.items():
                name = normalize_corner_name(raw_name)
                if name is None or name in seen_corners:
                    print(f"[GCP_IO] 跳过无法识别/重复的角点: {raw_name}")
                    continue
                if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                    print(f"[GCP_IO] {raw_name} 坐标格式错误，需要 [lon, lat]")
                    continue
                lon, lat = float(coords[0]), float(coords[1])

                pixel = infer_pixel_from_corner_name(name, image_width, image_height)
                if pixel is None:
                    pixel = [0, 0]

                control_points.append({
                    "name": name,
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "lon": round(lon, 8),
                    "lat": round(lat, 8),
                })
                seen_corners.add(name)

            if control_points:
                print(f"[GCP_IO] 从 corners_wgs84 解析 {len(control_points)} 个控制点")
            return control_points

    # 格式 1 & 3: control_points 或直接数组
    raw_points = data.get("control_points", data.get("points", data.get("gcp", [])))
    if isinstance(data, list):
        raw_points = data

    if not isinstance(raw_points, list):
        return []

    control_points = []
    seen_corners = set()
    for cp in raw_points:
        name = normalize_corner_name(cp.get("name", cp.get("id", "")))
        if name is None:
            continue
        if name in seen_corners:
            continue

        pixel = cp.get("pixel")
        if pixel is None or len(pixel) < 2:
            pixel = infer_pixel_from_corner_name(name, image_width, image_height)
            if pixel is None:
                pixel = [0, 0]
            else:
                pixel = list(pixel)

        lon = cp.get("lon", cp.get("longitude", cp.get("lng", None)))
        lat = cp.get("lat", cp.get("latitude", None))
        if lon is None or lat is None:
            continue

        control_points.append({
            "name": name,
            "pixel": [int(pixel[0]), int(pixel[1])],
            "lon": round(float(lon), 8),
            "lat": round(float(lat), 8),
        })
        seen_corners.add(name)

    return control_points


# ============================================================================
# 通用控制点解析（任意点名，不只 TL/TR/BL/BR）
# ============================================================================

def load_generic_gcp_file(file_path: str) -> List[Dict[str, Any]]:
    """解析通用控制点文件（支持任意点名）。

    支持格式:
    1. 点号, 像素X, 像素Y, 经度, 纬度 → 直接返回完整控制点
    2. 点号, 经度, 纬度 → 仅 lon/lat，pixel=None（需图上配准）
    3. CSV 带表头 (name/id, pixel_x/px, pixel_y/py, lon/longitude, lat/latitude)

    Args:
        file_path: 文件路径

    Returns:
        控制点列表 [{"name": "CP1", "pixel": [px, py] or None, "lon": ..., "lat": ...}, ...]
    """
    control_points: List[Dict[str, Any]] = []

    # 读取文件
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            text = f.read()
    except Exception as e:
        print(f"[GCP_IO] 读取通用控制点文件失败: {e}")
        return []

    lines = [l.strip() for l in text.split("\n") if l.strip()
             and not l.strip().startswith("#") and not l.strip().startswith("//")]

    if not lines:
        return []

    # 检测分隔符
    first_line = lines[0]
    if "\t" in first_line:
        delimiter = "\t"
    elif "," in first_line:
        delimiter = ","
    else:
        delimiter = None  # 空格分隔

    # 尝试 CSV (带表头 / 逗号)
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv" or (delimiter == "," and _is_header_line(first_line)):
        return _parse_generic_gcp_csv(text)
    elif delimiter == ",":
        return _parse_generic_gcp_delimited(lines, ",")
    elif delimiter == "\t":
        return _parse_generic_gcp_delimited(lines, "\t")
    else:
        return _parse_generic_gcp_delimited(lines, None)


def _parse_generic_gcp_csv(text: str) -> List[Dict[str, Any]]:
    """解析 CSV 格式的通用控制点。"""
    import csv
    import io

    control_points: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return []

    fn_lower = [h.strip().lower() for h in reader.fieldnames]

    for row_num, row in enumerate(reader, start=2):
        # 找点名
        name = None
        for possible in ["name", "id", "point_name", "station", "label", "点号", "点名"]:
            if possible in fn_lower:
                idx = fn_lower.index(possible)
                name = list(row.values())[idx].strip()
                break
        if name is None:
            name = f"CP{row_num - 1}"

        # 找 lon/lat
        lon, lat = None, None
        for plon in ["lon", "longitude", "lng", "经度", "x"]:
            if plon in fn_lower:
                try:
                    lon = float(list(row.values())[fn_lower.index(plon)])
                except (ValueError, IndexError):
                    pass
                break
        for plat in ["lat", "latitude", "纬度", "y"]:
            if plat in fn_lower:
                try:
                    lat = float(list(row.values())[fn_lower.index(plat)])
                except (ValueError, IndexError):
                    pass
                break

        if lon is None or lat is None:
            print(f"[GCP_IO] 通用CP CSV 第 {row_num} 行经纬度解析失败")
            continue

        # 找像素坐标（可选）
        pixel = None
        px, py = None, None
        for pcol in ["pixel_x", "px", "u", "pixelx"]:
            if pcol in fn_lower:
                try: px = float(list(row.values())[fn_lower.index(pcol)])
                except: pass
                break
        for pcol in ["pixel_y", "py", "v", "pixely"]:
            if pcol in fn_lower:
                try: py = float(list(row.values())[fn_lower.index(pcol)])
                except: pass
                break
        if px is not None and py is not None:
            pixel = [px, py]
        elif px is not None:
            pixel = [px, None]
        elif py is not None:
            pixel = [None, py]

        cp = {
            "name": name,
            "pixel": pixel,
            "lon": round(float(lon), 8),
            "lat": round(float(lat), 8),
        }
        control_points.append(cp)
        print(f"[GCP_IO] 通用CP CSV: {name}, pixel={pixel}, lon={lon:.6f}, lat={lat:.6f}")

    return control_points


def _parse_generic_gcp_delimited(lines: List[str], delimiter: str | None) -> List[Dict[str, Any]]:
    """解析分隔格式的通用控制点（含空格分隔）。"""
    control_points: List[Dict[str, Any]] = []

    # 跳表头
    start_idx = 0
    if _is_header_line(lines[0]):
        start_idx = 1

    for i, line in enumerate(lines[start_idx:], start=start_idx + 1):
        if delimiter:
            parts = [p.strip() for p in line.split(delimiter) if p.strip()]
        else:
            parts = line.split()

        # 提取浮点数
        nums = []
        names = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                names.append(p)

        n_nums = len(nums)
        if n_nums < 2:
            print(f"[GCP_IO] 通用CP 第 {i} 行数值不足，跳过: {line}")
            continue

        # 确定点名
        if names:
            name = names[0]
        else:
            name = f"CP{i}"

        # 根据数值个数判断格式
        if n_nums >= 4:
            # 格式: 像素X, 像素Y, 经度, 纬度
            # 假设中间两个是像素（几百~几千），外侧两个是经纬度（大数）
            # 更可靠：如果前两个 < 50000 则为像素坐标，后两个为经纬度
            px, py, lon, lat = nums[0], nums[1], nums[2], nums[3]
            if px > 180 or py > 90:
                # 可能像素和经纬度顺序反了，尝试: lon, lat, px, py
                lon2, lat2, px2, py2 = nums[0], nums[1], nums[2], nums[3]
                if lon2 <= 180 and lat2 <= 90:
                    lon, lat, px, py = lon2, lat2, px2, py2
            pixel = [px, py]
        elif n_nums == 3:
            # 格式: 点号(可能只有数字), 经度, 纬度  → 仅 lon/lat
            # 或是: 像素X, 像素Y, 经度  → 缺lat
            a, b, c = nums
            if abs(a) <= 180 and abs(b) <= 90:
                # a,b 像经纬度, c可能是纬度偏移 → 假设: lon=a, lat=b, pixel=None
                lon, lat = a, b
                pixel = None
                if name.startswith("CP") and abs(c) <= 90:
                    # c也是纬度候选 → 保留a,b
                    pass
            else:
                lon, lat = b, c
                pixel = None
        elif n_nums == 2:
            # 格式: 经度, 纬度
            lon, lat = nums[0], nums[1]
            pixel = None
        else:
            # n_nums == 5+: 点号数字, 像素X, 像素Y, 经度, 纬度
            if not names and n_nums >= 5:
                # 第一个数字可能是序号
                px, py = nums[0], nums[1]
                lon, lat = nums[2], nums[3]
                if px > 180 or py > 90:
                    # 尝试跳过第一个序号: 序号, px, py, lon, lat
                    if n_nums >= 5:
                        px, py, lon, lat = nums[1], nums[2], nums[3], nums[4]
                pixel = [px, py]
                name = name if not name.startswith("CP") else f"CP{i}"
            else:
                lon, lat = nums[0], nums[1]
                pixel = None

        # 范围检查：检测 lon/lat 是否填反
        # 仅当第一个值像纬度（0~90）且第二个值像经度（>90）时才交换
        # 中国经度（如 117）是合法的经度，不应与纬度（如 31）交换
        if abs(lon) < 90 and abs(lat) > 90:
            # lon 像纬度，lat 像经度 → 交换
            print(f"[GCP_IO] 通用CP {name} 疑似 lon/lat 填反，自动交换: "
                  f"lon={lon:.6f} → {lat:.6f}, lat={lat:.6f} → {lon:.6f}")
            lon, lat = lat, lon
        elif abs(lon) > 90 and abs(lat) > 90:
            # 两个都像经度 → 可能有问题，但不自动交换
            print(f"[GCP_IO] 通用CP {name}: lon={lon:.6f} 和 lat={lat:.6f} 都 >90，"
                  f"请检查是否填写有误")

        cp = {
            "name": name,
            "pixel": pixel,
            "lon": round(float(lon), 8),
            "lat": round(float(lat), 8),
        }
        control_points.append(cp)
        px_str = f"({cp['pixel'][0]},{cp['pixel'][1]})" if cp['pixel'] else "None"
        print(f"[GCP_IO] 通用CP: {name}, pixel={px_str}, lon={lon:.6f}, lat={lat:.6f}")

    return control_points


# ============================================================================
# 比赛图片顶点校准文件：序号;经度;纬度;高程
# ============================================================================

# 固定序号 → (显示角点名, 内部标准角点名, pixel 计算)
# 1=左下 2=右下 3=左上；4=右上（可选兼容）
VERTEX_ID_CORNER_MAP = {
    1: ("left_bottom", "bottom_left"),
    2: ("right_bottom", "bottom_right"),
    3: ("left_top", "top_left"),
    4: ("right_top", "top_right"),
}


def _strip_line_comment(line: str) -> str:
    """去掉行尾 // 或 # 注释。"""
    # 优先处理 //，再处理 #
    for marker in ("//", "#"):
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def _split_calibration_fields(line: str) -> List[str]:
    """按 ; ； , tab 或空白拆分字段。"""
    # 统一中文分号
    s = line.replace("；", ";")
    if ";" in s:
        parts = s.split(";")
    elif "," in s:
        parts = s.split(",")
    elif "\t" in s:
        parts = s.split("\t")
    else:
        parts = re.split(r"\s+", s.strip())
    return [p.strip() for p in parts if p.strip()]


def looks_like_corner_calibration_txt(file_path: str) -> bool:
    """启发式判断是否为「序号;经度;纬度;高程」顶点校准文件。"""
    try:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                lines = f.readlines()
    except Exception:
        return False
    id_hits = 0
    for raw in lines:
        line = _strip_line_comment(raw)
        if not line:
            continue
        parts = _split_calibration_fields(line)
        if len(parts) < 3:
            continue
        try:
            vid = int(float(parts[0]))
        except ValueError:
            continue
        if vid in (1, 2, 3, 4):
            try:
                float(parts[1])
                float(parts[2])
                id_hits += 1
            except ValueError:
                continue
    return id_hits >= 3


def parse_corner_calibration_txt(file_path: str) -> Dict[str, Any]:
    """解析比赛提供的图片顶点校准坐标文件。

    格式：序号;经度;纬度;高程
      1 = 左下角, 2 = 右下角, 3 = 左上角
      可选 4 = 右上角

    支持分隔符：; ； , tab 空格；支持行尾 // 与 # 注释。

    Returns:
        {
          "ok": bool,
          "error": str | None,
          "source_file": str,
          "records": [
            {"id": 1, "longitude": ..., "latitude": ..., "altitude": 0,
             "corner": "left_bottom", "name": "bottom_left"},
            ...
          ],
          "swap_suspect": bool,
        }
    """
    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "source_file": os.path.basename(file_path),
        "records": [],
        "swap_suspect": False,
    }
    try:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                lines = f.readlines()
    except Exception as exc:
        result["error"] = f"无法读取文件：{exc}"
        return result

    by_id: Dict[int, Dict[str, Any]] = {}
    for line_num, raw in enumerate(lines, start=1):
        line = _strip_line_comment(raw)
        if not line:
            continue
        parts = _split_calibration_fields(line)
        if len(parts) < 3:
            # 可能是表头
            if _is_header_line(line):
                continue
            result["error"] = "校准文件格式错误，应为：序号;经度;纬度;高程。"
            return result
        try:
            vid = int(float(parts[0]))
            lon = float(parts[1])
            lat = float(parts[2])
        except ValueError:
            if _is_header_line(line):
                continue
            result["error"] = "校准文件格式错误，应为：序号;经度;纬度;高程。"
            return result
        alt = 0.0
        if len(parts) >= 4:
            try:
                alt = float(parts[4 - 1])
            except ValueError:
                alt = 0.0

        if vid not in VERTEX_ID_CORNER_MAP:
            # 兼容其它序号，忽略但不中断
            print(f"[GCP_IO] 顶点校准：忽略未定义序号 {vid} @ line {line_num}")
            continue

        corner_label, std_name = VERTEX_ID_CORNER_MAP[vid]
        # 范围 / 填反嫌疑（不静默交换）
        if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
            # 若交换后合法，标记嫌疑
            if (-180.0 <= lat <= 180.0) and (-90.0 <= lon <= 90.0):
                result["swap_suspect"] = True
            else:
                result["error"] = (
                    f"第 {line_num} 行经纬度超出范围：lon={lon}, lat={lat}。"
                    "经度应在 [-180,180]，纬度应在 [-90,90]。"
                )
                return result
        elif -90.0 <= lon <= 90.0 and 100.0 <= abs(lat) <= 180.0:
            # lat 看似经度、lon 看似纬度
            result["swap_suspect"] = True

        by_id[vid] = {
            "id": vid,
            "longitude": lon,
            "latitude": lat,
            "altitude": alt,
            "corner": corner_label,
            "name": std_name,
        }

    for required in (1, 2, 3):
        if required not in by_id:
            result["error"] = "顶点校准文件必须包含 1=左下角、2=右下角、3=左上角。"
            return result

    # 按 1,2,3,(4) 顺序
    records = [by_id[i] for i in (1, 2, 3) if i in by_id]
    if 4 in by_id:
        records.append(by_id[4])
    result["records"] = records
    result["ok"] = True
    return result


def build_control_points_from_corner_records(
    records: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    swap_lon_lat: bool = False,
) -> List[Dict[str, Any]]:
    """将顶点校准记录绑定到图像像素控制点。"""
    w = int(image_width)
    h = int(image_height)
    cps: List[Dict[str, Any]] = []
    for rec in records:
        name = rec["name"]
        pixel = infer_pixel_from_corner_name(name, w, h)
        if pixel is None:
            continue
        lon = float(rec["longitude"])
        lat = float(rec["latitude"])
        if swap_lon_lat:
            lon, lat = lat, lon
        cps.append({
            "name": name,
            "pixel": [int(pixel[0]), int(pixel[1])],
            "lon": round(lon, 8),
            "lat": round(lat, 8),
            "altitude": float(rec.get("altitude", 0) or 0),
            "corner_id": str(rec.get("id", "")),
            "corner": rec.get("corner", ""),
        })
    return cps


# ============================================================================
# 统一入口
# ============================================================================

def load_gcp_file(file_path: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """根据文件扩展名自动选择解析器。

    返回:
        控制点列表，解析失败返回空列表。
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        # 优先尝试比赛顶点校准格式
        if looks_like_corner_calibration_txt(file_path):
            parsed = parse_corner_calibration_txt(file_path)
            if parsed.get("ok"):
                return build_control_points_from_corner_records(
                    parsed["records"], image_width, image_height,
                )
        return load_gcp_txt(file_path, image_width, image_height)
    elif ext == ".csv":
        return load_gcp_csv(file_path, image_width, image_height)
    elif ext == ".json":
        return load_gcp_json(file_path, image_width, image_height)
    else:
        # 未知扩展名，先尝试 TXT
        print(f"[GCP_IO] 未知文件扩展名 {ext}，尝试按 TXT 解析。")
        return load_gcp_txt(file_path, image_width, image_height)


# ============================================================================
# 验证工具（在 loaded GCP 和 image_size 已知时调用）
# ============================================================================

def validate_gcp_points(control_points: List[Dict[str, Any]],
                         image_width: int, image_height: int) -> Tuple[bool, str]:
    """在导入后验证控制点。

    Returns:
        (is_valid, error_message)
    """
    if not control_points:
        return False, "没有有效控制点。"

    if len(control_points) < 3:
        return False, f"控制点数量不足：需要至少 3 个，当前只有 {len(control_points)} 个。"

    # 检查每个点
    for cp in control_points:
        name = cp.get("name", "?")
        pixel = cp.get("pixel", [])
        lon = cp.get("lon", 0)
        lat = cp.get("lat", 0)

        if not (-180 <= lon <= 180):
            return False, f"控制点 {name} 经度 {lon} 超出范围 [-180, 180]。"
        if not (-90 <= lat <= 90):
            return False, f"控制点 {name} 纬度 {lat} 超出范围 [-90, 90]。可能是 lon/lat 顺序写反？"

        if len(pixel) >= 2:
            u, v = pixel[0], pixel[1]
            if u < 0 or u >= image_width:
                return False, f"控制点 {name} 像素 x={u} 超出图像范围 [0, {image_width-1}]。"
            if v < 0 or v >= image_height:
                return False, f"控制点 {name} 像素 y={v} 超出图像范围 [0, {image_height-1}]。"

    # 检查像素点共线
    if len(control_points) >= 3:
        pts = control_points[:3]
        u1, v1 = pts[0]["pixel"]
        u2, v2 = pts[1]["pixel"]
        u3, v3 = pts[2]["pixel"]
        area = abs((u2 - u1) * (v3 - v1) - (u3 - u1) * (v2 - v1))
        if area < 1.0:
            return False, "三个像素控制点近似共线，无法计算仿射变换。"

    return True, ""


# ============================================================================
# 角点默认像素坐标（四角定义）
# ============================================================================

CORNERS_DEF = {
    "top_left":     {"label": "左上 (TL)", "pixel_fn": lambda w, h: (0, 0)},
    "top_right":    {"label": "右上 (TR)", "pixel_fn": lambda w, h: (w - 1, 0)},
    "bottom_left":  {"label": "左下 (BL)", "pixel_fn": lambda w, h: (0, h - 1)},
    "bottom_right": {"label": "右下 (BR)", "pixel_fn": lambda w, h: (w - 1, h - 1)},
}

CORNERS_ORDER = ["top_left", "top_right", "bottom_left", "bottom_right"]
