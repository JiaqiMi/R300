"""Persistent large-image project and tile-index primitives.

All geometry in this module is expressed in original image pixels.  Preview
coordinates are metadata for display only and are never persisted as editing
geometry.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np


LARGE_IMAGE_THRESHOLD_DIM = 4096
LARGE_IMAGE_THRESHOLD_PIXELS = 16_000_000      # width * height > 16M
DEFAULT_TILE_SIZE = 2048
DEFAULT_TILE_OVERLAP = 256
DEFAULT_PREVIEW_MAX_SIDE = 3000
MAX_SAFE_RGB_MB = 500                           # 超过 500MB 禁止整图 RGB 加载
MAX_CRITICAL_RGB_MB = 2000                      # 超过 2GB 拒绝打开（blocked）


# ============================================================================
# 统一风险等级枚举
# ============================================================================

RISK_SAFE = "safe"           # 正常尺寸，可以全图加载
RISK_WARNING = "warning"     # 较大图片，建议大图模式
RISK_HIGH_RISK = "high_risk" # 大图，强制 preview-only 模式
RISK_BLOCKED = "blocked"     # 超大/异常，拒绝打开

VALID_RISK_LEVELS = {RISK_SAFE, RISK_WARNING, RISK_HIGH_RISK, RISK_BLOCKED}

RISK_MESSAGES = {
    RISK_SAFE: "图像尺寸正常，可以打开。",
    RISK_WARNING: "图像较大，建议使用大图模式。",
    RISK_HIGH_RISK: "图像较大，已强制进入大图预览模式，不会加载原图到画布。",
    RISK_BLOCKED: "图像过大或格式异常，无法安全生成预览。",
}

RISK_POLICIES = {
    RISK_SAFE: {
        "large_image_mode": False,
        "should_load_full_pixmap": True,
        "should_generate_preview": True,
        "may_continue": True,
    },
    RISK_WARNING: {
        "large_image_mode": True,
        "should_load_full_pixmap": False,
        "should_generate_preview": True,
        "may_continue": True,
    },
    RISK_HIGH_RISK: {
        "large_image_mode": True,
        "should_load_full_pixmap": False,
        "should_generate_preview": True,
        "may_continue": True,
    },
    RISK_BLOCKED: {
        "large_image_mode": False,
        "should_load_full_pixmap": False,
        "should_generate_preview": False,
        "may_continue": False,
    },
}


# ============================================================================
# 安全尺寸读取：只读文件头，不解码像素
# ============================================================================

def get_image_size_safe(image_path: str) -> tuple[int, int]:
    """只通过 PIL header 读取图片宽高，不解码像素数据。

    Returns (width, height) in pixels. 失败时抛异常，调用方必须 try/except。
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as img:
        w, h = map(int, img.size)
    if w <= 0 or h <= 0:
        raise ValueError(f"无效影像尺寸: {w}x{h}, path={image_path}")
    return w, h


def is_large_image(width: int, height: int) -> bool:
    """判断是否应进入大图模式。
    条件: width>4096 或 height>4096 或 width*height>16M
    """
    if width > LARGE_IMAGE_THRESHOLD_DIM or height > LARGE_IMAGE_THRESHOLD_DIM:
        return True
    return width * height > LARGE_IMAGE_THRESHOLD_PIXELS


def estimate_image_memory_mb(width: int, height: int) -> dict:
    """估算图片内存占用量（MB），不做实际加载。"""
    rgb_mb = width * height * 3 / (1024.0 * 1024.0)
    rgba_mb = width * height * 4 / (1024.0 * 1024.0)
    return {
        "raw_rgb_mb": round(rgb_mb, 2),
        "rgba_overlay_mb": round(rgba_mb, 2),
        "width": width,
        "height": height,
    }


def determine_risk_level(width: int, height: int, raw_rgb_mb: Optional[float] = None) -> str:
    """根据图像尺寸和内存估算，返回统一的风险等级。

    返回 RISK_SAFE / RISK_WARNING / RISK_HIGH_RISK / RISK_BLOCKED 之一。
    不会返回其他值。

    Args:
        width: 图像宽度
        height: 图像高度
        raw_rgb_mb: 预计算的内存估算值，如果为 None 则内部计算
    """
    if raw_rgb_mb is None:
        raw_rgb_mb = width * height * 3 / (1024.0 * 1024.0)

    # 超大：超过 2GB RGB 内存，拒绝打开
    if raw_rgb_mb > MAX_CRITICAL_RGB_MB:
        return RISK_BLOCKED

    # 大图：超过 500MB RGB 内存，强制 preview-only
    if raw_rgb_mb > MAX_SAFE_RGB_MB:
        return RISK_HIGH_RISK

    # 较大图片：dim > 4096 或 pixels > 16M
    if is_large_image(width, height):
        return RISK_WARNING

    return RISK_SAFE


def check_memory_budget(width: int, height: int) -> dict:
    """检查内存预算，返回风险报告。

    使用统一风险等级枚举。

    Returns dict containing:
        raw_rgb_mb, rgba_overlay_mb, width, height,
        risk_level (safe/warning/high_risk/blocked),
        max_safe_mb, warning message, policy dict
    """
    mem = estimate_image_memory_mb(width, height)
    risk_level = determine_risk_level(width, height, mem["raw_rgb_mb"])
    policy = RISK_POLICIES.get(risk_level, RISK_POLICIES[RISK_HIGH_RISK])
    mem["risk_level"] = risk_level
    mem["max_safe_mb"] = MAX_SAFE_RGB_MB
    mem["max_critical_mb"] = MAX_CRITICAL_RGB_MB
    mem["policy"] = policy
    mem["warning"] = RISK_MESSAGES.get(risk_level, RISK_MESSAGES[RISK_HIGH_RISK])
    # backward-compat: 旧代码可能检查 mem["high_risk"]
    mem["high_risk"] = (risk_level == RISK_HIGH_RISK)
    return mem


# ============================================================================
# 安全 Preview 生成：thumbnail first，绝不先 img.convert("RGB")
# ============================================================================

def generate_preview_safe(image_path: str, preview_path: str,
                          max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> np.ndarray:
    """安全生成预览图，绝不先完整加载原图为 RGB。

    1. 使用 PIL thumbnail() 在原始 mode 上降采样（渐进式解码）
    2. 降采样后再 convert("RGB")
    3. 保存 preview.png

    Returns preview numpy array (RGB). 失败时抛异常。
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    try:
        with Image.open(image_path) as img:
            # 关键：先 thumbnail 再 convert，避免全图 RGB 解码
            resized = img.copy()  # loads in original mode (often 1 or 3 band)
            resized.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
            rgb = resized.convert("RGB")
            preview = np.asarray(rgb, dtype=np.uint8)
    except Exception:
        # 降级：逐行读取
        try:
            preview = _generate_preview_line_by_line(image_path, max_side)
        except Exception:
            # 再降级：尝试 draft mode
            with Image.open(image_path) as img:
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")
                scale = min(max_side / max(img.size), 1.0)
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img.draft(img.mode, new_size)
                rgb = img.convert("RGB")
                preview = np.asarray(rgb, dtype=np.uint8)

    # 保存到磁盘
    _write_image(preview_path, preview)
    return preview


def _generate_preview_line_by_line(image_path: str, max_side: int) -> np.ndarray:
    """逐行降采样预览图生成，适用于超大 JPG/PNG。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as img:
        w, h = img.size

    scale = min(max_side / max(w, h), 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    from PIL import Image
    import numpy as np
    Image.MAX_IMAGE_PIXELS = None

    preview = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # 逐行重采样因子（整数步长）
    y_step = max(1, int(h / new_h))
    x_step = max(1, int(w / new_w))

    with Image.open(image_path) as img:
        for py in range(new_h):
            y0 = py * y_step
            y1 = min(h, y0 + y_step)
            if y1 <= y0:
                continue
            row_strip = img.crop((0, y0, w, y1))
            if row_strip.mode != "RGB":
                row_strip = row_strip.convert("RGB")
            row_arr = np.asarray(row_strip, dtype=np.uint8)
            # 水平降采样
            for px in range(new_w):
                x0_src = px * x_step
                x1_src = min(w, x0_src + x_step)
                if x1_src <= x0_src:
                    continue
                preview[py, px] = row_arr[:, x0_src:x1_src, :].mean(axis=1).mean(axis=0).astype(np.uint8)

    return preview


# ============================================================================
# 大图打开日志
# ============================================================================

def write_open_large_image_log(project_dir: str, image_path: str,
                               file_size_mb: float,
                               width: int, height: int,
                               preview_path: str,
                               preview_width: int, preview_height: int,
                               preview_scale: float,
                               memory_before_mb: Optional[float] = None,
                               memory_after_mb: Optional[float] = None,
                               steps_completed: Optional[list] = None,
                               error_traceback: str = "",
                               risk_level: str = "",
                               raw_rgb_mb: Optional[float] = None,
                               stage: str = "") -> str:
    """写入大图打开日志。

    Logs include: risk_level, memory estimate, traceback.
    Returns log file path.
    """
    from datetime import datetime
    logs_dir = Path(project_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "open_large_image.log"

    lines = [
        f"=== Large Image Open Log ===",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"image_path: {image_path}",
        f"file_size_mb: {file_size_mb:.2f}" if file_size_mb else "file_size_mb: unknown",
        f"image_width: {width}",
        f"image_height: {height}",
        f"raw_rgb_mb: {raw_rgb_mb:.2f}" if raw_rgb_mb is not None else "raw_rgb_mb: unknown",
        f"risk_level: {risk_level}" if risk_level else "risk_level: unknown",
        f"stage: {stage}" if stage else "stage: unknown",
        f"large_image_mode: true",
        f"preview_path: {preview_path}",
        f"preview_width: {preview_width}",
        f"preview_height: {preview_height}",
        f"preview_scale: {preview_scale:.6f}",
        f"memory_before: {memory_before_mb}" if memory_before_mb is not None else "memory_before: unknown",
        f"memory_after: {memory_after_mb}" if memory_after_mb is not None else "memory_after: unknown",
        f"steps_completed: {steps_completed or []}",
        f"error: {error_traceback}" if error_traceback else "error: none",
        f"================================",
    ]

    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(log_path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _write_image(path, image: np.ndarray) -> None:
    """安全写图到磁盘。path 可以是 str 或 Path。"""
    path = Path(path) if not isinstance(path, Path) else path
    source = np.asarray(image)
    if source.ndim == 3 and source.shape[2] >= 3:
        source = cv2.cvtColor(source[:, :, :3], cv2.COLOR_RGB2BGR)
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, source)
    if not ok:
        raise IOError(f"无法编码图像: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


class ImageRegionReader:
    """Read image metadata, previews and regions without retaining a full RGB array.

    Pillow keeps the source file lazy and only materializes the requested crop.
    Some codecs may internally decode more data, but the full image is never kept
    in RoadNet Studio's data model.  OpenCV is a compatibility fallback.
    """

    def __init__(self, image_path: str):
        self.image_path = str(Path(image_path).resolve())
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"影像不存在: {self.image_path}")
        self.backend = "pillow_region"
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(self.image_path) as image:
                self.width, self.height = map(int, image.size)
        except Exception:
            self.backend = "opencv_fallback"
            image = cv2.imread(self.image_path, cv2.IMREAD_REDUCED_COLOR_8)
            if image is None:
                raise ValueError(f"无法读取影像: {self.image_path}")
            # Reduced reads cannot provide the original size reliably.  Use the
            # lightweight video/image header reader first.
            capture = cv2.VideoCapture(self.image_path)
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if capture.isOpened() else int(image.shape[1] * 8)
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if capture.isOpened() else int(image.shape[0] * 8)
            capture.release()
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"影像尺寸无效: {self.width}x{self.height}")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def read_preview(self, max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> np.ndarray:
        """安全生成预览图：(1) thumbnail 降采样 (2) 再 convert RGB。
        绝不先全图转换再降采样，避免大图 OOM。
        """
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.image_path) as image:
            # Step 1: 先在原始 mode 上降采样（渐进式解码，不一次性加载全图 RGB）
            # 如果原始 mode 不支持 thumbnail，copy 创建可编辑副本
            resized = image.copy()
            resized.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
            # Step 2: 降采样后再转 RGB
            rgb = resized.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8).copy()

    def read_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        x0 = max(0, min(self.width, int(x0)))
        y0 = max(0, min(self.height, int(y0)))
        x1 = max(x0, min(self.width, int(x1)))
        y1 = max(y0, min(self.height, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(self.image_path) as image:
                crop = image.crop((x0, y0, x1, y1)).convert("RGB")
                return np.asarray(crop, dtype=np.uint8).copy()
        except Exception:
            # Compatibility fallback.  It is transient and never stored on the
            # project/layer object, so memory is released after the crop.
            bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"无法读取影像区域: {self.image_path}")
            return cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    def read_pixels(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        values = []
        for point in points:
            x = max(0, min(self.width - 1, int(round(float(point[0])))))
            y = max(0, min(self.height - 1, int(round(float(point[1])))))
            pixel = self.read_region(x, y, x + 1, y + 1)
            if pixel.size:
                values.append(pixel[0, 0])
        return np.asarray(values, dtype=np.uint8).reshape((-1, 3))


@dataclass
class LargeImageTile:
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int
    valid: bool = True
    black_ratio: float = 0.0
    border_invalid_ratio: float = 0.0

    def rect(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass
class LargeImageProject:
    image_path: str
    image_width: int
    image_height: int
    preview_path: str
    preview_scale: float
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    tile_index_path: str = ""
    coordinate_system: str = "image_pixel"
    geo_calibration_path: str = ""
    global_mask_path: str = ""
    global_graph_path: str = ""
    project_dir: str = ""
    valid_image_mask_path: str = ""
    last_error_log: str = ""
    # ★ 大图 working mask 状态字段（唯一 current working mask 的持久化）
    global_road_mask_path: str = ""
    working_road_mask_path: str = ""
    edited_global_road_mask_path: str = ""
    working_road_mask_preview_path: str = ""
    refined_main_road_mask_path: str = ""
    cleaned_working_mask_path: str = ""
    cleaned_working_mask_preview_path: str = ""
    final_edited_mask_path: str = ""
    final_edited_mask_preview_path: str = ""
    main_road_seed_strokes_path: str = ""
    # mask_source: global_road_mask / cleaned_working_mask / manual_after_cleaned /
    #              final_edited_mask / ribbon_hole_gap_filled / manual_edited / working_road_mask
    mask_source: str = ""
    # mask_edit_base: 当前编辑基于哪一层（如 cleaned_working_mask / global_road_mask）
    mask_edit_base: str = ""
    mask_dirty: bool = False
    formal_ready: bool = False
    preview_only: bool = False
    # 低像素正式 Mask 元数据
    lowres_work_image_path: str = ""
    lowres_road_mask_path: str = ""
    lowres_width: int = 0
    lowres_height: int = 0
    scale_x: float = 0.0
    scale_y: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir) / "large_image_project.json"

    def save(self) -> str:
        payload = asdict(self)
        _atomic_json(self.project_path, payload)
        return str(self.project_path)

    @classmethod
    def load(cls, path: str) -> "LargeImageProject":
        source = Path(path).resolve()
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        payload.setdefault("project_dir", str(source.parent))
        return cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size/overlap 参数无效")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_tile_rects(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    return [
        (x0, y0, min(width, x0 + tile_size), min(height, y0 + tile_size))
        for y0 in _axis_starts(height, tile_size, overlap)
        for x0 in _axis_starts(width, tile_size, overlap)
    ]


def create_large_image_project(
    image_path: str,
    output_root: str,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    project_name: Optional[str] = None,
) -> LargeImageProject:
    reader = ImageRegionReader(image_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = project_name or f"project_{stamp}"
    project_dir = Path(output_root).resolve() / name
    suffix = 2
    base = project_dir
    while project_dir.exists():
        project_dir = Path(f"{base}_{suffix}")
        suffix += 1
    project_dir.mkdir(parents=True, exist_ok=False)
    for child in ("samroad_large", "masks", "skeleton", "graph", "path", "reports"):
        (project_dir / child).mkdir(exist_ok=True)

    preview = reader.read_preview(preview_max_side)
    preview_path = project_dir / "preview.png"
    _write_image(preview_path, preview)
    scale = min(preview.shape[1] / reader.width, preview.shape[0] / reader.height)
    project = LargeImageProject(
        image_path=str(Path(image_path).resolve()),
        image_width=reader.width,
        image_height=reader.height,
        preview_path=str(preview_path),
        preview_scale=float(scale),
        tile_size=int(tile_size),
        tile_overlap=int(tile_overlap),
        tile_index_path=str(project_dir / "tile_index.json"),
        project_dir=str(project_dir),
    )
    project.save()
    return project


def regenerate_project_preview(project: LargeImageProject,
                               max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> np.ndarray:
    reader = ImageRegionReader(project.image_path)
    preview = reader.read_preview(max_side)
    preview_path = Path(project.preview_path or Path(project.project_dir) / "preview.png")
    _write_image(preview_path, preview)
    project.preview_path = str(preview_path)
    project.preview_scale = float(min(
        preview.shape[1] / reader.width, preview.shape[0] / reader.height,
    ))
    project.save()
    return preview


def generate_tile_index(
    project: LargeImageProject,
    *,
    black_threshold: int = 10,
    black_ratio_threshold: float = 0.8,
    progress: Optional[Callable[[int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> dict:
    reader = ImageRegionReader(project.image_path)
    from roadnet.valid_image import analyze_valid_image_mask
    preview = reader.read_preview(DEFAULT_PREVIEW_MAX_SIDE)
    preview_valid, valid_report = analyze_valid_image_mask(
        preview, int(black_threshold),
        max(64, int(4096 * (preview.shape[1] / max(1, project.image_width)) ** 2)),
    )
    valid_preview_path = Path(project.project_dir) / "valid_image_mask_preview.png"
    _write_image(valid_preview_path, preview_valid)
    rects = build_tile_rects(
        project.image_width, project.image_height,
        project.tile_size, project.tile_overlap,
    )
    tiles = []
    for index, (x0, y0, x1, y1) in enumerate(rects, 1):
        if cancelled and cancelled():
            raise RuntimeError("用户取消 tile index 生成")
        image = reader.read_region(x0, y0, x1, y1)
        black = np.all(image[:, :, :3] < int(black_threshold), axis=2)
        black_ratio = float(np.count_nonzero(black)) / float(max(1, black.size))
        ph, pw = preview_valid.shape
        px0 = max(0, min(pw - 1, int(np.floor(x0 * pw / project.image_width))))
        py0 = max(0, min(ph - 1, int(np.floor(y0 * ph / project.image_height))))
        px1 = max(px0 + 1, min(pw, int(np.ceil(x1 * pw / project.image_width))))
        py1 = max(py0 + 1, min(ph, int(np.ceil(y1 * ph / project.image_height))))
        preview_tile_valid = preview_valid[py0:py1, px0:px1]
        border_invalid_ratio = 1.0 - (
            float(np.count_nonzero(preview_tile_valid))
            / float(max(1, preview_tile_valid.size))
        )
        tiles.append(LargeImageTile(
            tile_id=f"tile_{index:04d}", x0=x0, y0=y0, x1=x1, y1=y1,
            width=x1 - x0, height=y1 - y0,
            # Internal dark regions are intentionally retained.  A tile is
            # skipped only when the border-connected invalid region dominates.
            valid=border_invalid_ratio <= float(black_ratio_threshold),
            black_ratio=round(black_ratio, 6),
            border_invalid_ratio=round(border_invalid_ratio, 6),
        ))
        if progress:
            progress(index, len(rects))
    payload = {
        "image_path": project.image_path,
        "image_width": project.image_width,
        "image_height": project.image_height,
        "coordinate_system": "image_pixel",
        "tile_size": project.tile_size,
        "overlap": project.tile_overlap,
        "black_threshold": int(black_threshold),
        "black_ratio_threshold": float(black_ratio_threshold),
        "valid_mask_method": "border_connected_preview",
        "valid_image_mask_preview": str(valid_preview_path),
        "valid_mask_report": valid_report,
        "tile_count": len(tiles),
        "valid_tile_count": sum(tile.valid for tile in tiles),
        "skipped_black_tile_count": sum(not tile.valid for tile in tiles),
        "tiles": [asdict(tile) for tile in tiles],
    }
    tile_index_path = Path(project.tile_index_path or Path(project.project_dir) / "tile_index.json")
    _atomic_json(tile_index_path, payload)
    project.tile_index_path = str(tile_index_path)
    project.save()
    return payload


def load_tile_index(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def current_process_memory_mb() -> Optional[float]:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        return None


def large_image_self_check(project: LargeImageProject, *, mask_shape=None,
                           calibration=None, final_graph=None,
                           task_points: Optional[Iterable] = None) -> dict:
    tile_index = {}
    if project.tile_index_path and os.path.isfile(project.tile_index_path):
        tile_index = load_tile_index(project.tile_index_path)
    calibration_size = (
        getattr(calibration, "image_width", None),
        getattr(calibration, "image_height", None),
    ) if calibration is not None else (None, None)
    calibration_valid_value = getattr(calibration, "is_valid", False) if calibration is not None else False
    calibration_valid = bool(
        calibration_valid_value() if callable(calibration_valid_value) else calibration_valid_value
    )
    task_points = list(task_points or [])
    task_original = all(
        getattr(point, "pixel_x", None) is not None and getattr(point, "pixel_y", None) is not None
        if not isinstance(point, dict)
        else point.get("pixel_x") is not None and point.get("pixel_y") is not None
        for point in task_points
    )
    mask_size = None if mask_shape is None else [int(mask_shape[1]), int(mask_shape[0])]
    graph_exists = bool(final_graph and (
        (isinstance(final_graph, dict) and final_graph.get("nodes"))
        or getattr(final_graph, "nodes", None)
    ))
    report = {
        "image_size": [project.image_width, project.image_height],
        "preview_size": None,
        "preview_scale": project.preview_scale,
        "tile_count": int(tile_index.get("tile_count", 0)),
        "valid_tile_count": int(tile_index.get("valid_tile_count", 0)),
        "skipped_black_tile_count": int(tile_index.get("skipped_black_tile_count", 0)),
        "global_mask_exists": bool(project.global_mask_path and os.path.isfile(project.global_mask_path)),
        "mask_size": mask_size,
        "mask_size_matches_original": mask_size == [project.image_width, project.image_height] if mask_size else False,
        "geo_calibration_exists": calibration_valid,
        "calibration_image_size": list(calibration_size),
        "calibration_size_matches": calibration_size == (project.image_width, project.image_height),
        "final_graph_exists": graph_exists or bool(project.global_graph_path and os.path.isfile(project.global_graph_path)),
        "task_points_use_original_pixel": task_original,
        "current_memory_mb": current_process_memory_mb(),
        "last_error_log": project.last_error_log,
        "coordinate_system": "image_pixel",
    }
    try:
        from PIL import Image
        with Image.open(project.preview_path) as image:
            report["preview_size"] = [int(image.width), int(image.height)]
    except Exception:
        pass
    reports_dir = Path(project.project_dir) / "reports"
    _atomic_json(reports_dir / "large_image_self_check.json", report)
    return report
