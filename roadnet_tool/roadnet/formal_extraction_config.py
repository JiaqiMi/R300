"""
正式道路提取配置模块。

定义三种提取模式、tile 参数、缓存策略等。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ===================================================================
# 提取模式
# ===================================================================

EXTRACTION_MODE_FAST_PREVIEW = "fast_preview"   # 快速预览
EXTRACTION_MODE_ROI = "roi"                      # ROI 正式提取
EXTRACTION_MODE_FULL = "full"                    # 全图正式提取

# 推理方式
INFER_MODE_PERSISTENT = "persistent_worker"
INFER_MODE_SUBPROCESS = "subprocess_per_tile"


@dataclass
class FormalExtractionConfig:
    """正式道路提取完整配置。"""

    # ── 基本参数 ──
    extraction_mode: str = EXTRACTION_MODE_FULL
    infer_mode: str = INFER_MODE_PERSISTENT
    output_dir: Path = field(default_factory=Path)

    # ── 输入 ──
    image_path: Path = field(default_factory=Path)
    image_width: int = 0
    image_height: int = 0

    # ── SAM-Road 模型路径 ──
    python_executable: Path = field(default_factory=Path)
    infer_script: Path = field(default_factory=Path)
    config_path: Path = field(default_factory=Path)
    sam_backbone_ckpt_path: Path = field(default_factory=Path)
    samroad_model_ckpt_path: Path = field(default_factory=Path)
    project_dir: Path = field(default_factory=Path)
    device: str = "cuda"
    mask_only_partial_load: bool = False

    # ── 推理后端适配器：samroadplus_portable / old_samroad / auto ──
    adapter_type: str = "auto"

    # ── Tile 参数 ──
    tile_size: int = 2048
    tile_overlap: int = 128
    tile_batch_size: int = 1           # GPU 显存允许时可设为 2 或 4
    max_tiles: Optional[int] = None    # None = 所有有效 tile

    # ── 跳过无效 tile ──
    skip_black_tile: bool = True
    skip_black_ratio_threshold: float = 0.80
    black_threshold: int = 10
    min_black_component_area: int = 4096
    valid_pixel_ratio_threshold: float = 0.10
    merge_method: str = "max"

    # ── ROI ──
    roi_polygons: List[List[Tuple[float, float]]] = field(default_factory=list)

    # ── 大图 tile 索引（可选，用于复用 tile_index.json 中的 tile_id）──
    tile_index_path: Optional[Path] = None

    # ── 断点续跑 ──
    resume_from_existing_tiles: bool = True

    # ── Debug ──
    debug_mode: bool = False

    # ── 比赛快速模式 ──
    competition_fast_mode: bool = False

    # ── 模型加载超时与心跳 ──
    model_load_timeout_seconds: int = 180        # 超时秒数，0 表示不超时
    heartbeat_interval_seconds: float = 2.0      # 心跳间隔

    # ── 测试模式 ──
    max_tiles_for_test: int = 0                  # 测试模式：只处理前 N 个 tile，0 表示禁用

    # ── 预热模式 ──
    warmup_only: bool = False                    # 仅加载模型，不提取 tile

    # ── 自动后续流程 ──
    auto_skeleton: bool = False
    auto_graph: bool = False

    def apply_competition_mode(self):
        """应用比赛快速模式默认值。"""
        self.extraction_mode = EXTRACTION_MODE_ROI
        self.infer_mode = INFER_MODE_PERSISTENT
        self.debug_mode = False
        self.tile_overlap = 128
        self.resume_from_existing_tiles = True
        self.auto_skeleton = False
        self.auto_graph = False
        self.tile_batch_size = 1

    def get_cache_key(self, tile_id: str, tile_bbox: Tuple[int, int, int, int]) -> str:
        """计算 tile 的缓存 key（用于断点续跑）。"""
        parts = [
            str(self.image_path),
            tile_id,
            f"{tile_bbox[0]}_{tile_bbox[1]}_{tile_bbox[2]}_{tile_bbox[3]}",
            str(self.samroad_model_ckpt_path),
            str(self.config_path),
            str(self.sam_backbone_ckpt_path),
            str(self.tile_size),
            str(self.tile_overlap),
            str(self.device),
        ]
        key_str = "|".join(parts)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "extraction_mode": self.extraction_mode,
            "infer_mode": self.infer_mode,
            "output_dir": str(self.output_dir),
            "image_path": str(self.image_path),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "tile_batch_size": self.tile_batch_size,
            "max_tiles": self.max_tiles,
            "skip_black_tile": self.skip_black_tile,
            "skip_black_ratio_threshold": self.skip_black_ratio_threshold,
            "black_threshold": self.black_threshold,
            "valid_pixel_ratio_threshold": self.valid_pixel_ratio_threshold,
            "merge_method": self.merge_method,
            "resume_from_existing_tiles": self.resume_from_existing_tiles,
            "debug_mode": self.debug_mode,
            "competition_fast_mode": self.competition_fast_mode,
            "auto_skeleton": self.auto_skeleton,
            "auto_graph": self.auto_graph,
            "device": self.device,
            "samroad_model_ckpt_path": str(self.samroad_model_ckpt_path),
            "config_path": str(self.config_path),
            "sam_backbone_ckpt_path": str(self.sam_backbone_ckpt_path),
        }
