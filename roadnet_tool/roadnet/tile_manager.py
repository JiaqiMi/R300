"""
TileManager - 大图切片管理器

支持大画幅影像的切片处理，用于：
1. 图像金字塔管理
2. Tile 生成和读取
3. 坐标转换（全局坐标 ↔ Tile 局部坐标）
4. Tile 掩码合并
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

import cv2
import numpy as np


# 大图检测阈值
LARGE_IMAGE_THRESHOLD = 4096

# 默认切片参数
DEFAULT_TILE_SIZE = 2048
DEFAULT_OVERLAP = 256


@dataclass
class Tile:
    """单块 Tile 的元数据"""
    id: str                    # e.g. "tile_000_000"
    x0: int                    # 左上角 x（全局像素坐标）
    y0: int                    # 左上角 y（全局像素坐标）
    x1: int                    # 右下角 x（全局像素坐标）
    y1: int                    # 右下角 y（全局像素坐标）
    width: int                 # tile 宽度（可能小于 tile_size，边界处）
    height: int                # tile 高度（可能小于 tile_size，边界处）

    def contains(self, x: int, y: int) -> bool:
        """检查点是否在 tile 内"""
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x0": self.x0, "y0": self.y0,
            "x1": self.x1, "y1": self.y1,
            "width": self.width, "height": self.height,
        }

    @staticmethod
    def from_dict(d: dict) -> Tile:
        return Tile(
            id=d["id"],
            x0=d["x0"], y0=d["y0"],
            x1=d["x1"], y1=d["y1"],
            width=d["width"], height=d["height"],
        )


@dataclass
class ImagePyramidLevel:
    """图像金字塔单层"""
    level: int
    scale: float               # 相对于原图的比例
    width: int
    height: int
    image: Optional[np.ndarray] = None


class TileManager:
    """
    大图切片管理器

    功能：
    1. 检测大图并进入大图模式
    2. 生成预览图（preview image）
    3. 生成图像金字塔（预留接口）
    4. 生成切片（tiles）
    5. 坐标转换（全局 ↔ tile 局部 ↔ preview）
    6. Tile 掩码合并
    """

    def __init__(
        self,
        image_path: str,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        large_image_threshold: int = LARGE_IMAGE_THRESHOLD,
        preview_max_size: int = 3000,
    ):
        """
        Args:
            image_path: 原始图像路径
            tile_size: 切片大小（默认 2048）
            overlap: 切片重叠区域（默认 256）
            large_image_threshold: 大图检测阈值（默认 4096）
            preview_max_size: 预览图最大尺寸（默认 3000）
        """
        self.image_path = image_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.large_image_threshold = large_image_threshold
        self.preview_max_size = preview_max_size

        # 图像信息
        self._image_rgb: Optional[np.ndarray] = None
        self._original_width: int = 0
        self._original_height: int = 0

        # 大图模式标志
        self._large_image_mode: bool = False
        self._preview_scale: float = 1.0
        self._preview_width: int = 0
        self._preview_height: int = 0

        # 图像金字塔
        self._pyramid_levels: List[ImagePyramidLevel] = []

        # 切片
        self._tiles: List[Tile] = []
        self._tile_dir: str = ""

        # 初始化
        self._load_image_info()

    def _load_image_info(self):
        """加载图像信息（不加载完整图像）"""
        # 读取图像尺寸
        cap = cv2.VideoCapture(self.image_path)
        if cap.isOpened():
            self._original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            # 尝试直接读取
            img = cv2.imread(self.image_path)
            if img is not None:
                self._original_height, self._original_width = img.shape[:2]
            else:
                raise FileNotFoundError(f"无法读取图像: {self.image_path}")

        # 检测大图模式
        max_dim = max(self._original_width, self._original_height)
        self._large_image_mode = max_dim > self.large_image_threshold

        if self._large_image_mode:
            # 计算预览图缩放比例
            self._preview_scale = min(1.0, self.preview_max_size / max_dim)
            self._preview_width = int(self._original_width * self._preview_scale)
            self._preview_height = int(self._original_height * self._preview_scale)
        else:
            self._preview_scale = 1.0
            self._preview_width = self._original_width
            self._preview_height = self._original_height

    # ===================================================================
    # 属性
    # ===================================================================

    @property
    def original_width(self) -> int:
        return self._original_width

    @property
    def original_height(self) -> int:
        return self._original_height

    @property
    def original_size(self) -> Tuple[int, int]:
        return (self._original_width, self._original_height)

    @property
    def preview_width(self) -> int:
        return self._preview_width

    @property
    def preview_height(self) -> int:
        return self._preview_height

    @property
    def preview_scale(self) -> float:
        return self._preview_scale

    @property
    def is_large_image_mode(self) -> bool:
        return self._large_image_mode

    @property
    def tiles(self) -> List[Tile]:
        return self._tiles

    @property
    def tile_dir(self) -> str:
        return self._tile_dir

    # ===================================================================
    # 预览图
    # ===================================================================

    def get_preview_image(self) -> np.ndarray:
        """
        获取预览图（大图模式下缩小的图像）

        Returns:
            预览图 RGB 图像
        """
        if self._image_rgb is None:
            self._image_rgb = cv2.cvtColor(
                cv2.imread(self.image_path), cv2.COLOR_BGR2RGB
            )

        if self._large_image_mode and self._preview_scale < 1.0:
            h, w = self._image_rgb.shape[:2]
            new_w = int(w * self._preview_scale)
            new_h = int(h * self._preview_scale)
            preview = cv2.resize(self._image_rgb, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
            return preview
        else:
            return self._image_rgb

    def get_original_image(self) -> np.ndarray:
        """
        获取原始图像（全分辨率）

        Returns:
            原始 RGB 图像
        """
        if self._image_rgb is None:
            self._image_rgb = cv2.cvtColor(
                cv2.imread(self.image_path), cv2.COLOR_BGR2RGB
            )
        return self._image_rgb

    # ===================================================================
    # 坐标转换
    # ===================================================================

    def preview_to_global(self, x_preview: int, y_preview: int) -> Tuple[int, int]:
        """预览图坐标 → 原图全局像素坐标"""
        x_global = int(x_preview / self._preview_scale)
        y_global = int(y_preview / self._preview_scale)
        return (x_global, y_global)

    def preview_to_global_f(self, x_preview: float, y_preview: float) -> Tuple[float, float]:
        """预览图坐标 → 原图全局像素坐标（浮点）"""
        x_global = x_preview / self._preview_scale
        y_global = y_preview / self._preview_scale
        return (x_global, y_global)

    def global_to_preview(self, x_global: int, y_global: int) -> Tuple[int, int]:
        """原图全局像素坐标 → 预览图坐标"""
        x_preview = int(x_global * self._preview_scale)
        y_preview = int(y_global * self._preview_scale)
        return (x_preview, y_preview)

    def global_to_preview_f(self, x_global: float, y_global: float) -> Tuple[float, float]:
        """原图全局像素坐标 → 预览图坐标（浮点）"""
        x_preview = x_global * self._preview_scale
        y_preview = y_global * self._preview_scale
        return (x_preview, y_preview)

    def global_to_tile(
        self, x_global: int, y_global: int, tile: Tile
    ) -> Tuple[int, int]:
        """全局像素坐标 → Tile 局部坐标"""
        return (x_global - tile.x0, y_global - tile.y0)

    def tile_to_global(
        self, x_local: int, y_local: int, tile: Tile
    ) -> Tuple[int, int]:
        """Tile 局部坐标 → 全局像素坐标"""
        return (x_local + tile.x0, y_local + tile.y0)

    def global_to_tile_f(
        self, x_global: float, y_global: float, tile: Tile
    ) -> Tuple[float, float]:
        """全局像素坐标 → Tile 局部坐标（浮点）"""
        return (x_global - tile.x0, y_global - tile.y0)

    def tile_to_global_f(
        self, x_local: float, y_local: float, tile: Tile
    ) -> Tuple[float, float]:
        """Tile 局部坐标 → 全局像素坐标（浮点）"""
        return (x_local + tile.x0, y_local + tile.y0)

    # ===================================================================
    # 多边形坐标转换
    # ===================================================================

    def polygon_global_to_preview(
        self, points_global: List[List[float]]
    ) -> List[List[float]]:
        """多边形全局坐标 → 预览图坐标"""
        return [[x * self._preview_scale, y * self._preview_scale]
                for x, y in points_global]

    def polygon_preview_to_global(
        self, points_preview: List[List[float]]
    ) -> List[List[float]]:
        """多边形预览图坐标 → 全局坐标"""
        scale = 1.0 / self._preview_scale
        return [[x * scale, y * scale] for x, y in points_preview]

    def polygon_global_to_tile(
        self, points_global: List[List[float]], tile: Tile
    ) -> List[List[float]]:
        """多边形全局坐标 → Tile 局部坐标"""
        return [[x - tile.x0, y - tile.y0] for x, y in points_global]

    def polygon_tile_to_global(
        self, points_local: List[List[float]], tile: Tile
    ) -> List[List[float]]:
        """多边形 Tile 局部坐标 → 全局坐标"""
        return [[x + tile.x0, y + tile.y0] for x, y in points_local]

    # ===================================================================
    # 切片生成
    # ===================================================================

    def generate_tiles(self, output_dir: str = None) -> List[Tile]:
        """
        生成切片元数据

        Args:
            output_dir: 输出目录（用于保存 tile 元数据）

        Returns:
            Tile 列表
        """
        if output_dir:
            self._tile_dir = os.path.join(output_dir, "tiles")
            os.makedirs(self._tile_dir, exist_ok=True)

        self._tiles = []
        step = self.tile_size - self.overlap

        tile_id = 0
        y = 0
        while y < self._original_height:
            x = 0
            while x < self._original_width:
                # 计算 tile 边界
                x0 = x
                y0 = y
                x1 = min(x + self.tile_size, self._original_width)
                y1 = min(y + self.tile_size, self._original_height)
                width = x1 - x0
                height = y1 - y0

                tile = Tile(
                    id=f"tile_{tile_id:03d}_{y // step:03d}_{x // step:03d}",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    width=width, height=height,
                )
                self._tiles.append(tile)

                x += step
                tile_id += 1
            y += step

        # 保存 tile 元数据
        if output_dir:
            self.save_tile_metadata()

        return self._tiles

    def save_tile_metadata(self, path: str = None):
        """保存 tile 元数据到 JSON"""
        if path is None:
            path = os.path.join(self._tile_dir, "tiles.json")

        data = {
            "image_path": self.image_path,
            "original_width": self._original_width,
            "original_height": self._original_height,
            "preview_scale": self._preview_scale,
            "preview_width": self._preview_width,
            "preview_height": self._preview_height,
            "large_image_mode": self._large_image_mode,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "tiles": [t.to_dict() for t in self._tiles],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[TileManager] Tile 元数据已保存: {path}")

    def load_tile_metadata(self, path: str):
        """从 JSON 加载 tile 元数据"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._original_width = data["original_width"]
        self._original_height = data["original_height"]
        self._preview_scale = data["preview_scale"]
        self._preview_width = data["preview_width"]
        self._preview_height = data["preview_height"]
        self._large_image_mode = data["large_image_mode"]
        self.tile_size = data["tile_size"]
        self.overlap = data["overlap"]
        self._tiles = [Tile.from_dict(t) for t in data["tiles"]]

        print(f"[TileManager] Tile 元数据已加载: {len(self._tiles)} tiles")

    # ===================================================================
    # Tile 读取
    # ===================================================================

    def read_tile(self, tile: Tile) -> np.ndarray:
        """
        读取 tile 区域的全分辨率图像

        Args:
            tile: Tile 元数据

        Returns:
            RGB 图像 (H, W, 3)
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self.image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tile_img = img_rgb[tile.y0:tile.y1, tile.x0:tile.x1]
        return tile_img

    def read_tile_with_overlap(self, tile: Tile) -> Tuple[np.ndarray, Tile]:
        """
        读取带重叠区域的 tile

        Args:
            tile: Tile 元数据

        Returns:
            (带重叠的图像, 扩展后的 Tile)
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self.image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 扩展边界
        x0_ex = max(0, tile.x0 - self.overlap)
        y0_ex = max(0, tile.y0 - self.overlap)
        x1_ex = min(self._original_width, tile.x1 + self.overlap)
        y1_ex = min(self._original_height, tile.y1 + self.overlap)

        tile_img = img_rgb[y0_ex:y1_ex, x0_ex:x1_ex]

        # 创建扩展后的 tile
        extended_tile = Tile(
            id=f"{tile.id}_extended",
            x0=x0_ex, y0=y0_ex, x1=x1_ex, y1=y1_ex,
            width=x1_ex - x0_ex, height=y1_ex - y0_ex,
        )

        return tile_img, extended_tile

    # ===================================================================
    # Tile 掩码处理
    # ===================================================================

    def save_tile_mask(self, tile: Tile, mask: np.ndarray, name: str = "mask"):
        """
        保存 tile 的掩码

        Args:
            tile: Tile 元数据
            mask: 二值掩码 (H, W)
            name: 掩码名称
        """
        if not self._tile_dir:
            raise ValueError("Tile 目录未设置，请先调用 generate_tiles()")

        filename = f"{tile.id}_{name}.png"
        path = os.path.join(self._tile_dir, filename)
        cv2.imwrite(path, mask)
        print(f"[TileManager] Tile 掩码已保存: {path}")

    def load_tile_mask(self, tile: Tile, name: str = "mask") -> np.ndarray:
        """加载 tile 的掩码"""
        filename = f"{tile.id}_{name}.png"
        path = os.path.join(self._tile_dir, filename)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask

    def merge_tile_masks(
        self,
        mask_name: str = "mask",
        method: str = "max",
        output_dir: str = None,
    ) -> np.ndarray:
        """
        合并所有 tile 的掩码为全局掩码

        Args:
            mask_name: 掩码名称
            method: 合并方法
                - "max": 取最大值（用于道路掩码）
                - "mean": 取平均值
                - "first": 保留第一个非零值
            output_dir: 输出目录

        Returns:
            全局掩码 (original_height, original_width)
        """
        if not self._tiles:
            raise ValueError("没有 tiles，请先调用 generate_tiles()")

        # 创建全局掩码
        global_mask = np.zeros(
            (self._original_height, self._original_width), dtype=np.uint8
        )
        weight_map = np.zeros(
            (self._original_height, self._original_width), dtype=np.float32
        )

        # 中心区域权重为 1，边缘权重线性递减
        center_margin = self.overlap // 2

        for tile in self._tiles:
            try:
                tile_mask = self.load_tile_mask(tile, mask_name)
                if tile_mask is None:
                    continue

                h, w = tile_mask.shape[:2]

                # 计算权重
                weight = np.ones((h, w), dtype=np.float32)

                # 边缘权重递减
                if center_margin > 0:
                    # 顶部边缘
                    weight[:center_margin] = np.linspace(0, 1, center_margin)[:, np.newaxis]
                    # 底部边缘
                    weight[-center_margin:] = np.linspace(1, 0, center_margin)[:, np.newaxis]
                    # 左侧边缘
                    weight[:, :center_margin] *= np.linspace(0, 1, center_margin)[np.newaxis, :]
                    # 右侧边缘
                    weight[:, -center_margin:] *= np.linspace(1, 0, center_margin)[np.newaxis, :]

                # 累加掩码
                if method == "max":
                    global_mask[tile.y0:tile.y1, tile.x0:tile.x1] = np.maximum(
                        global_mask[tile.y0:tile.y1, tile.x0:tile.x1],
                        tile_mask
                    )
                else:
                    global_mask[tile.y0:tile.y1, tile.x0:tile.x1] = np.maximum(
                        global_mask[tile.y0:tile.y1, tile.x0:tile.x1],
                        tile_mask
                    )
                    weight_map[tile.y0:tile.y1, tile.x0:tile.x1] += weight

            except FileNotFoundError:
                continue

        # 二值化
        if method != "max" and np.any(weight_map > 0):
            # 对于加权平均后的结果，取平均后阈值化
            pass

        global_mask = (global_mask > 127).astype(np.uint8) * 255

        # 保存全局掩码
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"global_{mask_name}.png")
            cv2.imwrite(output_path, global_mask)
            print(f"[TileManager] 全局掩码已保存: {output_path}")

        return global_mask

    # ===================================================================
    # 图像金字塔（预留接口）
    # ===================================================================

    def build_pyramid(self, max_level: int = 4):
        """
        构建图像金字塔（预留接口，V1 可选实现）

        Args:
            max_level: 金字塔最大层级数
        """
        if self._image_rgb is None:
            self._image_rgb = self.get_original_image()

        self._pyramid_levels = []
        current = self._image_rgb.copy()
        scale = 1.0

        for level in range(max_level):
            h, w = current.shape[:2]
            self._pyramid_levels.append(ImagePyramidLevel(
                level=level,
                scale=scale,
                width=w,
                height=h,
                image=current.copy(),
            ))

            if level < max_level - 1:
                h_half = max(1, h // 2)
                w_half = max(1, w // 2)
                current = cv2.resize(current, (w_half, h_half),
                                      interpolation=cv2.INTER_AREA)
                scale *= 0.5

        print(f"[TileManager] 图像金字塔已构建: {len(self._pyramid_levels)} 层")

    def get_pyramid_level(self, level: int) -> Optional[ImagePyramidLevel]:
        """获取金字塔指定层级"""
        if 0 <= level < len(self._pyramid_levels):
            return self._pyramid_levels[level]
        return None

    # ===================================================================
    # 实用方法
    # ===================================================================

    def find_tile_at(self, x_global: int, y_global: int) -> Optional[Tile]:
        """查找包含指定全局坐标的 tile"""
        for tile in self._tiles:
            if tile.contains(x_global, y_global):
                return tile
        return None

    def get_tile_index(self, x_global: int, y_global: int) -> int:
        """获取指定全局坐标所在的 tile 索引"""
        for i, tile in enumerate(self._tiles):
            if tile.contains(x_global, y_global):
                return i
        return -1

    def get_covering_tiles(
        self, x_global: int, y_global: int, margin: int = 0
    ) -> List[Tile]:
        """
        获取覆盖指定点及其周围区域的 tiles

        Args:
            x_global, y_global: 全局坐标
            margin: 扩展边距（像素）

        Returns:
            覆盖该区域的 tile 列表
        """
        covering = []
        x0 = max(0, x_global - margin)
        y0 = max(0, y_global - margin)
        x1 = min(self._original_width, x_global + margin)
        y1 = min(self._original_height, y_global + margin)

        for tile in self._tiles:
            # 检查 tile 是否与扩展区域相交
            if (tile.x0 < x1 and tile.x1 > x0 and
                tile.y0 < y1 and tile.y1 > y0):
                covering.append(tile)

        return covering

    # ===================================================================
    # 项目保存/加载
    # ===================================================================

    def to_dict(self) -> dict:
        """导出为字典（用于项目保存）"""
        return {
            "image_path": self.image_path,
            "original_width": self._original_width,
            "original_height": self._original_height,
            "preview_scale": self._preview_scale,
            "preview_width": self._preview_width,
            "preview_height": self._preview_height,
            "large_image_mode": self._large_image_mode,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "tile_dir": self._tile_dir,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TileManager:
        """从字典创建（用于项目加载）"""
        tm = cls(
            image_path=data["image_path"],
            tile_size=data.get("tile_size", DEFAULT_TILE_SIZE),
            overlap=data.get("overlap", DEFAULT_OVERLAP),
        )
        tm._original_width = data["original_width"]
        tm._original_height = data["original_height"]
        tm._preview_scale = data["preview_scale"]
        tm._preview_width = data["preview_width"]
        tm._preview_height = data["preview_height"]
        tm._large_image_mode = data["large_image_mode"]
        tm._tile_dir = data.get("tile_dir", "")
        return tm

    # ===================================================================
    # 统计信息
    # ===================================================================

    def get_info(self) -> str:
        """获取大图信息摘要"""
        info = [
            f"图像路径: {self.image_path}",
            f"原始尺寸: {self._original_width} x {self._original_height}",
            f"大图模式: {'是' if self._large_image_mode else '否'}",
        ]

        if self._large_image_mode:
            info.append(f"预览缩放: {self._preview_scale:.4f}")
            info.append(f"预览尺寸: {self._preview_width} x {self._preview_height}")
            info.append(f"切片大小: {self.tile_size}")
            info.append(f"重叠区域: {self.overlap}")
            info.append(f"Tile 数量: {len(self._tiles)}")

        return "\n".join(info)


# =============================================================================
# 便捷函数
# =============================================================================

def is_large_image(image_path: str, threshold: int = LARGE_IMAGE_THRESHOLD) -> bool:
    """
    检查图像是否为大图

    Args:
        image_path: 图像路径
        threshold: 大图阈值

    Returns:
        是否为大图
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    return max(w, h) > threshold


def compute_preview_scale(
    width: int,
    height: int,
    max_size: int = 3000
) -> float:
    """
    计算预览图缩放比例

    Args:
        width, height: 原始图像尺寸
        max_size: 预览图最大尺寸

    Returns:
        缩放比例
    """
    max_dim = max(width, height)
    if max_dim <= max_size:
        return 1.0
    return max_size / max_dim
