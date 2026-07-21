"""快速预览分割 Worker 单元测试。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from roadnet.preview_segmentation import (
    PreviewSegmentationWorker,
    PreviewSegmentationResult,
    generate_preview_segmentation,
    _compute_cache_key,
    _load_cache,
    _save_cache,
    DEFAULT_PREVIEW_MAX_SIDE,
    DEFAULT_COMPETITION_MAX_SIDE,
)


def _make_sample_preview(path: Path, size=(600, 800)):
    """创建一个模拟的 preview.png 用于测试。"""
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # 灰色背景
    image[:, :] = (128, 128, 128)
    # 白色道路区域
    image[200:400, 100:300] = (200, 200, 200)
    image[200:400, 500:700] = (210, 210, 210)
    # 绿地
    image[50:150, :] = (100, 150, 100)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image


class PreviewSegmentationWorkerTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._preview_path = self._root / "preview.png"
        self._preview_rgb = _make_sample_preview(self._preview_path)
        self._output_dir = str(self._root / "mask_preview")

        # 正样本：白色道路区域
        self._pos_rgb = np.array([[200, 200, 200]], dtype=np.uint8)
        # 负样本：绿色区域
        self._neg_rgb = np.array([[100, 150, 100]], dtype=np.uint8)

        self._base_config = {
            "mode": "hsv",
            "h_margin": 10,
            "s_margin": 40,
            "v_margin": 50,
            "use_negative_samples": True,
            "sample_radius": 3,
            "preview_blur_kernel": 3,
            "preview_open_kernel": 3,
            "preview_close_kernel": 3,
        }

    def tearDown(self):
        self._tmp.cleanup()

    def test_generate_preview_segmentation_returns_valid_mask(self):
        """基本分割：返回非空二值 mask，尺寸匹配。"""
        mask = generate_preview_segmentation(
            self._preview_rgb,
            self._pos_rgb,
            self._neg_rgb,
            self._base_config,
            preview_max_side=800,
        )
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.ndim, 2)
        self.assertEqual(mask.shape, self._preview_rgb.shape[:2])
        self.assertTrue(mask.max() <= 255)
        self.assertTrue(mask.min() >= 0)

    def test_preview_seg_only_uses_preview_not_original(self):
        """验证分割结果来自 preview，不是原始大图。"""
        mask = generate_preview_segmentation(
            self._preview_rgb,
            self._pos_rgb,
            self._neg_rgb,
            self._base_config,
            preview_max_side=800,
        )
        self.assertEqual(mask.shape[0], 800)
        self.assertEqual(mask.shape[1], 600)

    def test_downscale_when_preview_too_large(self):
        """preview 超过 max_side 时应自动缩放到 max_side。"""
        large_preview = np.zeros((3000, 4000, 3), dtype=np.uint8)
        mask = generate_preview_segmentation(
            large_preview,
            self._pos_rgb,
            self._neg_rgb,
            self._base_config,
            preview_max_side=800,
        )
        # 由于 scale < 1.0，最后会恢复到原尺寸
        self.assertEqual(mask.shape, (3000, 4000))

    def test_competition_mode_smaller_max_side(self):
        """比赛模式应使用更小的 max_side。"""
        large_preview = np.zeros((3000, 4000, 3), dtype=np.uint8)
        mask = generate_preview_segmentation(
            large_preview,
            self._pos_rgb,
            self._neg_rgb,
            self._base_config,
            preview_max_side=DEFAULT_COMPETITION_MAX_SIDE,
        )
        self.assertEqual(mask.shape, (3000, 4000))

    def test_cache_key_stable_for_same_params(self):
        """相同参数生成相同缓存 key。"""
        key1 = _compute_cache_key(
            str(self._preview_path), 1500, self._base_config,
        )
        key2 = _compute_cache_key(
            str(self._preview_path), 1500, self._base_config,
        )
        self.assertEqual(key1, key2)

    def test_cache_key_changes_with_different_params(self):
        """不同参数生成不同缓存 key。"""
        key1 = _compute_cache_key(
            str(self._preview_path), 1500, self._base_config,
        )
        altered = dict(self._base_config)
        altered["h_margin"] = 20
        key2 = _compute_cache_key(
            str(self._preview_path), 1500, altered,
        )
        self.assertNotEqual(key1, key2)

    def test_cache_save_and_load(self):
        """缓存写入和读取循环。"""
        output_dir = str(self._root / "cache_test")
        test_mask = np.ones((800, 600), dtype=np.uint8) * 255

        key = _compute_cache_key(
            str(self._preview_path), 1500, self._base_config,
        )
        _save_cache(output_dir, key, test_mask)
        loaded = _load_cache(output_dir, key)

        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded, test_mask)

    def test_cache_miss_returns_none(self):
        """未命中缓存返回 None。"""
        loaded = _load_cache(
            str(self._root / "empty_cache"), "nonexistent_key",
        )
        self.assertIsNone(loaded)

    def test_preview_result_has_preview_only_flag(self):
        """结果必须标记 preview_only=True。"""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        result = PreviewSegmentationResult(
            preview_mask=mask,
            preview_only=True,
        )
        self.assertTrue(result.preview_only)

    def test_segmentation_does_not_read_original_image(self):
        """验证快速预览不会调用 cv2.imread 读取原始大图。"""
        source_path = Path(__file__).parent.parent / "roadnet" / "preview_segmentation.py"
        source_lines = source_path.read_text(encoding="utf-8").splitlines()

        # 只检查非注释、非 docstring 行的代码
        in_docstring = False
        code_only = []
        for line in source_lines:
            stripped = line.strip()
            # 跳过 docstring 中的内容
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # 跳过注释行
            if stripped.startswith("#"):
                continue
            code_only.append(stripped)

        code_text = "\n".join(code_only).lower()
        self.assertNotIn("cv2.imread(original", code_text,
                         "代码中不应包含 cv2.imread(original_*)")
        self.assertNotIn("qpixmap(original", code_text)
        self.assertNotIn("qimage(original", code_text)

    def test_performance_report_saved(self):
        """性能报告 JSON 格式正确。"""
        mask = generate_preview_segmentation(
            self._preview_rgb,
            self._pos_rgb,
            self._neg_rgb,
            self._base_config,
            preview_max_side=800,
        )
        os.makedirs(self._output_dir, exist_ok=True)

        from roadnet.preview_segmentation import \
            PreviewSegmentationWorker as PSW
        # 使用 worker 的 _write_report 方法（需要实例化）
        worker = PSW(
            preview_path=str(self._preview_path),
            pos_samples_rgb=self._pos_rgb,
            neg_samples_rgb=self._neg_rgb,
            output_dir=self._output_dir,
            config=self._base_config,
            preview_max_side=800,
        )
        report = worker._build_report(
            preview_size=(800, 600),
            elapsed=0.123,
            steps=["load", "segment"],
            cache_used=False,
        )
        worker._write_report(self._output_dir, report)

        report_path = os.path.join(self._output_dir,
                                   "preview_segmentation_report.json")
        self.assertTrue(os.path.isfile(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["elapsed_seconds"], 0.123)
        self.assertTrue(loaded["preview_only"])
        self.assertIn("operation_steps", loaded)

    def test_slow_preview_warning(self):
        """耗时 > 5s 生成 warning。"""
        worker = PreviewSegmentationWorker(
            preview_path=str(self._preview_path),
            pos_samples_rgb=self._pos_rgb,
            neg_samples_rgb=self._neg_rgb,
            output_dir=self._output_dir,
            config=self._base_config,
            preview_max_side=800,
        )
        report = worker._build_report(
            preview_size=(800, 600),
            elapsed=6.5,
            steps=[],
            cache_used=False,
        )
        self.assertIn("warning", report)
        self.assertIn("耗时过长", report["warning"])

    def test_fast_preview_no_warning(self):
        """耗时 < 5s 不生成 warning。"""
        worker = PreviewSegmentationWorker(
            preview_path=str(self._preview_path),
            pos_samples_rgb=self._pos_rgb,
            neg_samples_rgb=self._neg_rgb,
            output_dir=self._output_dir,
            config=self._base_config,
            preview_max_side=800,
        )
        report = worker._build_report(
            preview_size=(800, 600),
            elapsed=2.0,
            steps=[],
            cache_used=False,
        )
        self.assertNotIn("warning", report)


if __name__ == "__main__":
    unittest.main()
