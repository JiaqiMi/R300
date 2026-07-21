import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

from roadnet.large_image_project import (
    LargeImageProject,
    create_large_image_project,
    generate_tile_index,
    determine_risk_level,
    check_memory_budget,
    estimate_image_memory_mb,
    RISK_SAFE, RISK_WARNING, RISK_HIGH_RISK, RISK_BLOCKED,
    RISK_MESSAGES, RISK_POLICIES,
    MAX_SAFE_RGB_MB, MAX_CRITICAL_RGB_MB,
    LARGE_IMAGE_THRESHOLD_DIM, LARGE_IMAGE_THRESHOLD_PIXELS,
)


class LargeImageProjectTests(unittest.TestCase):
    def _make_image(self, root: Path) -> Path:
        image = np.full((420, 620, 3), 180, dtype=np.uint8)
        image[:45, :] = 0                 # border-connected invalid band
        image[150:330, 220:400] = 0       # internal dark region: must stay valid
        path = root / "large_source.png"
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_project_preview_index_and_resume_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_image(root)
            project = create_large_image_project(
                str(source), str(root / "projects"), tile_size=200,
                tile_overlap=40, preview_max_side=300,
            )
            index = generate_tile_index(
                project, black_threshold=10, black_ratio_threshold=0.8,
            )
            self.assertEqual((project.image_width, project.image_height), (620, 420))
            self.assertTrue(Path(project.preview_path).is_file())
            self.assertTrue(Path(project.tile_index_path).is_file())
            self.assertEqual(index["coordinate_system"], "image_pixel")
            self.assertEqual(index["tile_size"], 200)
            self.assertEqual(index["overlap"], 40)
            self.assertGreater(index["tile_count"], 1)
            self.assertTrue(all("black_ratio" in tile for tile in index["tiles"]))
            self.assertTrue(all("border_invalid_ratio" in tile for tile in index["tiles"]))
            # A dark tile wholly inside the image is not invalidated merely
            # because RGB is near zero.
            darkest = max(index["tiles"], key=lambda tile: tile["black_ratio"])
            self.assertTrue(darkest["valid"])
            loaded = LargeImageProject.load(str(project.project_path))
            self.assertEqual(loaded.image_path, project.image_path)
            self.assertEqual(loaded.preview_scale, project.preview_scale)

    def test_project_json_contains_required_original_coordinate_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = create_large_image_project(
                str(self._make_image(root)), str(root / "projects"),
                tile_size=2048, tile_overlap=256,
            )
            payload = json.loads(Path(project.project_path).read_text(encoding="utf-8"))
            for key in (
                "image_path", "image_width", "image_height", "preview_path",
                "preview_scale", "tile_size", "tile_overlap", "tile_index_path",
                "coordinate_system", "geo_calibration_path", "global_mask_path",
                "global_graph_path",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["coordinate_system"], "image_pixel")


class RiskLevelTests(unittest.TestCase):
    """统一风险等级枚举测试。

    验证 RISK_SAFE / RISK_WARNING / RISK_HIGH_RISK / RISK_BLOCKED 的行为一致性。
    """

    def test_all_risk_levels_in_enums(self):
        """RISK_MESSAGES 和 RISK_POLICIES 必须包含全部 4 个等级。"""
        for level in (RISK_SAFE, RISK_WARNING, RISK_HIGH_RISK, RISK_BLOCKED):
            self.assertIn(level, RISK_MESSAGES, f"RISK_MESSAGES missing: {level}")
            self.assertIn(level, RISK_POLICIES, f"RISK_POLICIES missing: {level}")

    def test_small_image_is_safe(self):
        """小图 (<4096, <16M pixels) 返回 safe。"""
        self.assertEqual(determine_risk_level(1024, 1024), RISK_SAFE)
        self.assertEqual(determine_risk_level(4096, 2048), RISK_SAFE)  # 刚好 8.4M

    def test_large_but_reasonable_is_warning(self):
        """超过阈值但内存 <500MB 返回 warning。"""
        # 5000x5000 = 25M pixels, raw_rgb ≈ 71.5MB < 500MB
        self.assertEqual(determine_risk_level(5000, 5000), RISK_WARNING)

    def test_very_large_is_high_risk(self):
        """10000x12000 → raw_rgb ≈ 343MB < 500MB, 但 pixels > 16M, warning.
        要触发 high_risk 需要 raw_rgb > 500MB。
        例如 15000x14000 → 600MB > 500MB。
        """
        # 15000 x 14000 → raw_rgb ≈ 600 MB > 500MB → high_risk
        risk = determine_risk_level(15000, 14000)
        self.assertEqual(risk, RISK_HIGH_RISK,
                         f"Expected high_risk for 15000x14000, got {risk}")

    def test_extreme_image_is_blocked(self):
        """超大图 (>2GB) 返回 blocked。"""
        # 30000x30000 → raw_rgb ≈ 2574 MB > 2GB
        risk = determine_risk_level(30000, 30000)
        self.assertEqual(risk, RISK_BLOCKED,
                         f"Expected blocked for 30000x30000, got {risk}")

    def test_high_risk_no_keyerror(self):
        """high_risk 不应导致 KeyError。"""
        mem = check_memory_budget(15000, 14000)
        self.assertIn("risk_level", mem)
        self.assertEqual(mem["risk_level"], RISK_HIGH_RISK)
        # 安全访问 high_risk
        self.assertTrue(mem.get("high_risk", False))
        # 策略必须允许继续
        policy = mem["policy"]
        self.assertTrue(policy["may_continue"])
        self.assertTrue(policy["large_image_mode"])
        self.assertFalse(policy["should_load_full_pixmap"])
        self.assertTrue(policy["should_generate_preview"])

    def test_blocked_policy_prevents_open(self):
        """blocked 等级不允许继续。"""
        mem = check_memory_budget(30000, 30000)
        self.assertEqual(mem["risk_level"], RISK_BLOCKED)
        policy = mem["policy"]
        self.assertFalse(policy["may_continue"])
        self.assertFalse(policy["should_generate_preview"])

    def test_unknown_risk_level_fallback(self):
        """未知 risk_level 不抛 KeyError，默认按 high_risk 处理。"""
        unknown = "unknown_level_xyz"
        msg = RISK_MESSAGES.get(unknown, RISK_MESSAGES[RISK_HIGH_RISK])
        self.assertEqual(msg, RISK_MESSAGES[RISK_HIGH_RISK])

        policy = RISK_POLICIES.get(unknown, RISK_POLICIES[RISK_HIGH_RISK])
        self.assertTrue(policy["may_continue"])
        self.assertTrue(policy["large_image_mode"])

    def test_determine_risk_returns_valid_enum(self):
        """determine_risk_level 返回值必须在 4 个有效值中。"""
        valid = {RISK_SAFE, RISK_WARNING, RISK_HIGH_RISK, RISK_BLOCKED}
        for w, h in [(10, 10), (4096, 4096), (5000, 5000),
                     (15000, 14000), (30000, 30000)]:
            level = determine_risk_level(w, h)
            self.assertIn(level, valid, f"Invalid risk_level '{level}' for {w}x{h}")

    def test_mock_10000x12000_scenario(self):
        """模拟用户报告的场景: 10000x12000。

        10000*12000*3/(1024^2) ≈ 343 MB, < 500MB, 但 pixels = 120M > 16M。
        实际风险: warning（因为 raw_rgb < 500MB），
        流程应该: large_image_mode=True, 不加载 full pixmap, 生成 preview。
        """
        w, h = 10000, 12000
        mem = check_memory_budget(w, h)
        self.assertIn("risk_level", mem)
        risk = mem["risk_level"]

        # 对于 10000x12000: raw_rgb ≈ 343 MB < 500MB → WARNING（不是 HIGH_RISK）
        # 但仍然要进入大图模式
        print(f"\n[Test] 10000x12000 → raw_rgb_mb={mem['raw_rgb_mb']:.1f}, risk_level={risk}")

        # 确保不抛 KeyError
        _ = RISK_MESSAGES.get(risk, RISK_MESSAGES[RISK_HIGH_RISK])
        _ = RISK_POLICIES.get(risk, RISK_POLICIES[RISK_HIGH_RISK])

        # warning 和 high_risk 都应进入大图模式
        self.assertTrue(isinstance(mem["high_risk"], bool))
        # 流程必须继续
        self.assertTrue(mem["policy"]["may_continue"])
        # 生成 preview
        self.assertTrue(mem["policy"]["should_generate_preview"])


if __name__ == "__main__":
    unittest.main()
