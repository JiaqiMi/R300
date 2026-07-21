import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.samroadplus_runner import (  # noqa: E402
    SAMRoadPlusConfig,
    load_samroadplus_config,
    sam_backbone_required,
    save_samroadplus_config,
    scan_samroadplus_project,
    validate_samroadplus_config,
)
from roadnet.samroad_single_adapter import load_single_output  # noqa: E402


SAMROAD_PYTHON = Path(r"C:\Users\小马\.conda\envs\samroad\python.exe")


class SAMRoadPlusRunnerTests(unittest.TestCase):
    def test_scan_finds_non_legacy_names_and_separates_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "predict.py").write_text("print('predict')", encoding="utf-8")
            (root / "custom.yml").write_text("SKIP_SAM_CKPT_LOAD: false", encoding="utf-8")
            (root / "trained.pt").write_bytes(b"model")
            (root / "sam_vit_b_01ec64.pth").write_bytes(b"sam")
            scan = scan_samroadplus_project(root)
            self.assertEqual(scan.preferred_infer_script.name, "predict.py")
            self.assertEqual(scan.preferred_config.name, "custom.yml")
            self.assertEqual(scan.preferred_checkpoint.name, "trained.pt")
            self.assertEqual(scan.preferred_sam_backbone.name, "sam_vit_b_01ec64.pth")

    def test_portable_config_can_skip_separate_sam_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.yaml"
            config_path.write_text(
                "NO_SAM: false\nSKIP_SAM_CKPT_LOAD: true\nSAM_CKPT_PATH: ''\n",
                encoding="utf-8",
            )
            for name in ("python.exe", "infer.py", "model_state_dict.pth", "image.png"):
                (root / name).write_bytes(b"x")
            config = SAMRoadPlusConfig(
                project_dir=root,
                python_executable=root / "python.exe",
                infer_script=root / "infer.py",
                config_path=config_path,
                model_ckpt_path=root / "model_state_dict.pth",
                input_image=root / "image.png",
                sam_backbone_ckpt_path=Path(),
            )
            self.assertFalse(sam_backbone_required(config_path))
            self.assertEqual(validate_samroadplus_config(config), [])

    def test_config_round_trip_preserves_plus_paths_and_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plus.yaml"
            config = SAMRoadPlusConfig(ignore_graph=True, tile_size=2048, overlap=256)
            save_samroadplus_config(config, str(path))
            restored = load_samroadplus_config(str(path))
            self.assertEqual(restored.model_type, "samroadplus_portable")
            self.assertTrue(restored.ignore_graph)
            self.assertEqual((restored.tile_size, restored.overlap), (2048, 256))
            self.assertEqual(restored.project_dir, config.project_dir)

    @unittest.skipUnless(SAMROAD_PYTHON.is_file(), "SAM-Road Python environment unavailable")
    def test_bridge_maps_outputs_and_writes_logs_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            infer_script = root / "run_infer.py"
            infer_script.write_text(textwrap.dedent("""
                import argparse
                from pathlib import Path
                import cv2
                import numpy as np
                import torch
                import yaml

                def load_config(path):
                    with open(path, encoding='utf-8') as stream:
                        return yaml.safe_load(stream) or {}

                class SAMRoadplus(torch.nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.head = torch.nn.Linear(2, 2)

                def main():
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--image')
                    parser.add_argument('--output_dir')
                    parser.add_argument('--config')
                    parser.add_argument('--checkpoint')
                    parser.add_argument('--device')
                    args = parser.parse_args()
                    target = Path(args.output_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target / 'pred_mask.png'), np.full((24, 32), 180, np.uint8))
                    cv2.imwrite(str(target / 'overlay.png'), np.full((24, 32, 3), 90, np.uint8))
                    (target / 'pred_graph.json').write_text('{"nodes": [], "edges": []}', encoding='utf-8')
                    print('mock portable inference complete')

                if __name__ == '__main__':
                    main()
            """), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text("SKIP_SAM_CKPT_LOAD: true\n", encoding="utf-8")
            image_path = root / "image.png"
            image_path.write_bytes(b"input is unused by mock")
            checkpoint = root / "trained.pth"
            make_checkpoint = (
                "import sys, torch; "
                f"sys.path.insert(0, {str(root)!r}); "
                "import run_infer; "
                f"torch.save(run_infer.SAMRoadplus({{}}).state_dict(), {str(checkpoint)!r})"
            )
            subprocess.run([str(SAMROAD_PYTHON), "-c", make_checkpoint], check=True)
            bridge = ROOT / "samroad_bridge" / "run_samroadplus_bridge.py"
            proc = subprocess.run([
                str(SAMROAD_PYTHON), str(bridge),
                "--project-dir", str(root),
                "--infer-script", str(infer_script),
                "--image", str(image_path),
                "--config", str(config_path),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(output),
                "--device", "cpu",
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            for name in (
                "road_mask.png", "viz.png", "graph.json", "metadata.json",
                "samroadplus_stdout.log", "samroadplus_stderr.log",
            ):
                self.assertTrue((output / name).is_file(), name)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["success"])
            self.assertTrue(metadata["has_road_mask"])
            self.assertFalse(metadata["has_itsc_mask"])
            self.assertTrue(metadata["has_graph"])
            self.assertFalse(metadata["final_graph_overwritten"])
            self.assertEqual(metadata["output_mapping"]["road_mask"]["source"].split(os.sep)[-1], "pred_mask.png")

    def test_dialog_contains_model_selector_and_plus_controls(self):
        source = (ROOT / "gui" / "samroad_single_run_dialog.py").read_text(encoding="utf-8")
        self.assertIn("samroad_single_image", source)
        self.assertIn("SAM-RoadPlus Portable（新训练结果）", source)
        self.assertIn("扫描 SAM-RoadPlus 工程", source)
        self.assertIn("检查 checkpoint/config 匹配", source)
        self.assertIn("只导入 road_mask，忽略 SAM-RoadPlus graph", source)

    def test_standard_graph_json_loads_only_as_reference_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Use OpenCV through the test runtime for the required mask.
            import cv2
            import numpy as np
            cv2.imwrite(str(root / "road_mask.png"), np.full((20, 30), 255, np.uint8))
            graph = {
                "nodes": [{"id": "a", "x": 2, "y": 3}, {"id": "b", "x": 20, "y": 10}],
                "edges": [{"id": "ab", "start": "a", "end": "b", "polyline": [[2, 3], [20, 10]]}],
            }
            (root / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
            output = load_single_output(str(root))
            self.assertTrue(output.has_graph)
            self.assertEqual((output.node_count, output.edge_count), (2, 1))
            self.assertEqual(output.graph_edges[0]["source"], "samroadplus_reference")


if __name__ == "__main__":
    unittest.main()
