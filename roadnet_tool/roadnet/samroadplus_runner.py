"""SAM-RoadPlus Portable project discovery, configuration and bridge commands."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


MODEL_TYPE = "samroadplus_portable"
DEFAULT_CONFIG_PATH = "config/samroadplus_config.yaml"
INFER_NAMES = ("infer_single.py", "run_infer.py", "predict.py", "inference.py", "demo.py", "infer.py")
MASK_NAMES = (
    "road_mask.png", "mask.png", "pred_mask.png", "road_pred.png",
    "road_prediction.png", "seg.png", "segmentation.png", "binary_mask.png",
    "output_mask.png",
)
VIZ_NAMES = ("viz.png", "vis.png", "result.png", "overlay.png", "topo_vis.png", "preview.png")
GRAPH_NAMES = ("graph.p", "graph.pkl", "graph.json", "pred_graph.json", "topology.json")


def path_is_set(value) -> bool:
    return str(value or "").strip() not in ("", ".")


@dataclass
class SAMRoadPlusConfig:
    model_type: str = MODEL_TYPE
    project_dir: Path = Path("D:/samroadplus_portable_infer")
    python_executable: Path = Path("C:/Users/小马/.conda/envs/samroad/python.exe")
    infer_script: Path = Path("D:/samroadplus_portable_infer/infer.py")
    config_path: Path = Path("D:/samroadplus_portable_infer/config.yaml")
    model_ckpt_path: Path = Path("D:/samroadplus_portable_infer/model_state_dict.pth")
    sam_backbone_ckpt_path: Path = Path()
    input_image: Path = Path()
    output_dir: Path = Path()
    device: str = "cuda"
    auto_import_after_run: bool = True
    ignore_graph: bool = True
    inference_mode: str = "auto"
    tile_size: int = 1024
    overlap: int = 128
    skip_black_tile: bool = True


@dataclass
class SAMRoadPlusScanResult:
    project_dir: Path
    infer_scripts: List[Path] = field(default_factory=list)
    config_files: List[Path] = field(default_factory=list)
    model_checkpoints: List[Path] = field(default_factory=list)
    sam_backbones: List[Path] = field(default_factory=list)
    output_candidates: Dict[str, List[Path]] = field(default_factory=dict)

    @property
    def preferred_infer_script(self) -> Optional[Path]:
        return self.infer_scripts[0] if self.infer_scripts else None

    @property
    def preferred_config(self) -> Optional[Path]:
        return self.config_files[0] if self.config_files else None

    @property
    def preferred_checkpoint(self) -> Optional[Path]:
        return self.model_checkpoints[0] if self.model_checkpoints else None

    @property
    def preferred_sam_backbone(self) -> Optional[Path]:
        return self.sam_backbones[0] if self.sam_backbones else None

    def to_dict(self) -> dict:
        return {
            "project_dir": str(self.project_dir),
            "infer_scripts": [str(path) for path in self.infer_scripts],
            "config_files": [str(path) for path in self.config_files],
            "model_checkpoints": [str(path) for path in self.model_checkpoints],
            "sam_backbones": [str(path) for path in self.sam_backbones],
            "output_candidates": {
                key: [str(path) for path in values]
                for key, values in self.output_candidates.items()
            },
        }


def _priority(path: Path, preferred_names) -> tuple:
    name = path.name.lower()
    try:
        index = tuple(item.lower() for item in preferred_names).index(name)
    except ValueError:
        index = len(preferred_names)
    return index, len(path.parts), str(path).lower()


def scan_samroadplus_project(project_dir) -> SAMRoadPlusScanResult:
    """Discover actual portable inference assets without assuming old names."""
    root = Path(project_dir).expanduser().resolve()
    result = SAMRoadPlusScanResult(project_dir=root)
    if not root.is_dir():
        return result

    python_files = [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]
    named_scripts = [path for path in python_files if path.name.lower() in INFER_NAMES]
    if not named_scripts:
        named_scripts = [
            path for path in python_files
            if any(token in path.name.lower() for token in ("infer", "predict", "demo"))
        ]
    result.infer_scripts = sorted(named_scripts, key=lambda path: _priority(path, INFER_NAMES))

    configs = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    config_py = [path for path in python_files if path.name.lower() == "config.py"]
    result.config_files = sorted(configs + config_py, key=lambda path: _priority(path, ("config.yaml", "config.yml", "config.py")))

    checkpoints = list(root.rglob("*.ckpt")) + list(root.rglob("*.pth")) + list(root.rglob("*.pt"))
    sam_backbones = [
        path for path in checkpoints
        if "sam_vit_" in path.name.lower() or "sam_ckpt" in str(path.parent).lower()
    ]
    sam_ids = {str(path).lower() for path in sam_backbones}
    result.model_checkpoints = sorted(
        [path for path in checkpoints if str(path).lower() not in sam_ids],
        key=lambda path: _priority(path, ("model_state_dict.pth", "model.ckpt", "checkpoint.pth")),
    )
    result.sam_backbones = sorted(sam_backbones, key=lambda path: _priority(path, ("sam_vit_b_01ec64.pth",)))

    all_files = [path for path in root.rglob("*") if path.is_file()]
    result.output_candidates = {
        "road_mask": sorted([path for path in all_files if path.name.lower() in MASK_NAMES]),
        "viz": sorted([path for path in all_files if path.name.lower() in VIZ_NAMES]),
        "graph": sorted([path for path in all_files if path.name.lower() in GRAPH_NAMES]),
    }
    return result


def read_model_config(config_path) -> dict:
    path = Path(config_path)
    if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml"):
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def sam_backbone_required(config_path) -> bool:
    data = read_model_config(config_path)
    if not data:
        return False
    return not bool(data.get("NO_SAM", False)) and not bool(data.get("SKIP_SAM_CKPT_LOAD", False))


def validate_samroadplus_config(config: SAMRoadPlusConfig) -> List[str]:
    errors: List[str] = []
    required = (
        (config.project_dir, "SAM-RoadPlus 工程目录", "dir"),
        (config.python_executable, "SAM-RoadPlus Python 解释器", "file"),
        (config.infer_script, "SAM-RoadPlus 推理脚本", "file"),
        (config.config_path, "SAM-RoadPlus config", "file"),
        (config.model_ckpt_path, "SAM-RoadPlus 模型权重", "file"),
        (config.input_image, "输入图像", "file"),
    )
    for value, label, kind in required:
        path = Path(value) if value else Path()
        valid = path.is_dir() if kind == "dir" else path.is_file()
        if not valid:
            errors.append(f"{label}不存在: {value}")
    if config.device not in ("cuda", "cpu"):
        errors.append(f"device 必须为 cuda 或 cpu，当前为: {config.device}")
    if config.tile_size <= config.overlap or config.overlap < 0:
        errors.append("tile_size 必须大于 overlap，且 overlap 不能小于 0")

    if Path(config.config_path).is_file():
        model_config = read_model_config(config.config_path)
        if not model_config:
            errors.append(f"无法读取 SAM-RoadPlus config: {config.config_path}")
        if sam_backbone_required(config.config_path):
            backbone = Path(config.sam_backbone_ckpt_path) if path_is_set(config.sam_backbone_ckpt_path) else Path()
            configured = str(model_config.get("SAM_CKPT_PATH", "") or "").strip()
            configured_path = Path(configured)
            if configured and not configured_path.is_absolute():
                configured_path = Path(config.project_dir) / configured_path
            if not backbone.is_file() and not configured_path.is_file():
                errors.append(
                    "当前 config 需要独立 SAM backbone，但未找到有效权重；"
                    f"字段={config.sam_backbone_ckpt_path}, config.SAM_CKPT_PATH={configured}"
                )
    return errors


def load_samroadplus_config(path: str = DEFAULT_CONFIG_PATH) -> SAMRoadPlusConfig:
    config = SAMRoadPlusConfig()
    source = Path(path)
    if not source.is_file():
        return config
    with source.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    section = data.get("samroadplus", data)
    path_fields = {
        "project_dir": "project_dir",
        "python_exe": "python_executable",
        "infer_script": "infer_script",
        "config_file": "config_path",
        "model_ckpt": "model_ckpt_path",
        "sam_backbone_ckpt": "sam_backbone_ckpt_path",
        "output_dir": "output_dir",
    }
    for yaml_name, attr in path_fields.items():
        if section.get(yaml_name):
            setattr(config, attr, Path(section[yaml_name]))
    for name in ("model_type", "device", "inference_mode"):
        if section.get(name) is not None:
            setattr(config, name, str(section[name]))
    for name in ("auto_import_after_run", "ignore_graph", "skip_black_tile"):
        if section.get(name) is not None:
            setattr(config, name, bool(section[name]))
    for name in ("tile_size", "overlap"):
        if section.get(name) is not None:
            setattr(config, name, int(section[name]))
    return config


def save_samroadplus_config(config: SAMRoadPlusConfig, path: str = DEFAULT_CONFIG_PATH):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    def portable(value):
        return str(value).replace("\\", "/") if path_is_set(value) else ""
    data = {"samroadplus": {
        "model_type": config.model_type,
        "project_dir": portable(config.project_dir),
        "python_exe": portable(config.python_executable),
        "infer_script": portable(config.infer_script),
        "config_file": portable(config.config_path),
        "model_ckpt": portable(config.model_ckpt_path),
        "sam_backbone_ckpt": portable(config.sam_backbone_ckpt_path),
        "output_dir": portable(config.output_dir),
        "device": config.device,
        "auto_import_after_run": config.auto_import_after_run,
        "ignore_graph": config.ignore_graph,
        "inference_mode": config.inference_mode,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "skip_black_tile": config.skip_black_tile,
    }}
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def create_samroadplus_output_dir(base_dir: str, image_path: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = Path(base_dir) / f"samroadplus_{Path(image_path).stem}_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def bridge_script_path() -> Path:
    return Path(__file__).resolve().parent.parent / "samroad_bridge" / "run_samroadplus_bridge.py"


def build_samroadplus_bridge_command(config: SAMRoadPlusConfig, *, check_only=False) -> List[str]:
    cmd = [
        str(config.python_executable), str(bridge_script_path()),
        "--project-dir", str(config.project_dir),
        "--infer-script", str(config.infer_script),
        "--config", str(config.config_path),
        "--checkpoint", str(config.model_ckpt_path),
        "--output-dir", str(config.output_dir),
        "--device", config.device,
    ]
    if config.input_image:
        cmd.extend(("--image", str(config.input_image)))
    if path_is_set(config.sam_backbone_ckpt_path):
        cmd.extend(("--sam-backbone", str(config.sam_backbone_ckpt_path)))
    if config.ignore_graph:
        cmd.append("--ignore-graph")
    if check_only:
        cmd.append("--check-only")
    return cmd


def run_samroadplus_preflight(config: SAMRoadPlusConfig, timeout: int = 180) -> dict:
    """Run strict model construction/state_dict shape validation."""
    config.output_dir = config.output_dir or Path("outputs/samroadplus_preflight")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_samroadplus_bridge_command(config, check_only=True)
    proc = subprocess.run(
        cmd, cwd=str(config.project_dir), capture_output=True, text=True,
        timeout=timeout, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("SAMROADPLUS_CHECK_JSON="):
            try:
                payload = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                pass
            break
    return payload or {
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "return_code": proc.returncode,
    }
