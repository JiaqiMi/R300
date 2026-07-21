#!/usr/bin/env python3
"""
SAM-Road Runner 模块。

负责管理 SAM-Road 外部推理的调用，包括：
1. 配置读取与验证
2. QProcess 异步调用（不阻塞 GUI）
3. 日志记录
4. dry-run / mock 模式支持

设计原则：
- 使用 QProcess 而非 subprocess，保证 GUI 不卡死
- 所有路径使用 pathlib.Path 处理
- 不使用 shell=True
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import yaml


# ===================================================================
# 配置数据结构
# ===================================================================

@dataclass
class SAMRoadRunConfig:
    """一次 SAM-Road 运行的完整配置。"""
    project_dir: Path = Path("D:/sam_road-main")
    python_executable: Path = Path()
    bridge_script: Path = Path("samroad_bridge/run_samroad_bridge.py")
    config_file: Path = Path("config/toponet_vitb_512_cityscale.yaml")
    sam_backbone_ckpt_path: Path = Path()
    samroad_model_ckpt_path: Path = Path()
    input_image: Path = Path()
    output_dir: Path = Path()
    tile_size: int = 1024
    overlap: int = 128
    device: str = "cuda"
    auto_import_after_run: bool = True
    dry_run: bool = False
    run_mode: str = "bridge"  # "bridge" | "direct"
    mask_only_partial_load: bool = False

    @property
    def is_valid(self) -> bool:
        """检查必要字段是否填写。"""
        return (
            self.project_dir.is_dir()
            and self.input_image.is_file()
            and self.output_dir != Path()
        )


# ===================================================================
# 配置持久化
# ===================================================================

def load_config(config_path: str) -> dict:
    """从 YAML 文件加载 SAM-Road 配置。"""
    path = Path(config_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_config(config_path: str, config: dict):
    """保存 SAM-Road 配置到 YAML 文件。"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def dict_to_runconfig(data: dict) -> SAMRoadRunConfig:
    """从字典加载运行配置。"""
    sr = data.get("samroad", {})
    c = SAMRoadRunConfig()

    if sr.get("project_dir"):
        c.project_dir = Path(sr["project_dir"])
    if sr.get("python_executable"):
        c.python_executable = Path(sr["python_executable"])
    if sr.get("bridge_script"):
        c.bridge_script = Path(sr["bridge_script"])
    if sr.get("config_file"):
        c.config_file = Path(sr["config_file"])
    if sr.get("sam_backbone_ckpt_path"):
        c.sam_backbone_ckpt_path = Path(sr["sam_backbone_ckpt_path"])
    if sr.get("samroad_model_ckpt_path"):
        c.samroad_model_ckpt_path = Path(sr["samroad_model_ckpt_path"])
    # 向后兼容旧字段名
    if sr.get("checkpoint_path") and not sr.get("samroad_model_ckpt_path"):
        c.samroad_model_ckpt_path = Path(sr["checkpoint_path"])
    if sr.get("tile_size"):
        c.tile_size = int(sr["tile_size"])
    if sr.get("overlap"):
        c.overlap = int(sr["overlap"])
    if sr.get("device"):
        c.device = str(sr["device"])
    if sr.get("auto_import_after_run") is not None:
        c.auto_import_after_run = bool(sr["auto_import_after_run"])
    if sr.get("dry_run") is not None:
        c.dry_run = bool(sr["dry_run"])
    if sr.get("run_mode"):
        c.run_mode = str(sr["run_mode"])
    if sr.get("mask_only_partial_load") is not None:
        c.mask_only_partial_load = bool(sr["mask_only_partial_load"])

    return c


def runconfig_to_dict(config: SAMRoadRunConfig) -> dict:
    """将运行配置序列化为字典（用于 YAML 保存）。"""
    return {
        "samroad": {
            "project_dir": str(config.project_dir).replace("\\", "/"),
            "python_executable": str(config.python_executable).replace("\\", "/"),
            "bridge_script": str(config.bridge_script).replace("\\", "/"),
            "config_file": str(config.config_file).replace("\\", "/"),
            "sam_backbone_ckpt_path": str(config.sam_backbone_ckpt_path).replace("\\", "/"),
            "samroad_model_ckpt_path": str(config.samroad_model_ckpt_path).replace("\\", "/"),
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "device": config.device,
            "auto_import_after_run": config.auto_import_after_run,
            "dry_run": config.dry_run,
            "run_mode": config.run_mode,
            "mask_only_partial_load": config.mask_only_partial_load,
        }
    }


# ===================================================================
# SAM-Road 项目扫描
# ===================================================================

def scan_entry_scripts(project_dir: str) -> list[Path]:
    """扫描 SAM-Road 项目目录中可能的推理入口脚本。"""
    candidates = [
        "inferencer.py",
        "inference.py",
        "test.py",
        "main.py",
        "sam_road.py",
        "run_samroad_bridge.py",
    ]
    project = Path(project_dir)
    found = []
    for name in candidates:
        path = project / name
        if path.is_file():
            found.append(path)
    return found


def scan_checkpoints(project_dir: str) -> list[Path]:
    """扫描 SAM-Road 项目目录中的模型权重文件。"""
    pattern_list = ["*.ckpt", "*.pth", "*.pt"]
    project = Path(project_dir)
    found = []
    for pattern in pattern_list:
        found.extend(project.rglob(pattern))
    return found


# ===================================================================
# 验证
# ===================================================================

def validate_config(config: SAMRoadRunConfig) -> list[str]:
    """验证运行配置，返回错误信息列表。"""
    errors = []

    if not config.python_executable or not Path(config.python_executable).is_file():
        errors.append(f"Python 解释器未找到: {config.python_executable}")

    if config.run_mode == "bridge" and not config.bridge_script:
        errors.append("Bridge 脚本路径为空")

    if not config.input_image or not Path(config.input_image).is_file():
        errors.append(f"输入图像未找到: {config.input_image}")

    if not config.dry_run:
        if not config.samroad_model_ckpt_path or not Path(config.samroad_model_ckpt_path).is_file():
            errors.append(f"SAM-Road 模型权重未找到: {config.samroad_model_ckpt_path}")
        if not config.sam_backbone_ckpt_path or not Path(config.sam_backbone_ckpt_path).is_file():
            errors.append(f"SAM backbone 权重未找到: {config.sam_backbone_ckpt_path}")
        if not config.project_dir.is_dir():
            errors.append(f"SAM-Road 项目目录不存在: {config.project_dir}")

    return errors


# ===================================================================
# 命令构造
# ===================================================================

def build_command(config: SAMRoadRunConfig) -> list[str]:
    """构造 QProcess / subprocess 的命令参数列表（不使用 shell=True）。"""
    if config.run_mode == "bridge":
        bridge_full = config.bridge_script
        if not bridge_full.is_absolute():
            # 相对路径相对于当前工作目录
            bridge_full = Path.cwd() / bridge_full

        cmd = [
            str(config.python_executable),
            str(bridge_full),
            "--samroad-project", str(config.project_dir),
            "--image", str(config.input_image),
            "--sam-backbone-checkpoint", str(config.sam_backbone_ckpt_path) if not config.dry_run else "mock",
            "--checkpoint", str(config.samroad_model_ckpt_path) if not config.dry_run else "mock",
            "--output", str(config.output_dir),
            "--config", str(config.config_file),
            "--device", config.device,
        ]

        if config.dry_run:
            cmd.append("--dry-run")
        if config.mask_only_partial_load:
            cmd.append("--mask-only-partial-load")

        return cmd

    # direct 模式暂未实现
    raise NotImplementedError(f"Unsupported run_mode: {config.run_mode}")


# ===================================================================
# 输出目录管理
# ===================================================================

def create_output_dir(base_dir: str, image_path: str) -> Path:
    """基于图像名创建 SAM-Road 输出目录。

    命名规则: <base_dir>/samroad_<image_stem>_<timestamp>/
    """
    base = Path(base_dir)
    image_stem = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"samroad_{image_stem}_{timestamp}"
    output_dir = base / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ===================================================================
# 运行结果数据结构
# ===================================================================

@dataclass
class SAMRoadRunResult:
    """一次 SAM-Road 运行的结果。"""
    success: bool = False
    return_code: int = -1
    output_dir: Path = field(default_factory=Path)
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    is_dry_run: bool = False
    is_partial_load: bool = False
    partial_load_warning: str = ""
    output_diagnostics: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    found_files: list[str] = field(default_factory=list)

    @classmethod
    def from_process_result(
        cls,
        return_code: int,
        output_dir: Path,
        stdout: str,
        stderr: str,
        is_dry_run: bool = False,
    ) -> "SAMRoadRunResult":
        result = cls(
            success=(return_code == 0 and (
                is_dry_run or (Path(output_dir) / "road_mask.png").is_file()
            )),
            return_code=return_code,
            output_dir=output_dir,
            stdout=stdout,
            stderr=stderr,
            is_dry_run=is_dry_run,
        )

        # 尝试从 metadata 读取额外信息
        metadata_path = output_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                result.node_count = meta.get("node_count", 0)
                result.edge_count = meta.get("edge_count", 0)
                result.elapsed_seconds = meta.get("elapsed_seconds", 0)
                result.is_partial_load = meta.get("partial_load", False)
                result.partial_load_warning = meta.get("partial_load_warning", "")
            except Exception:
                pass

        # 列出输出文件
        if output_dir.is_dir():
            result.found_files = sorted(
                str(path.relative_to(output_dir)).replace("\\", "/")
                for path in output_dir.rglob("*") if path.is_file()
            )

        return result
