#!/usr/bin/env python3
"""
SAM-Road Checkpoint 结构诊断工具。

用法:
    python tools/inspect_samroad_checkpoint.py \
        --checkpoint D:/sam_road_single_image_share/checkpoints/model.ckpt \
        [--config D:/sam_road_single_image_share/config/toponet_vitb_256_spacenet_4060_night.yaml] \
        [--project-dir D:/sam_road_single_image_share]

功能:
    1. 读取 checkpoint，打印顶层 keys 和关键参数 shape
    2. 如果提供 --config，构建模型并比较 shape
    3. 扫描 project-dir 下的所有 config/*.yaml 和 checkpoints/*.ckpt 文件
    4. 如果存在多个组合，逐个尝试 shape 匹配
    5. 输出推荐组合
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def prepare_import_paths(project_dir: str):
    """将 SAM-Road 项目目录加入 Python search path。"""
    project_dir = Path(project_dir).resolve()
    sam_dir = project_dir / "sam"
    for p in [str(project_dir), str(sam_dir)]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    sep = os.pathsep
    old = os.environ.get("PYTHONPATH", "")
    paths = [str(project_dir), str(sam_dir)]
    if old:
        paths.append(old)
    os.environ["PYTHONPATH"] = sep.join(paths)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.chdir(str(project_dir))


def inspect_checkpoint(ckpt_path: str):
    """纯 torch 加载，不依赖模型代码。"""
    import torch
    print("=" * 60)
    print(f"[INSPECT] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"[INSPECT] Checkpoint top-level keys: {list(ckpt.keys())}")

    # 打印 hyper/extra 信息
    for meta_key in ["hyper_parameters", "config", "args", "hparams", "epoch", "global_step",
                     "optimizer_states", "lr_schedulers", "pytorch-lightning_version"]:
        if meta_key in ckpt:
            val = ckpt[meta_key]
            if isinstance(val, dict):
                print(f"\n[INSPECT] {meta_key} (dict, {len(val)} keys):")
                for k, v in val.items():
                    if isinstance(v, (str, int, float, bool)):
                        print(f"    {k}: {v}")
                    elif isinstance(v, (list, tuple)) and len(v) <= 10:
                        print(f"    {k}: {v}")
                    else:
                        print(f"    {k}: {type(v).__name__}")
            else:
                print(f"[INSPECT] {meta_key}: {val}")

    state_dict = ckpt.get("state_dict", {})
    if not state_dict:
        print("[INSPECT] ERROR: No 'state_dict' found in checkpoint!")
        return

    print(f"\n[INSPECT] state_dict: {len(state_dict)} keys total")

    # ── 关键参数 shape ──
    key_filters = [
        ("topo_net", "topo_net 相关"),
        ("pair_proj", "pair_proj 相关"),
        ("road_decoder", "road_decoder 相关"),
        ("keypoint_decoder", "keypoint_decoder 相关"),
        ("image_encoder", "image_encoder 相关"),
        ("sam_backbone", "SAM backbone 相关"),
        ("embedding", "embedding 相关"),
    ]
    for prefix, label in key_filters:
        matched = {k: v for k, v in state_dict.items() if prefix in k}
        if matched:
            print(f"\n  [{label}] ({len(matched)} params):")
            for k, v in sorted(matched.items()):
                shape = tuple(v.shape)
                print(f"    {k}: {shape}")

    # ── 总体 shape 统计 ──
    print("\n" + "-" * 40)
    print("[INSPECT] 所有 state_dict 参数 shape 一览:")
    for k, v in sorted(state_dict.items()):
        shape = tuple(v.shape)
        print(f"  {k}: {shape}")

    return state_dict


def build_model_with_config(config_path: str, project_dir: str):
    """用给定 config 构建 SAMRoad 模型，返回模型和 config。"""
    from utils import load_config
    from model import SAMRoad
    config = load_config(config_path)
    if config is None:
        raise RuntimeError(
            "Config load failed: config object is None. "
            "Please check yaml path and config parser."
        )
    # 解析相对路径
    _resolve_config_paths(config, project_dir)
    net = SAMRoad(config)
    return net, config


def _resolve_config_paths(config, project_dir: str):
    """将 config 中相对路径转为绝对路径。"""
    fields = ["SAM_CKPT_PATH", "TRAINED_CKPT_PATH", "CKPT_PATH",
              "MODEL_CKPT_PATH", "DATA_ROOT", "IMAGE_DIR", "SAVE_DIR"]
    for fn in fields:
        if hasattr(config, fn):
            val = getattr(config, fn)
            if val is None:
                print(f"[CONFIG] Warning: {fn} is None, skipping path resolve")
                continue
            p = Path(str(val))
            if not p.is_absolute():
                setattr(config, fn, str((Path(project_dir) / p).resolve()))


def compare_shapes(model, checkpoint_state_dict: dict):
    """比较模型参数和 checkpoint 参数 shape，返回不匹配列表。"""
    model_state = dict(model.state_dict())
    mismatches = []
    matches = 0
    only_in_ckpt = set(checkpoint_state_dict.keys()) - set(model_state.keys())
    only_in_model = set(model_state.keys()) - set(checkpoint_state_dict.keys())

    for key in sorted(set(model_state.keys()) & set(checkpoint_state_dict.keys())):
        model_shape = tuple(model_state[key].shape)
        ckpt_shape = tuple(checkpoint_state_dict[key].shape)
        if model_shape == ckpt_shape:
            matches += 1
        else:
            mismatches.append((key, model_shape, ckpt_shape))

    print(f"\n[COMPARE] Matched: {matches}, Mismatched: {len(mismatches)}")
    if only_in_ckpt:
        print(f"[COMPARE] Keys only in checkpoint: {len(only_in_ckpt)}")
    if only_in_model:
        print(f"[COMPARE] Keys only in model: {len(only_in_model)}")

    if mismatches:
        print("\n" + "!" * 60)
        print("[COMPARE] SHAPE MISMATCHES:")
        for key, m_shape, c_shape in mismatches:
            print(f"  {key}")
            print(f"    model:      {m_shape}")
            print(f"    checkpoint: {c_shape}")
        print("!" * 60)
    else:
        print("[COMPARE] OK: All shapes match!")

    return mismatches


def print_config_fields(config):
    """打印 config 对象的所有非内置字段（安全模式：不会调用 None 上的方法）。"""
    print(f"\n{'=' * 60}")
    print("[CONFIG] All config fields:")
    for attr in sorted(dir(config)):
        if attr.startswith("_"):
            continue
        val = getattr(config, attr)
        if callable(val):
            continue
        if val is None:
            print(f"  {attr}: None")
        elif isinstance(val, (str, int, float, bool)):
            print(f"  {attr}: {val}")
        elif isinstance(val, (list, tuple)):
            if len(val) <= 20:
                print(f"  {attr}: {val}")
            else:
                print(f"  {attr}: {type(val).__name__} (len={len(val)})")
        elif hasattr(val, "keys"):
            # dict-like object
            print(f"  {attr}: {type(val).__name__} ({len(val)} keys)")
        else:
            print(f"  {attr}: {type(val).__name__}")


def scan_project(project_dir: str):
    """扫描 project_dir 下的所有 config 和 checkpoint 文件。"""
    project_dir = Path(project_dir)
    configs = sorted(project_dir.glob("config/*.yaml")) + sorted(project_dir.glob("config/*.yml"))
    checkpoints = sorted(project_dir.glob("checkpoints/*.ckpt")) + sorted(project_dir.glob("checkpoints/*.pt")) + \
                  sorted(project_dir.glob("checkpoints/*.pth"))
    print(f"\n[SCAN] Project: {project_dir}")
    print(f"[SCAN] Configs ({len(configs)}):")
    for c in configs:
        print(f"    {c.relative_to(project_dir)}")
    print(f"[SCAN] Checkpoints ({len(checkpoints)}):")
    for ck in checkpoints:
        size_mb = ck.stat().st_size / (1024 * 1024)
        print(f"    {ck.relative_to(project_dir)} ({size_mb:.1f} MB)")
    return configs, checkpoints


def find_matching_pair(configs, checkpoints, project_dir: str):
    """逐个尝试 (config, checkpoint) 组合，找到 shape 匹配的组合。"""
    import torch
    print(f"\n{'=' * 60}")
    print("[MATCH] Scanning all (config, checkpoint) combinations...")

    results = []
    for cfg_path in configs:
        print(f"\n  >> Trying config: {cfg_path.name}")
        try:
            net, config = build_model_with_config(str(cfg_path), project_dir)
            model_sd = dict(net.state_dict())
        except Exception as e:
            print(f"    X Failed to build model: {e}")
            continue

        for ckpt_path in checkpoints:
            print(f"      Testing checkpoint: {ckpt_path.name} ... ", end="")
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                ckpt_sd = ckpt.get("state_dict", {})
                if not ckpt_sd:
                    print("No state_dict, skip")
                    continue

                mismatches = []
                for key in sorted(set(model_sd.keys()) & set(ckpt_sd.keys())):
                    if tuple(model_sd[key].shape) != tuple(ckpt_sd[key].shape):
                        mismatches.append(key)
            except Exception as e:
                print(f"Error: {e}")
                continue

            if mismatches:
                print(f"{len(mismatches)} mismatches")
                # 打印前 3 个不匹配的关键参数
                for mk in mismatches[:3]:
                    print(f"          {mk}: model {tuple(model_sd[mk].shape)} vs ckpt {tuple(ckpt_sd[mk].shape)}")
            else:
                print("OK: MATCH!")
                results.append((str(cfg_path), str(ckpt_path), 0))

    if results:
        print("\n" + "=" * 60)
        print("[MATCH] Found matching combination(s):")
        for cfg, ck, _ in results:
            print(f"  Config:     {cfg}")
            print(f"  Checkpoint: {ck}")
            print()
    else:
        print("\n" + "!" * 60)
        print("[MATCH] NO matching (config, checkpoint) pair found!")
        print("[MATCH] The checkpoint was likely trained with a different config.")
        print("[MATCH] You need to obtain the matching config or re-train the model.")
        print("!" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Inspect SAM-Road checkpoint structure")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--config", default=None, help="Path to config .yaml file (optional, for model comparison)")
    parser.add_argument("--project-dir", default="D:/sam_road_single_image_share",
                        help="SAM-Road project root for sys.path setup")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all configs and checkpoints in project-dir")
    parser.add_argument("--match", action="store_true",
                        help="Brute-force try all (config, checkpoint) pairs")
    args = parser.parse_args()

    # ★ 必须在 import model 之前设置路径
    prepare_import_paths(args.project_dir)

    # 1. 检查 checkpoint 结构
    print("\n" + "=" * 60)
    print("PHASE 1: Checkpoint Structure")
    print("=" * 60)
    state_dict = inspect_checkpoint(args.checkpoint)

    # 2. 打印 config 内容 / 构建模型比较
    if args.config:
        print("\n" + "=" * 60)
        print("PHASE 2: Config + Model Shape Comparison")
        print("=" * 60)
        try:
            net, config = build_model_with_config(args.config, args.project_dir)
            print_config_fields(config)
            compare_shapes(net, state_dict)
        except Exception as e:
            print(f"[ERROR] Failed to build model with config: {e}")

    # 3. 扫描项目文件
    if args.scan or args.match:
        configs, checkpoints = scan_project(args.project_dir)

    # 4. 自动匹配
    if args.match:
        find_matching_pair(configs, checkpoints, args.project_dir)


if __name__ == "__main__":
    main()
