#!/usr/bin/env python3
"""
SAM-Road 桥接脚本 — 统一 RoadNet Studio 和 SAM-Road 之间的输入/输出接口。

此脚本被 RoadNet Studio 通过 QProcess/subprocess 异步调用，
负责调用 SAM-Road 的推理逻辑并将结果输出为 RoadNet Studio 可识别的格式。

用法:
    python run_samroad_bridge.py \
        --samroad-project D:/sam_road-main \
        --image <input_image.jpg> \
        --sam-backbone-checkpoint <sam_vit_b_01ec64.pth> \
        --checkpoint <model.ckpt> \
        --output <output_dir> \
        --config config/toponet_vitb_512_cityscale.yaml \
        --device cuda \
        [--dry-run]

输出目录结构:
    <output_dir>/
        road_mask_raw.png               ← 道路 mask (0/255 二值)
        road_mask_samroad_score.png     ← 道路得分图 (0-255)
        keypoint_mask_samroad_score.png ← keypoint 得分图
        draft_graph.json                ← 提取的 draft graph
        draft_graph_overlay.png         ← graph 叠加可视化
        road_skeleton.png               ← skeleton (如果可生成)
        metadata.json                   ← 运行元数据
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# ===================================================================
# Runtime 环境准备
# 必须在 import model / matplotlib / model.py 之前调用。
# 解决 Windows + 中文用户名 + QProcess 环境变量不完整时
# matplotlib 无法确定 home 目录的问题。
# ===================================================================

def prepare_runtime_env(output_dir=None):
    """在导入 SAM-Road model 前创建安全的运行环境。"""
    base = Path(output_dir or "outputs/samroad_runtime").resolve()
    runtime_dir = base / "_runtime"
    mpl_dir = runtime_dir / "matplotlib"
    home_dir = runtime_dir / "home"

    mpl_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)

    # matplotlib 后端必须在 import pyplot 之前设置
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    # Windows 下避免 pathlib.Path.home() 因中文用户名或环境缺失失败
    os.environ.setdefault("HOME", str(home_dir))
    os.environ.setdefault("USERPROFILE", str(home_dir))

    # Python / Qt 日志统一 UTF-8
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # ★ 禁用 wandb 在线模式，防止模型初始化时失败
    os.environ.setdefault("WANDB_MODE", "disabled")

    # 确保 matplotlib 使用 Agg 后端
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass


def prepare_project_import_paths(project_dir):
    """将 SAM-Road 项目目录及其 sam 子目录加入 Python 搜索路径。

    必须在 import model / segment_anything 之前调用。

    场景：
        D:/sam_road_single_image_share/
            ├── model.py          # from sam.segment_anything.modeling...
            └── sam/
                └── segment_anything/
                    └── predictor.py  # from segment_anything.modeling import Sam (绝对导入!)

    因此 sys.path / PYTHONPATH 必须同时包含：
        - D:/sam_road_single_image_share      (让 from model import SAMRoad 生效)
        - D:/sam_road_single_image_share/sam  (让 from segment_anything.modeling import Sam 生效)

    返回错误信息列表（空 = 成功）。
    """
    project_dir = Path(project_dir).resolve()
    sam_dir = project_dir / "sam"
    errors = []

    if not project_dir.is_dir():
        errors.append(f"Project directory not found: {project_dir}")
    if not sam_dir.is_dir():
        errors.append(f"sam directory not found: {sam_dir}")

    for p in [project_dir, sam_dir]:
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)

    # 同步 PYTHONPATH
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [str(project_dir), str(sam_dir)]
    if old_pythonpath:
        paths.append(old_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)

    return errors


def verify_segment_anything_import(project_dir):
    """在正式 import model 前做自检，确保 segment_anything 可被导入。"""
    project_dir = Path(project_dir).resolve()
    sam_dir = project_dir / "sam"

    print(f"[BRIDGE] project_dir: {project_dir}")
    print(f"[BRIDGE] sam_dir: {sam_dir}")
    print(f"[BRIDGE] sys.path head: {sys.path[:5]}")
    print(f"[BRIDGE] PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

    try:
        import segment_anything
        print(f"[BRIDGE] segment_anything import OK: {segment_anything.__file__}")
        return True
    except ModuleNotFoundError:
        print("[BRIDGE] ERROR: 无法导入 segment_anything。", file=sys.stderr)
        print(f"[BRIDGE] 请检查 {sam_dir}/segment_anything 是否存在，", file=sys.stderr)
        print(f"[BRIDGE] 并确认 {sam_dir} 已加入 PYTHONPATH。", file=sys.stderr)
        return False


def parse_args():
    p = argparse.ArgumentParser(description="SAM-Road Bridge for RoadNet Studio")
    p.add_argument("--samroad-project", required=True, help="SAM-Road project root directory")
    p.add_argument("--image", required=True, help="Input satellite image (JPG/PNG)")
    p.add_argument("--checkpoint", required=True, help="SAM-Road model checkpoint (.ckpt) — model.ckpt for net.load_state_dict()")
    p.add_argument("--sam-backbone-checkpoint", required=True,
                   help="SAM ViT-B backbone weights (.pth) — sam_vit_b_01ec64.pth for config.SAM_CKPT_PATH")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--config", default="config/toponet_vitb_256_spacenet_4060_night.yaml",
                   help="Config file (relative to samroad-project)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true", help="Mock mode: skip actual inference")
    p.add_argument("--mask-only-partial-load", action="store_true",
                   help="Debug: use strict=False to load checkpoint partially, only for mask testing (topo graph unreliable)")
    return p.parse_args()


def validate_paths(args) -> list[str]:
    errors = []
    project_dir = Path(args.samroad_project)
    if not project_dir.is_dir():
        errors.append(f"SAM-Road project dir not found: {project_dir}")
    image_path = Path(args.image)
    if not image_path.is_file():
        errors.append(f"Input image not found: {image_path}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        errors.append(f"SAM-Road model checkpoint not found: {ckpt_path}")
    backbone_path = Path(args.sam_backbone_checkpoint)
    if not backbone_path.is_file():
        errors.append(f"SAM backbone weights not found: {backbone_path}")
    return errors


def load_matching_state_dict(model, checkpoint_state_dict):
    """安全地加载 checkpoint，只加载 shape 匹配的参数。

    与 strict=False 的区别：
    - strict=False 会默默跳过 missing/unexpected key（全部加载 or 部分失败）
    - 本函数精确过滤：只加载 key 存在且 shape 完全一致的参数

    Returns:
        tuple[list[dict], list[str], list[str]]:
            - skipped: 被跳过的参数列表 [{key, ckpt_shape, model_shape}]
            - missing: model 中有但 ckpt 没有的参数 (来自 load_state_dict strict=False)
            - unexpected: ckpt 中有但 model 没有的参数
    """
    import torch
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[dict] = []

    for k, v in checkpoint_state_dict.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append({
                "key": k,
                "ckpt_shape": list(v.shape) if hasattr(v, "shape") else None,
                "model_shape": list(model_state[k].shape) if k in model_state else None,
            })

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return skipped, missing, unexpected


def resolve_project_path(project_dir, path_value):
    """将相对于 project_dir 的路径转为绝对路径。

    Args:
        project_dir: 项目根目录（Path 或 str）
        path_value: 需要解析的路径值（str 或 PathLike）
    Returns:
        str: 绝对路径
    """
    p = Path(str(path_value))
    if p.is_absolute():
        return str(p)
    return str((Path(project_dir) / p).resolve())


def run_inference(args) -> dict:
    """调用 SAM-Road 真实推理逻辑，返回输出文件路径字典。"""
    project_dir = Path(args.samroad_project)

    # 确保项目路径在 sys.path 中（防止 main() 漏调）
    prepare_project_import_paths(str(project_dir))

    # 自检：segment_anything 必须可导入
    if not verify_segment_anything_import(str(project_dir)):
        raise ImportError(
            f"无法导入 segment_anything。请检查 {project_dir}/sam/segment_anything 是否存在。"
        )

    import numpy as np
    import cv2
    import torch

    from utils import load_config
    from model import SAMRoad

    # 加载配置
    config_path = project_dir / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(str(config_path))

    # ★ 将 config 中所有相对路径转为基于 project_dir 的绝对路径
    _relative_path_fields = [
        "SAM_CKPT_PATH",
        "TRAINED_CKPT_PATH",
        "CKPT_PATH",
        "MODEL_CKPT_PATH",
        "DATA_ROOT",
        "IMAGE_DIR",
        "SAVE_DIR",
    ]
    for field_name in _relative_path_fields:
        if hasattr(config, field_name):
            old_val = str(getattr(config, field_name))
            new_val = resolve_project_path(str(project_dir), old_val)
            if old_val != new_val:
                setattr(config, field_name, new_val)
                print(f"[BRIDGE] Config path resolved: {field_name}: {old_val} -> {new_val}")

    # ★★★★★ 路径覆盖：SAM backbone 权重使用命令行传入的路径 ★★★★★
    if hasattr(args, "sam_backbone_checkpoint") and args.sam_backbone_checkpoint:
        config.SAM_CKPT_PATH = str(Path(args.sam_backbone_checkpoint).resolve())
        print(f"[BRIDGE] Overriding config.SAM_CKPT_PATH with cli arg: {config.SAM_CKPT_PATH}")

    # ★★★★★ 配置与 checkpoint 诊断 ★★★★★
    print(f"[BRIDGE] project_dir: {project_dir}")
    print(f"[BRIDGE] cwd: {os.getcwd()}")
    print(f"[BRIDGE] config_path: {config_path}")
    print(f"[BRIDGE] SAM backbone checkpoint (cli): {args.sam_backbone_checkpoint}")
    print(f"[BRIDGE] SAM backbone checkpoint resolved: {config.SAM_CKPT_PATH}")
    print(f"[BRIDGE] SAM backbone exists: {Path(config.SAM_CKPT_PATH).exists()}")
    print(f"[BRIDGE] SAM-Road model checkpoint (cli): {args.checkpoint}")
    print(f"[BRIDGE] SAM-Road model checkpoint exists: {Path(args.checkpoint).exists()}")

    # 打印 config 所有非内置字段（安全模式：不调用 None 上的方法）
    print(f"[BRIDGE] === Config fields dump ===")
    for attr in sorted(dir(config)):
        if attr.startswith("_") or callable(getattr(config, attr, None)):
            continue
        val = getattr(config, attr)
        if val is None:
            print(f"[BRIDGE]   {attr}: None")
        elif isinstance(val, (str, int, float, bool)):
            print(f"[BRIDGE]   {attr}: {val}")
        elif isinstance(val, (list, tuple)):
            if len(val) <= 20:
                print(f"[BRIDGE]   {attr}: {val}")
            else:
                print(f"[BRIDGE]   {attr}: {type(val).__name__} (len={len(val)})")
        elif hasattr(val, "keys"):
            print(f"[BRIDGE]   {attr}: {type(val).__name__} ({len(val)} keys)")
        else:
            print(f"[BRIDGE]   {attr}: {type(val).__name__}")
    print(f"[BRIDGE] === Config dump end ===")

    if not Path(config.SAM_CKPT_PATH).exists():
        raise FileNotFoundError(
            f"SAM backbone checkpoint not found: {config.SAM_CKPT_PATH}. "
            f"Expected file exists at {project_dir}/sam_ckpts/sam_vit_b_01ec64.pth"
        )

    # 构建设备
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Bridge] Using device: {device}")

    # 构建模型
    net = SAMRoad(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # ★★★★★ Shape 预检查：在 load_state_dict 之前比较关键参数 shape ★★★★★
    ckpt_sd = checkpoint.get("state_dict", {})
    model_sd = dict(net.state_dict())
    _mismatches = []
    _matched = 0
    for key in sorted(set(model_sd.keys()) & set(ckpt_sd.keys())):
        m_shape = tuple(model_sd[key].shape)
        c_shape = tuple(ckpt_sd[key].shape)
        if m_shape == c_shape:
            _matched += 1
        else:
            _mismatches.append((key, m_shape, c_shape))

    print(f"[BRIDGE] Shape comparison: {_matched} matched, {len(_mismatches)} mismatched")
    if _mismatches:
        print(f"[BRIDGE] === Shape mismatches ===")
        for key, m_shape, c_shape in _mismatches[:20]:
            print(f"[BRIDGE]   {key}")
            print(f"[BRIDGE]     model:      {m_shape}")
            print(f"[BRIDGE]     checkpoint: {c_shape}")
        if len(_mismatches) > 20:
            print(f"[BRIDGE]   ... and {len(_mismatches) - 20} more")

        if args.mask_only_partial_load:
            print(f"[BRIDGE] ==============================================")
            print(f"[BRIDGE] WARNING: --mask-only-partial-load enabled")
            print(f"[BRIDGE] Partial load — only for mask testing, topo graph results are UNRELIABLE!")
            print(f"[BRIDGE] ==============================================")
            skipped, missing, unexpected = load_matching_state_dict(net, ckpt_sd)
            print(f"[BRIDGE] Partial load: {len(skipped)} params skipped, {len(missing)} missing, {len(unexpected)} unexpected")
            if skipped:
                print(f"[BRIDGE] --- Skipped parameters (shape mismatch) ---")
                for s in skipped[:30]:
                    print(f"[BRIDGE]   SKIP: {s['key']}")
                    print(f"[BRIDGE]         ckpt_shape={s['ckpt_shape']}, model_shape={s['model_shape']}")
                if len(skipped) > 30:
                    print(f"[BRIDGE]   ... and {len(skipped) - 30} more skipped")
            # ★ 保存 skipped 信息到 results 中，供 save_outputs 标注
            run_inference._partial_load_skipped = skipped
            run_inference._partial_load_missing = list(missing)
            run_inference._partial_load_unexpected = list(unexpected)
        else:
            raise RuntimeError(
                f"Config and checkpoint do not match: {len(_mismatches)} parameter shapes differ.\n"
                f"First mismatch: {_mismatches[0][0]}: model {_mismatches[0][1]} vs ckpt {_mismatches[0][2]}\n"
                f"Please use a matching config-checkpoint pair, or use --mask-only-partial-load for debug.\n"
                f"Run: python tools/inspect_samroad_checkpoint.py --checkpoint ... --config ... --match\n"
                f"to automatically find matching combinations."
            )
    else:
        print(f"[Bridge] Loading checkpoint: {args.checkpoint}")
        net.load_state_dict(ckpt_sd, strict=True)
    net.eval()
    net.to(device)

    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Cannot read image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"[Bridge] Image size: {img_rgb.shape[0]}x{img_rgb.shape[1]}")

    # 调用 inferencer 中的推理函数
    from inferencer import infer_one_img

    start = time.time()
    pred_nodes, pred_edges, keypoint_mask, road_mask = infer_one_img(net, img_rgb, config)
    elapsed = time.time() - start
    print(f"[Bridge] Inference completed in {elapsed:.1f}s")
    print(f"[Bridge] Nodes: {len(pred_nodes) if len(pred_nodes.shape) > 0 else 0}, "
          f"Edges: {len(pred_edges) if len(pred_edges.shape) > 0 else 0}")

    result = {
        "pred_nodes": pred_nodes,
        "pred_edges": pred_edges,
        "keypoint_mask": keypoint_mask,
        "road_mask": road_mask,
        "elapsed_seconds": elapsed,
        "image_shape": img_rgb.shape[:2],  # (H, W)
        "partial_load": args.mask_only_partial_load,
    }
    # 附加 partial load 诊断信息
    if args.mask_only_partial_load:
        result["partial_skipped"] = getattr(run_inference, "_partial_load_skipped", [])
        result["partial_missing"] = getattr(run_inference, "_partial_load_missing", [])
        result["partial_unexpected"] = getattr(run_inference, "_partial_load_unexpected", [])
    return result


def save_outputs(results: dict, output_dir: Path, args):
    """将推理结果保存为 RoadNet Studio 可识别的文件格式。"""
    import numpy as np
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    road_mask = results["road_mask"]        # uint8, 0-255
    keypoint_mask = results["keypoint_mask"] # uint8, 0-255
    pred_nodes = results["pred_nodes"]       # (rc format) [N, 2]
    pred_edges = results["pred_edges"]       # [N, 2] index pairs
    img_h, img_w = results["image_shape"]

    # ── 1. 保存 road mask 原始版 ──
    mask_bin = (road_mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(output_dir / "road_mask_raw.png"), mask_bin)
    print(f"[Bridge] Saved: road_mask_raw.png ({mask_bin.shape[1]}x{mask_bin.shape[0]})")

    # ── 2. 保存 road mask 得分图 ──
    score_img = road_mask.astype(np.uint8)
    cv2.imwrite(str(output_dir / "road_mask_samroad_score.png"), score_img)
    print(f"[Bridge] Saved: road_mask_samroad_score.png")

    # ── 3. 保存 keypoint 得分图 ──
    cv2.imwrite(str(output_dir / "keypoint_mask_samroad_score.png"), keypoint_mask)
    print(f"[Bridge] Saved: keypoint_mask_samroad_score.png")

    # ── 4. 保存 draft_graph.json ──
    # SAM-Road 输出节点格式是 (rc)，转换为 RoadNet Studio draft graph 格式 (x, y)
    node_count = 0
    edge_count = 0
    draft_nodes = []
    if pred_nodes is not None and len(pred_nodes.shape) == 2 and pred_nodes.shape[0] > 0:
        node_count = pred_nodes.shape[0]
        for i in range(node_count):
            r, c = pred_nodes[i, 1], pred_nodes[i, 0]  # SAM-Road 输出是 (row, col)
            draft_nodes.append({
                "id": i,
                "x": float(c),
                "y": float(r),
                "type": "junction",
            })

    draft_edges = []
    if pred_edges is not None and len(pred_edges.shape) == 2 and pred_edges.shape[0] > 0:
        edge_count = pred_edges.shape[0]
        for i in range(edge_count):
            src, tgt = int(pred_edges[i, 0]), int(pred_edges[i, 1])
            # 构建 path: [[y_src, x_src], [y_tgt, x_tgt]]
            if src < node_count and tgt < node_count:
                path = [
                    [draft_nodes[src]["y"], draft_nodes[src]["x"]],
                    [draft_nodes[tgt]["y"], draft_nodes[tgt]["x"]],
                ]
                draft_edges.append({
                    "id": i,
                    "from": src,
                    "to": tgt,
                    "length_px": 0.0,
                    "path": path,
                })

    is_partial = results.get("partial_load", False)
    draft_graph = {
        "nodes": draft_nodes,
        "edges": draft_edges,
        "metadata": {
            "generator": "SAM-Road Bridge for RoadNet Studio",
            "image_size": {"width": img_w, "height": img_h},
            "node_count": node_count,
            "edge_count": edge_count,
            "config": args.config,
            "device": args.device,
            "created_at": datetime.now().isoformat(),
            "partial_load": is_partial,
            "partial_load_warning": (
                "PARTIAL LOAD GRAPH — topo graph is UNRELIABLE! "
                "Only road_mask and keypoint_mask are valid."
            ) if is_partial else "",
        },
    }

    graph_path = output_dir / "draft_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(draft_graph, f, indent=2, ensure_ascii=False)
    print(f"[Bridge] Saved: draft_graph.json ({node_count} nodes, {edge_count} edges)")

    # ── 5. 保存 overlay 可视化 ──
    try:
        from roadnet.samroad_adapter import load_samroad_output
        overlay_dir = str(output_dir)
        samroad_out = load_samroad_output(overlay_dir)
        if samroad_out.has_graph and samroad_out.has_mask:
            import cv2
            overlay_img = cv2.cvtColor(samroad_out.mask_raw, cv2.COLOR_GRAY2BGR)
            for node in samroad_out.nodes:
                cv2.circle(overlay_img, (int(node["x"]), int(node["y"])), 3, (0, 255, 0), -1)
            for edge in samroad_out.edges:
                pts = edge.get("points_pixel", [])
                if len(pts) >= 2:
                    p0 = (int(pts[0][0]), int(pts[0][1]))
                    p1 = (int(pts[-1][0]), int(pts[-1][1]))
                    cv2.line(overlay_img, p0, p1, (255, 255, 255), 2)
            cv2.imwrite(str(output_dir / "draft_graph_overlay.png"), overlay_img)
            # partial load: 在 overlay 上叠加警告文字
            if is_partial:
                cv2.putText(overlay_img, "PARTIAL LOAD - GRAPH UNRELIABLE",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(str(output_dir / "draft_graph_overlay.png"), overlay_img)
            print(f"[Bridge] Saved: draft_graph_overlay.png")
    except Exception as e:
        print(f"[Bridge] Warning: Could not generate overlay: {e}")

    # ── 6. 保存 metadata.json ──
    is_partial = results.get("partial_load", False)
    metadata = {
        "generator": "SAM-Road Bridge for RoadNet Studio",
        "image": str(args.image),
        "image_shape": [img_h, img_w],
        "node_count": node_count,
        "edge_count": edge_count,
        "elapsed_seconds": results.get("elapsed_seconds", 0),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "created_at": datetime.now().isoformat(),
        "partial_load": is_partial,
    }
    if is_partial:
        skipped = results.get("partial_skipped", [])
        metadata["partial_load_warning"] = (
            "PARTIAL LOAD — only for road_mask/kp_mask testing. "
            "Topo graph (draft_graph.json) is UNRELIABLE. "
            f"Skipped {len(skipped)} mismatched params."
        )
        if skipped:
            metadata["partial_skipped_keys"] = [s["key"] for s in skipped[:50]]
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[Bridge] Saved: metadata.json")

    # ── 7. partial load 模式：写显式警告文件 ──
    if is_partial:
        warning_text = (
            "WARNING: PARTIAL LOAD MODE\n"
            "==========================\n"
            "This output was generated with --mask-only-partial-load.\n"
            "Only road_mask_raw.png and keypoint_mask_samroad_score.png are valid.\n"
            "draft_graph.json, draft_graph_overlay.png are UNRELIABLE — do NOT use as reference graph.\n"
            "\n"
            f"Skipped {len(results.get('partial_skipped', []))} mismatched parameters:\n"
        )
        for s in results.get("partial_skipped", [])[:50]:
            warning_text += f"  - {s['key']}: ckpt={s['ckpt_shape']} vs model={s['model_shape']}\n"
        (output_dir / "partial_load_warning.txt").write_text(warning_text, encoding="utf-8")
        print(f"[Bridge] Saved: partial_load_warning.txt")


def run_dry(args):
    """Dry-run: 生成 mock 输出，标记为 MOCK，不调用真实模型。"""
    import numpy as np
    import cv2

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Cannot read image: {args.image}")
    img_h, img_w = img.shape[:2]

    print("=" * 60)
    print("  DRY RUN / MOCK OUTPUT, NOT REAL SAM-ROAD INFERENCE")
    print("=" * 60)

    # ── Mock mask: 全黑 ──
    mock_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "road_mask_raw.png"), mock_mask)
    cv2.imwrite(str(output_dir / "road_mask_samroad_score.png"), mock_mask)
    cv2.imwrite(str(output_dir / "keypoint_mask_samroad_score.png"), mock_mask)

    # ── Mock graph: 极小测试图 ──
    mock_nodes = [
        {"id": 0, "x": float(img_w * 0.2), "y": float(img_h * 0.5), "type": "junction"},
        {"id": 1, "x": float(img_w * 0.8), "y": float(img_h * 0.5), "type": "junction"},
    ]
    mock_edges = [
        {"id": 0, "from": 0, "to": 1, "length_px": 0.0,
         "path": [[mock_nodes[0]["y"], mock_nodes[0]["x"]],
                  [mock_nodes[1]["y"], mock_nodes[1]["x"]]]},
    ]
    draft_graph = {
        "nodes": mock_nodes,
        "edges": mock_edges,
        "metadata": {
            "generator": "SAM-Road Bridge DRY-RUN (MOCK)",
            "image_size": {"width": img_w, "height": img_h},
            "node_count": 2,
            "edge_count": 1,
            "dry_run": True,
            "created_at": datetime.now().isoformat(),
        },
    }
    with open(output_dir / "draft_graph.json", "w", encoding="utf-8") as f:
        json.dump(draft_graph, f, indent=2)

    # ── Mock overlay ──
    overlay = cv2.cvtColor(mock_mask, cv2.COLOR_GRAY2BGR)
    for n in mock_nodes:
        cv2.circle(overlay, (int(n["x"]), int(n["y"])), 5, (0, 0, 255), -1)
    cv2.line(overlay, (int(mock_nodes[0]["x"]), int(mock_nodes[0]["y"])),
             (int(mock_nodes[1]["x"]), int(mock_nodes[1]["y"])), (0, 0, 255), 2)
    cv2.imwrite(str(output_dir / "draft_graph_overlay.png"), overlay)

    print(f"[Bridge] MOCK outputs saved to: {output_dir}")
    print(f"[Bridge] Files: road_mask_raw.png, draft_graph.json, draft_graph_overlay.png")


def main():
    args = parse_args()

    # 验证路径
    errors = []
    if not args.dry_run:
        errors = validate_paths(args)
    else:
        # dry-run 模式下只检查 image
        if not Path(args.image).is_file():
            errors.append(f"Input image not found: {args.image}")

    if errors:
        for e in errors:
            print(f"[Bridge] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dry-run 模式 ──
    if args.dry_run:
        run_dry(args)
        print("[Bridge] DRY-RUN completed successfully.")
        return

    # ── 真实推理模式 ──
    print(f"[Bridge] SAM-Road project: {args.samroad_project}")
    print(f"[Bridge] Input image: {args.image}")
    print(f"[Bridge] SAM backbone checkpoint: {args.sam_backbone_checkpoint}")
    print(f"[Bridge] SAM-Road model checkpoint: {args.checkpoint}")
    print(f"[Bridge] Config: {args.config}")
    print(f"[Bridge] Output: {output_dir}")
    print(f"[Bridge] Device: {args.device}")

    # ★ Step 1: 设置 matplotlib home 目录等运行时环境变量
    prepare_runtime_env(output_dir=str(output_dir))
    print(f"[Bridge] Runtime env prepared: MPLBACKEND=Agg, MPLCONFIGDIR/HOME/USERPROFILE set")

    # ★ Step 2: 将 SAM-Road 项目目录和 sam 子目录加入 sys.path + PYTHONPATH
    #   必须在 import model (→ model.py → segment_anything) 之前执行
    path_errors = prepare_project_import_paths(str(args.samroad_project))
    if path_errors:
        for e in path_errors:
            print(f"[Bridge] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[Bridge] Import paths prepared: project_dir + sam/ added to sys.path")

    results = run_inference(args)
    save_outputs(results, output_dir, args)
    print("[Bridge] Done.")


if __name__ == "__main__":
    main()
