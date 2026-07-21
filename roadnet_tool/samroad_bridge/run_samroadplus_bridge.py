"""Bridge SAM-RoadPlus Portable inference into RoadNet Studio outputs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROADNET_ROOT = Path(__file__).resolve().parents[1]
if str(ROADNET_ROOT) not in sys.path:
    sys.path.insert(0, str(ROADNET_ROOT))
from roadnet.samroad_output_diagnostics import (  # noqa: E402
    GRAPH_NAMES, MASK_NAMES, VIZ_NAMES, diagnose_and_standardize_samroad_outputs,
)


ITSC_NAMES = ("itsc_mask.png", "keypoint_mask.png", "intersection_mask.png")


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-RoadPlus Portable bridge")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--infer-script", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam-backbone", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--ignore-graph", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def configure_environment(project_dir: Path, output_dir: Path):
    runtime = output_dir / "_runtime"
    home = runtime / "home"
    mpl = runtime / "matplotlib"
    home.mkdir(parents=True, exist_ok=True)
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "MPLCONFIGDIR": str(mpl),
        "MPLBACKEND": "Agg",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
    })
    paths = [str(project_dir)]
    if (project_dir / "sam").is_dir():
        paths.append(str(project_dir / "sam"))
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([old] if old else []))
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def prepare_runtime_config(config_path: Path, backbone_path: Path, output_dir: Path) -> Path:
    data = load_yaml(config_path)
    needs_backbone = not bool(data.get("NO_SAM", False)) and not bool(data.get("SKIP_SAM_CKPT_LOAD", False))
    if needs_backbone and backbone_path != Path():
        data["SAM_CKPT_PATH"] = str(backbone_path)
        runtime_path = output_dir / "samroadplus_runtime_config.yaml"
        import yaml
        with runtime_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
        return runtime_path
    return config_path


def strict_checkpoint_check(infer_script: Path, config_path: Path, checkpoint: Path) -> dict:
    """Construct the real model and perform strict state_dict shape loading."""
    result = {
        "success": False,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "model_class": "",
        "state_dict_key_count": 0,
        "message": "",
    }
    try:
        import torch
        spec = importlib.util.spec_from_file_location("roadnet_samroadplus_infer", str(infer_script))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载推理脚本: {infer_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config_loader = getattr(module, "load_config", None)
        model_class = getattr(module, "SAMRoadplus", None) or getattr(module, "SAMRoadPlus", None)
        if not callable(config_loader) or model_class is None:
            raise RuntimeError(
                "推理入口未暴露 load_config 和 SAMRoadplus 模型类，无法执行 shape 预检"
            )
        model_config = config_loader(str(config_path))
        model = model_class(model_config)
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"checkpoint 不是 state_dict，实际类型: {type(state).__name__}")
        result["state_dict_key_count"] = len(state)
        result["model_class"] = type(model).__name__
        model.load_state_dict(state, strict=True)
        result["success"] = True
        result["message"] = "config/checkpoint 所有 key 与 shape 严格匹配"
        del state, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result["message"] = f"config/checkpoint 不匹配: {type(exc).__name__}: {exc}"
    return result


def find_named_file(root: Path, names) -> Optional[Path]:
    candidates = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in names]
    if not candidates:
        return None
    priorities = {name: index for index, name in enumerate(names)}
    return sorted(candidates, key=lambda path: (priorities.get(path.name.lower(), 999), len(path.parts), str(path)))[0]


def copy_standard_outputs(raw_dir: Path, output_dir: Path, ignore_graph: bool) -> dict:
    mapping = {}
    specs = (
        ("road_mask", MASK_NAMES, "road_mask.png"),
        ("itsc_mask", ITSC_NAMES, "itsc_mask.png"),
        ("viz", VIZ_NAMES, "viz.png"),
    )
    for key, names, target_name in specs:
        source = find_named_file(raw_dir, names)
        if source is not None:
            target = output_dir / target_name
            shutil.copy2(source, target)
            mapping[key] = {"source": str(source), "standard": str(target)}

    graph_source = None if ignore_graph else find_named_file(raw_dir, GRAPH_NAMES)
    if graph_source is not None:
        suffix = graph_source.suffix.lower()
        target_name = "graph.p" if suffix in (".p", ".pkl", ".pickle") else "graph.json"
        target = output_dir / target_name
        shutil.copy2(graph_source, target)
        mapping["graph"] = {"source": str(graph_source), "standard": str(target)}
    return mapping


def write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    project = Path(args.project_dir).resolve()
    infer_script = Path(args.infer_script).resolve()
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    backbone = Path(args.sam_backbone).resolve() if args.sam_backbone else Path()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_environment(project, output_dir)
    os.chdir(project)

    runtime_config = prepare_runtime_config(config_path, backbone, output_dir)
    check = strict_checkpoint_check(infer_script, runtime_config, checkpoint)
    print("SAMROADPLUS_CHECK_JSON=" + json.dumps(check, ensure_ascii=False))
    if args.check_only:
        write_json(output_dir / "samroadplus_preflight.json", check)
        return 0 if check["success"] else 2
    if not check["success"]:
        (output_dir / "samroadplus_stdout.log").write_text("", encoding="utf-8")
        (output_dir / "samroadplus_stderr.log").write_text(check["message"], encoding="utf-8")
        write_json(output_dir / "metadata.json", {
            "model_type": "samroadplus_portable", "success": False,
            "preflight": check, "has_road_mask": False,
            "has_itsc_mask": False, "has_viz": False, "has_graph": False,
        })
        print(check["message"], file=sys.stderr)
        return 2

    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(infer_script),
        "--image", str(Path(args.image).resolve()),
        "--output_dir", str(raw_dir),
        "--config", str(runtime_config),
        "--checkpoint", str(checkpoint),
        "--device", args.device,
    ]
    stdout_path = output_dir / "samroadplus_stdout.log"
    stderr_path = output_dir / "samroadplus_stderr.log"
    print(f"[SamRoadRun] project_dir = {project}")
    print(f"[SamRoadRun] python_exe = {sys.executable}")
    print(f"[SamRoadRun] infer_script = {infer_script}")
    print(f"[SamRoadRun] input_image = {Path(args.image).resolve()}")
    print(f"[SamRoadRun] output_dir = {output_dir}")
    print(f"[SamRoadRun] command = {' '.join(command)}")
    print(f"[SamRoadRun] stdout_path = {stdout_path}")
    print(f"[SamRoadRun] stderr_path = {stderr_path}")
    started = time.time()
    proc = subprocess.run(command, cwd=str(project), env=os.environ.copy(), capture_output=True, text=True)
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    print(f"[SamRoadRun] return_code = {proc.returncode}")
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    mapping = copy_standard_outputs(raw_dir, output_dir, args.ignore_graph)
    mask_path = output_dir / "road_mask.png"
    metadata = {
        "model_type": "samroadplus_portable",
        "success": proc.returncode == 0 and mask_path.is_file(),
        "project_dir": str(project),
        "infer_script": str(infer_script),
        "config_file": str(config_path),
        "runtime_config_file": str(runtime_config),
        "model_checkpoint": str(checkpoint),
        "sam_backbone_checkpoint": str(backbone) if args.sam_backbone else "",
        "checkpoint_config_match": True,
        "preflight": check,
        "return_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "output_mapping": mapping,
        "has_road_mask": mask_path.is_file(),
        "has_itsc_mask": (output_dir / "itsc_mask.png").is_file(),
        "has_viz": (output_dir / "viz.png").is_file(),
        "has_graph": (output_dir / "graph.p").is_file() or (output_dir / "graph.json").is_file(),
        "graph_import_policy": "ignored" if args.ignore_graph else "reference_only",
        "final_graph_overwritten": False,
    }
    try:
        import cv2
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is not None:
            metadata["original_width"] = int(mask.shape[1])
            metadata["original_height"] = int(mask.shape[0])
    except Exception:
        pass
    write_json(output_dir / "metadata.json", metadata)
    diagnostics = diagnose_and_standardize_samroad_outputs(
        output_dir,
        project_dir=project,
        started_at=started,
        ignore_graph=args.ignore_graph,
    )
    metadata.update(diagnostics)
    metadata["success"] = bool(proc.returncode == 0 and diagnostics["road_mask_exists"])
    metadata["has_road_mask"] = diagnostics["road_mask_exists"]
    write_json(output_dir / "metadata.json", metadata)
    if not metadata["success"]:
        if proc.returncode == 0:
            print("推理进程成功，但未识别到 road mask 输出", file=sys.stderr)
            return 3
        return proc.returncode or 1
    print(f"[SAM-RoadPlus Bridge] standardized road_mask: {mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
