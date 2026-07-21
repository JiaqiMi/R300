"""Recursive SAM-Road output discovery and standard-name recovery."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path


SCAN_SUFFIXES = {".png", ".jpg", ".jpeg", ".json", ".p", ".pkl", ".npy"}
MASK_NAMES = (
    "road_mask.png", "mask.png", "pred_mask.png", "road_pred.png",
    "road_prediction.png", "seg.png", "segmentation.png", "binary_mask.png",
    "output_mask.png",
)
VIZ_NAMES = (
    "viz.png", "vis.png", "result.png", "overlay.png", "topo_vis.png", "preview.png",
)
GRAPH_NAMES = ("graph.p", "graph.pkl", "graph.json", "pred_graph.json", "topology.json")
PROJECT_OUTPUT_DIRS = ("save", "runs", "outputs", "output", "results")


def _recursive_files(root: Path, *, since: float | None = None):
    if not root.is_dir():
        return []
    result = []
    try:
        iterator = root.rglob("*")
        for path in iterator:
            try:
                if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                    continue
                if since is not None and path.stat().st_mtime < since:
                    continue
                result.append(path)
            except OSError:
                continue
    except OSError:
        pass
    return result


def _is_mask_candidate(path: Path) -> bool:
    name = path.name.lower()
    if name in MASK_NAMES:
        return True
    excluded = ("itsc", "keypoint", "junction", "intersection", "valid", "ignore", "skeleton", "overlay")
    return "mask" in name and not any(token in name for token in excluded)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _ordered(candidates, aliases):
    priority = {name: index for index, name in enumerate(aliases)}
    return sorted(
        set(candidates),
        key=lambda path: (
            priority.get(path.name.lower(), len(aliases)),
            -path.stat().st_mtime if path.exists() else 0,
            len(path.parts), str(path).lower(),
        ),
    )


def _copy_image_as_png(source: Path, target: Path):
    if source.suffix.lower() == ".png":
        shutil.copy2(source, target)
        return
    import cv2
    import numpy as np
    if source.suffix.lower() == ".npy":
        image = np.asarray(np.load(str(source), allow_pickle=False)).squeeze()
        if image.ndim != 2:
            raise ValueError(f"mask npy 必须是二维数组，实际 shape={image.shape}")
        if image.dtype != np.uint8:
            finite = np.nan_to_num(image.astype(np.float32), copy=False)
            if float(finite.max(initial=0.0)) <= 1.0:
                finite *= 255.0
            image = np.clip(finite, 0, 255).astype(np.uint8)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError(f"无法读取候选 mask: {source}")
    if not cv2.imwrite(str(target), image):
        raise OSError(f"无法写入标准 road_mask.png: {target}")


def _copy_standard(source: Path, target: Path, kind: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    if kind in ("mask", "viz"):
        _copy_image_as_png(source, target)
    else:
        shutil.copy2(source, target)


def _read_existing_metadata(path: Path):
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def diagnose_and_standardize_samroad_outputs(
    output_dir,
    *,
    project_dir=None,
    started_at: float | None = None,
    ignore_graph: bool = False,
):
    """Scan outputs, recover aliases, write metadata, and return a report.

    External project output folders are considered only for files modified
    during the current run (15-second tolerance), preventing stale masks from
    older inference runs from being silently imported.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    local_files = _recursive_files(output)
    external_files = []
    searched_dirs = []
    project = Path(project_dir).resolve() if project_dir else None
    if project is not None and project.is_dir():
        since = (float(started_at) - 15.0) if started_at else time.time() - 3600.0
        for name in PROJECT_OUTPUT_DIRS:
            candidate_dir = project / name
            searched_dirs.append(str(candidate_dir))
            if candidate_dir.is_dir():
                external_files.extend(_recursive_files(candidate_dir, since=since))

    all_candidates = list(local_files)
    mapping = {}
    warnings = []

    local_masks = _ordered([path for path in local_files if _is_mask_candidate(path) and _nonempty(path)], MASK_NAMES)
    external_masks = _ordered([path for path in external_files if _is_mask_candidate(path) and _nonempty(path)], MASK_NAMES)
    mask_candidates = local_masks + [path for path in external_masks if path not in local_masks]
    target_mask = output / "road_mask.png"
    if not _nonempty(target_mask) and mask_candidates:
        source = next((path for path in mask_candidates if path.resolve() != target_mask.resolve()), None)
        if source is None:
            warnings.append("road_mask.png 为空，且没有其他可用候选 mask")
        else:
            try:
                _copy_standard(source, target_mask, "mask")
                mapping["road_mask"] = {"source": str(source), "standard": str(target_mask)}
            except Exception as exc:
                warnings.append(f"候选 mask 标准化失败: {source}: {exc}")

    local_viz = _ordered([path for path in local_files if path.name.lower() in VIZ_NAMES], VIZ_NAMES)
    external_viz = _ordered([path for path in external_files if path.name.lower() in VIZ_NAMES], VIZ_NAMES)
    viz_candidates = local_viz + [path for path in external_viz if path not in local_viz]
    target_viz = output / "viz.png"
    if not target_viz.is_file() and viz_candidates:
        source = viz_candidates[0]
        try:
            _copy_standard(source, target_viz, "viz")
            mapping["viz"] = {"source": str(source), "standard": str(target_viz)}
        except Exception as exc:
            warnings.append(f"候选 viz 标准化失败: {source}: {exc}")

    local_graphs = _ordered([path for path in local_files if path.name.lower() in GRAPH_NAMES], GRAPH_NAMES)
    external_graphs = _ordered([path for path in external_files if path.name.lower() in GRAPH_NAMES], GRAPH_NAMES)
    graph_candidates = local_graphs + [path for path in external_graphs if path not in local_graphs]
    if not ignore_graph and not (output / "graph.p").is_file() and not (output / "graph.json").is_file() and graph_candidates:
        source = graph_candidates[0]
        target_graph = output / ("graph.p" if source.suffix.lower() in (".p", ".pkl") else "graph.json")
        try:
            _copy_standard(source, target_graph, "graph")
            mapping["graph"] = {"source": str(source), "standard": str(target_graph)}
        except Exception as exc:
            warnings.append(f"候选 graph 标准化失败: {source}: {exc}")

    # Re-scan after copies, and preserve any runner-specific metadata fields.
    final_files = _recursive_files(output)
    all_output_files = []
    for path in output.rglob("*"):
        if path.is_file():
            try:
                all_output_files.append(str(path.relative_to(output)).replace("\\", "/"))
            except ValueError:
                all_output_files.append(str(path))
    report = _read_existing_metadata(output / "metadata.json")
    report.update({
        "output_dir": str(output),
        "files_found": sorted(str(path.relative_to(output)).replace("\\", "/") for path in final_files),
        "all_files_found": sorted(all_output_files),
        "road_mask_exists": _nonempty(target_mask),
        "candidate_mask_files": [str(path) for path in mask_candidates],
        "candidate_viz_files": [str(path) for path in viz_candidates],
        "candidate_graph_files": [str(path) for path in graph_candidates],
        "project_output_dirs_scanned": searched_dirs,
        "external_recent_files": sorted(str(path) for path in set(external_files)),
        "output_mapping": {**report.get("output_mapping", {}), **mapping},
        "mask_source": mapping.get("road_mask", {}).get("source", str(target_mask) if target_mask.is_file() else ""),
        "warnings": list(report.get("warnings", [])) + warnings,
    })
    if "metadata.json" not in report["all_files_found"]:
        report["all_files_found"].append("metadata.json")
        report["all_files_found"].sort()
    if "metadata.json" not in report["files_found"]:
        report["files_found"].append("metadata.json")
        report["files_found"].sort()
    report["has_road_mask"] = report["road_mask_exists"]
    report["has_viz"] = (output / "viz.png").is_file()
    report["has_graph"] = (output / "graph.p").is_file() or (output / "graph.json").is_file()
    report["output_validation_success"] = bool(report["road_mask_exists"])
    if "success" in report:
        report["success"] = bool(report["success"] and report["road_mask_exists"])
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report
