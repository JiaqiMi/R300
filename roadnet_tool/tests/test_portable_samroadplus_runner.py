"""Tests for SAMRoad++ Portable adapter."""

import os

from roadnet.portable_samroadplus_runner import (
    is_portable_project,
    resolve_portable_paths,
    validate_portable_paths,
    build_portable_command,
    build_portable_env,
    normalize_output_mask,
    find_mask_candidates,
    resolve_portable_project_dir,
    apply_portable_config,
    DEFAULT_PORTABLE_PROJECT_DIR,
)


def _make_portable_project(root):
    for name in ("infer.py", "model_portable.py", "config.yaml", "model_state_dict.pth"):
        (root / name).write_text("x", encoding="utf-8")
    (root / "sam").mkdir()
    return root


def test_is_portable_project_true(tmp_path):
    _make_portable_project(tmp_path)
    assert is_portable_project(str(tmp_path)) is True


def test_is_portable_project_false_missing_files(tmp_path):
    (tmp_path / "infer.py").write_text("x", encoding="utf-8")
    assert is_portable_project(str(tmp_path)) is False


def test_resolve_portable_paths(tmp_path):
    paths = resolve_portable_paths(str(tmp_path))
    assert paths["infer_script"].name == "infer.py"
    assert paths["config"].name == "config.yaml"
    assert paths["checkpoint"].name == "model_state_dict.pth"
    assert paths["sam_dir"].name == "sam"


def test_validate_portable_paths_ok(tmp_path):
    _make_portable_project(tmp_path)
    out = tmp_path / "out"
    errors = validate_portable_paths("python", str(tmp_path), str(out))
    # python 可能不是文件路径，忽略 python 相关错误后应无其它错误
    non_python = [e for e in errors if "Python 解释器" not in e]
    assert non_python == []


def test_validate_portable_paths_missing(tmp_path):
    errors = validate_portable_paths("python", str(tmp_path), str(tmp_path / "out"))
    assert any("infer.py" in e for e in errors)
    assert any("sam" in e for e in errors)


def test_resolve_portable_project_dir_default():
    assert str(resolve_portable_project_dir("")) == DEFAULT_PORTABLE_PROJECT_DIR
    assert str(resolve_portable_project_dir(None)) == DEFAULT_PORTABLE_PROJECT_DIR
    assert str(resolve_portable_project_dir("None")) == DEFAULT_PORTABLE_PROJECT_DIR


def test_resolve_portable_project_dir_explicit(tmp_path):
    assert str(resolve_portable_project_dir(str(tmp_path))) == str(tmp_path)


def test_apply_portable_config_overrides(tmp_path):
    _make_portable_project(tmp_path)

    class _Cfg:
        project_dir = None
        infer_script = None
        config_path = None
        samroad_model_ckpt_path = None

    cfg = _Cfg()
    adapter = apply_portable_config(cfg, str(tmp_path))
    assert adapter == "samroadplus_portable"
    assert cfg.infer_script.name == "infer.py"
    assert cfg.config_path.name == "config.yaml"
    assert cfg.samroad_model_ckpt_path.name == "model_state_dict.pth"
    assert str(cfg.project_dir) == str(tmp_path.resolve())


def test_build_portable_command(tmp_path):
    _make_portable_project(tmp_path)
    cmd = build_portable_command(
        "python.exe", str(tmp_path), "img.png", "out_dir", "cuda"
    )
    assert cmd[0] == "python.exe"
    assert cmd[1].endswith("infer.py")
    assert "--image" in cmd and "img.png" in cmd
    assert "--output_dir" in cmd and "out_dir" in cmd
    assert "--config" in cmd
    assert "--checkpoint" in cmd
    assert "--device" in cmd and "cuda" in cmd


def test_build_portable_env_pythonpath(tmp_path):
    _make_portable_project(tmp_path)
    env = build_portable_env({}, str(tmp_path))
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert str(tmp_path.resolve()) in parts
    assert str((tmp_path / "sam").resolve()) in parts


def test_normalize_output_mask_prefers_road_mask(tmp_path):
    (tmp_path / "segmentation.png").write_bytes(b"\x89PNG")
    (tmp_path / "road_mask.png").write_bytes(b"\x89PNG")
    selected, candidates = normalize_output_mask(str(tmp_path))
    assert selected is not None
    assert os.path.basename(selected) == "road_mask.png"
    assert len(candidates) >= 2


def test_normalize_output_mask_copies_alternate(tmp_path):
    (tmp_path / "pred_mask.png").write_bytes(b"\x89PNG")
    selected, candidates = normalize_output_mask(str(tmp_path))
    assert os.path.isfile(os.path.join(str(tmp_path), "road_mask.png"))
    assert selected.endswith("road_mask.png")


def test_normalize_output_mask_none_when_empty(tmp_path):
    selected, candidates = normalize_output_mask(str(tmp_path))
    assert selected is None
    assert candidates == []


def test_find_mask_candidates_recursive(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "mask.png").write_bytes(b"\x89PNG")
    found = find_mask_candidates(str(tmp_path))
    assert len(found) == 1
    assert found[0].endswith("mask.png")
