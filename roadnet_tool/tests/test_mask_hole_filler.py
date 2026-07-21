"""Tests for road ribbon guided hole & gap fill."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from roadnet.mask_hole_filler import (
    fill_holes_and_gaps_guided_by_ribbon,
    save_ribbon_hole_gap_artifacts,
)


def _road_strip(h=120, w=200, y0=50, y1=70):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, 10:w - 10] = 255
    ribbon = np.zeros((h, w), dtype=np.uint8)
    ribbon[y0 - 5:y1 + 5, 5:w - 5] = 255
    return mask, ribbon


def test_fills_internal_hole_inside_ribbon():
    mask, ribbon = _road_strip()
    # internal hole
    mask[55:62, 80:88] = 0
    repaired, report = fill_holes_and_gaps_guided_by_ribbon(mask, ribbon)
    assert report["filled_hole_count"] >= 1
    assert repaired[58, 84] == 255
    # outside ribbon unchanged zeros stay zero
    assert repaired[10, 10] == 0


def test_fills_ribbon_gap_not_closed_hole():
    mask, ribbon = _road_strip()
    # open gap connected to background but inside ribbon
    mask[50:70, 90:100] = 0
    # leave road on both sides so surround_ratio is decent
    repaired, report = fill_holes_and_gaps_guided_by_ribbon(
        mask, ribbon,
        config={
            "max_gap_area_px": 800,
            "max_gap_diameter_px": 40,
            "min_surround_ratio_for_gap": 0.30,
            "max_gap_distance_to_mask_px": 10,
        },
    )
    assert report["candidate_gap_count"] >= 1
    assert report["filled_gap_count"] >= 1
    assert repaired[60, 95] == 255


def test_rejects_outside_ribbon():
    mask, ribbon = _road_strip()
    # hole far outside ribbon
    mask[10:18, 10:18] = 255
    mask[12:16, 12:16] = 0
    repaired, report = fill_holes_and_gaps_guided_by_ribbon(mask, ribbon)
    # the outer ring is not ribbon-constrained road structure; hole should not fill
    assert repaired[14, 14] == 0


def test_respects_ignore_and_valid():
    mask, ribbon = _road_strip()
    mask[55:62, 80:88] = 0
    ignore = np.zeros_like(mask)
    ignore[55:62, 80:88] = 255
    repaired, report = fill_holes_and_gaps_guided_by_ribbon(
        mask, ribbon, ignore_mask=ignore,
    )
    assert repaired[58, 84] == 0
    assert any(r.get("reason") == "inside_ignore" for r in report["rejected_holes"])

    mask2, ribbon2 = _road_strip()
    mask2[55:62, 80:88] = 0
    valid = np.full_like(mask2, 255)
    valid[55:62, 80:88] = 0
    repaired2, report2 = fill_holes_and_gaps_guided_by_ribbon(
        mask2, ribbon2, valid_area_mask=valid,
    )
    assert repaired2[58, 84] == 0
    assert any(r.get("reason") == "outside_valid_area" for r in report2["rejected_holes"])


def test_no_fill_outside_ribbon_on_gap():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 10:40] = 255
    ribbon = np.zeros((100, 100), dtype=np.uint8)
    ribbon[40:60, 10:50] = 255
    repaired, _report = fill_holes_and_gaps_guided_by_ribbon(mask, ribbon)
    assert repaired[20, 80] == 0
    added = (repaired > 0) & (mask == 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    allowed = cv2.dilate(ribbon, k)
    assert np.count_nonzero(added & (allowed == 0)) == 0


def test_artifacts_and_report(tmp_path: Path):
    mask, ribbon = _road_strip()
    mask[55:62, 80:88] = 0
    mask[50:70, 110:118] = 0
    repaired, report = fill_holes_and_gaps_guided_by_ribbon(mask, ribbon)
    paths = save_ribbon_hole_gap_artifacts(
        mask, repaired, report, tmp_path,
        preview_size=(100, 60),
        input_mask_path="in.png",
    )
    required = [
        "mask_before_ribbon_fill.png",
        "road_ribbon_mask.png",
        "hole_candidates.png",
        "gap_candidates.png",
        "accepted_holes_overlay.png",
        "accepted_gaps_overlay.png",
        "rejected_candidates_overlay.png",
        "ribbon_hole_gap_filled_mask.png",
        "ribbon_hole_gap_filled_mask_preview.png",
        "ribbon_hole_gap_fill_report.json",
    ]
    for name in required:
        assert name in paths
        assert Path(paths[name]).is_file()
    data = json.loads(Path(paths["ribbon_hole_gap_fill_report.json"]).read_text(encoding="utf-8"))
    assert "filled_hole_count" in data
    assert "filled_gap_count" in data
    assert "accepted_holes" in data
    assert "rejected_gaps" in data
