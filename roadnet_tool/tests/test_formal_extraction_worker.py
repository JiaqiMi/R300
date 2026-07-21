"""Tests for formal extraction worker tile_id helpers."""

from roadnet.formal_extraction_worker import normalize_tile_id


def test_normalize_tile_id_int():
    assert normalize_tile_id(1) == "tile_000001"


def test_normalize_tile_id_digit_string():
    assert normalize_tile_id("1") == "tile_000001"


def test_normalize_tile_id_zero_padded_digits():
    assert normalize_tile_id("000001") == "tile_000001"


def test_normalize_tile_id_prefixed():
    assert normalize_tile_id("tile_000001") == "tile_000001"


def test_normalize_tile_id_none_with_fallback():
    assert normalize_tile_id(None, fallback_index=2) == "tile_000002"


def test_normalize_tile_id_empty_string_with_fallback():
    assert normalize_tile_id("", fallback_index=3) == "tile_000003"
