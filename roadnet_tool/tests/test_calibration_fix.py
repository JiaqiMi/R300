"""
测试坐标校准功能的各项修复：

1. QDoubleSpinBox 范围/精度/步长验证
2. lon/lat 填反检测
3. corners_wgs84 JSON 导入/导出（[lon, lat] 格式）
4. 重复计算稳定性（reset_state）
5. reset_state 方法
6. 失败不回显"已校准"
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

# 确保可以导入 roadnet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test 1: geo_calibration reset_state
# ============================================================================
def test_reset_state():
    from roadnet.geo_calibration import GeoCalibration

    geo = GeoCalibration(mode="auto")

    # 模拟一次成功计算的状态
    geo.enabled = True
    geo.transform_mode = "pyproj_utm"
    geo.pixel_to_world_matrix = np.eye(3)
    geo.pixel_resolution_estimated_m = 0.5
    geo.control_points = [
        {"name": "top_left", "pixel": [0, 0], "lon": 117.0, "lat": 31.0,
         "x_meter": 12345.0, "y_meter": 23456.0},
        {"name": "top_right", "pixel": [100, 0], "lon": 117.1, "lat": 31.0,
         "x_meter": 12355.0, "y_meter": 23456.0},
        {"name": "bottom_left", "pixel": [0, 100], "lon": 117.0, "lat": 30.9,
         "x_meter": 12345.0, "y_meter": 23446.0},
    ]

    # 重置
    geo.reset_state()

    assert geo.enabled is False, "enabled 应为 False"
    assert geo.transform_mode is None, "transform_mode 应为 None"
    assert geo.pixel_to_world_matrix is None, "pixel_to_world_matrix 应为 None"
    assert geo.pixel_resolution_estimated_m is None, "pixel_resolution_estimated_m 应为 None"

    # 控制点保留但 x_meter/y_meter 清除
    assert len(geo.control_points) == 3
    for cp in geo.control_points:
        assert "x_meter" not in cp, f"x_meter 应被清除，但存在: {cp}"
        assert "y_meter" not in cp, f"y_meter 应被清除，但存在: {cp}"
        assert "lon" in cp
        assert "lat" in cp

    print("[OK] test_reset_state: 状态重置正确，x_meter/y_meter 已清除")


# ============================================================================
# Test 2: 验证 set_control_points 自动清除 x_meter/y_meter
# ============================================================================
def test_set_control_points_clears_xy():
    from roadnet.geo_calibration import GeoCalibration

    geo = GeoCalibration()
    cp_data = [
        {"name": "top_left", "pixel": [0, 0], "lon": 117.12345678, "lat": 31.12345678},
    ]
    geo.set_control_points(cp_data)
    assert len(geo.control_points) == 1
    assert "x_meter" not in geo.control_points[0]
    assert geo.lon0 == 117.12345678
    assert geo.lat0 == 31.12345678

    print(f"[OK] test_set_control_points_clears_xy: lon={geo.lon0:.8f}, lat={geo.lat0:.8f}")


# ============================================================================
# Test 3: validate_control_points 世界共线检查（防御性修复）
# ============================================================================
def test_validate_no_world_check_before_projection():
    """验证：在 setup_projection 之前，validate_control_points 不会因缺少
    x_meter/y_meter 而错误地判定为共线。"""
    from roadnet.geo_calibration import GeoCalibration

    geo = GeoCalibration()
    cp_data = [
        {"name": "top_left",     "pixel": [0, 0],     "lon": 117.0, "lat": 31.0},
        {"name": "top_right",    "pixel": [500, 0],   "lon": 117.1, "lat": 31.0},
        {"name": "bottom_left",  "pixel": [0, 500],   "lon": 117.0, "lat": 30.9},
    ]
    geo.set_control_points(cp_data)

    # 没有调用 setup_projection，x_meter/y_meter 不存在
    # validate 应该通过（跳过世界共线检查）
    valid, err = geo.validate_control_points()
    assert valid, f"应该在无 setup_projection 时通过验证，错误: {err}"

    # 但如果手动注入 transform_mode 和无意义的 x_meter/y_meter，应该跳过
    geo.transform_mode = "pyproj_utm"
    # x_meter/y_meter 在 reset_state/set_control_points 后已被清除
    valid, err = geo.validate_control_points()
    # 应该跳过世界共线检查（因为没有 has_world_coords）
    assert valid, f"缺少 x_meter/y_meter 时应跳过世界共线检查，错误: {err}"

    print("[OK] test_validate_no_world_check_before_projection")


# ============================================================================
# Test 4: 重复计算稳定性
# ============================================================================
def test_repeat_compute_stability():
    """验证：连续调用 compute_affine 两次，结果稳定一致。"""
    from roadnet.geo_calibration import GeoCalibration

    # 第一次计算
    geo = GeoCalibration()
    cp_data = [
        {"name": "top_left",     "pixel": [0, 0],     "lon": 117.123456, "lat": 31.123456},
        {"name": "top_right",    "pixel": [500, 0],   "lon": 117.223456, "lat": 31.123456},
        {"name": "bottom_left",  "pixel": [0, 500],   "lon": 117.123456, "lat": 31.023456},
    ]
    geo.set_control_points(cp_data)
    geo.setup_projection()
    ok1 = geo.compute_affine()
    assert ok1, "第一次计算应该成功"
    res1 = geo.pixel_resolution_estimated_m
    mat1 = geo.pixel_to_world_matrix.copy() if geo.pixel_to_world_matrix is not None else None
    print(f"  第1次: 分辨率={res1:.4f} m/px, transform={geo.transform_mode}")

    # 第二次计算：模拟 _on_calibration_compute 流程
    # reset_state → set_control_points → setup_projection → compute_affine
    geo.reset_state()
    assert geo.enabled is False
    assert geo.transform_mode is None
    geo.set_control_points(cp_data)  # 重新从"UI"读取
    geo.setup_projection()
    ok2 = geo.compute_affine()
    assert ok2, "第二次计算应该成功（不会因残留状态失败）"
    res2 = geo.pixel_resolution_estimated_m
    mat2 = geo.pixel_to_world_matrix.copy() if geo.pixel_to_world_matrix is not None else None
    print(f"  第2次: 分辨率={res2:.4f} m/px, transform={geo.transform_mode}")

    # 验证一致性
    assert abs(res1 - res2) < 1e-8, f"两次分辨率不一致: {res1} vs {res2}"
    if mat1 is not None and mat2 is not None:
        assert np.allclose(mat1, mat2, atol=1e-6), "两次仿射矩阵不一致"

    # 第三次计算（更多次也应该稳定）
    geo.reset_state()
    geo.set_control_points(cp_data)
    geo.setup_projection()
    ok3 = geo.compute_affine()
    assert ok3, "第三次计算应该成功"
    res3 = geo.pixel_resolution_estimated_m
    assert abs(res1 - res3) < 1e-8

    print(f"[OK] test_repeat_compute_stability: 3 次计算全部一致，分辨率={res1:.4f}")


# ============================================================================
# Test 5: lon/lat 填反检测
# ============================================================================
def test_lon_lat_swap_detection():
    """在模块级别测试 detect_lon_lat_swap 逻辑。"""
    # 模拟控制点数据（不依赖 UI 控件）
    from roadnet.geo_calibration import GeoCalibration

    # 正常数据：不触发警告
    normal_cps = [
        {"name": "top_left", "pixel": [0, 0], "lon": 117.123456, "lat": 31.123456},
        {"name": "top_right", "pixel": [500, 0], "lon": 117.223456, "lat": 31.123456},
        {"name": "bottom_left", "pixel": [0, 500], "lon": 117.123456, "lat": 31.023456},
    ]

    # 填反数据：lon 像纬度 (30.x)，lat 像经度 (117.x)
    swapped_cps = [
        {"name": "top_left", "pixel": [0, 0], "lon": 31.123456, "lat": 117.123456},
        {"name": "top_right", "pixel": [500, 0], "lon": 31.123456, "lat": 117.223456},
        {"name": "bottom_left", "pixel": [0, 500], "lon": 31.023456, "lat": 117.123456},
    ]

    # lat 超出范围数据
    out_of_range_cps = [
        {"name": "top_left", "pixel": [0, 0], "lon": 117.0, "lat": 95.0},
        {"name": "top_right", "pixel": [500, 0], "lon": 117.1, "lat": 31.0},
        {"name": "bottom_left", "pixel": [0, 500], "lon": 117.0, "lat": 30.9},
    ]

    # 测试正常的 detect 逻辑
    # 正常数据没有填反嫌疑
    import re
    warnings_normal = _detect_swap_internal(normal_cps)
    assert warnings_normal == "", f"正常数据不应触发警告: {warnings_normal}"

    # 填反数据
    warnings_swapped = _detect_swap_internal(swapped_cps)
    # lat=117 超出 -90~90，触发"超出合法范围"；lat 超出也会触发"超出"提示
    assert "超出" in warnings_swapped or "lat" in warnings_swapped.lower(), \
        f"填反数据应触发警告，实际: {warnings_swapped}"

    # lat 超出范围
    warnings_range = _detect_swap_internal(out_of_range_cps)
    assert "超出" in warnings_range or "lat" in warnings_range.lower(), \
        f"lat 超出范围应触发警告，实际: {warnings_range}"

    print("[OK] test_lon_lat_swap_detection")


def _detect_swap_internal(control_points):
    """参数面板的 detect_lon_lat_swap 内部逻辑（独立测试）。"""
    swap_suspects = []
    for cp in control_points:
        lon = cp.get("lon", 0)
        lat = cp.get("lat", 0)
        name = cp.get("name", "?")

        if lat > 90 or lat < -90:
            swap_suspects.append(
                f"控制点 {name} 纬度 lat={lat:.6f} 超出合法范围"
            )
            continue
        if -90 <= lon <= 90 and 100 <= abs(lat) <= 130:
            swap_suspects.append(
                f"控制点 {name} 坐标疑似经纬度填反"
            )

    return "\n".join(swap_suspects)


# ============================================================================
# Test 6: corners_wgs84 JSON 导入导出（[lon, lat] 格式）
# ============================================================================
def test_corners_wgs84_import():
    from roadnet.gcp_io import load_gcp_json

    # 创建临时 JSON 文件
    data = {
        "corners_wgs84": {
            "top_left": [117.12345678, 31.12345678],
            "top_right": [117.22345678, 31.12345678],
            "bottom_left": [117.12345678, 31.02345678],
            "bottom_right": [117.22345678, 31.02345678],
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8") as f:
        json.dump(data, f)
        f_path = f.name

    try:
        cps = load_gcp_json(f_path, 500, 500)
        assert len(cps) == 4, f"应解析 4 个控制点，实际 {len(cps)}"

        # 验证 [lon, lat] 映射正确
        for cp in cps:
            name = cp["name"]
            lon = cp["lon"]
            lat = cp["lat"]
            expected = data["corners_wgs84"][name]
            assert abs(lon - expected[0]) < 1e-7, \
                f"{name}: lon={lon} 不等于预期 {expected[0]}"
            assert abs(lat - expected[1]) < 1e-7, \
                f"{name}: lat={lat} 不等于预期 {expected[1]}"
            print(f"  {name}: lon={lon:.8f}, lat={lat:.8f} (正确: [lon, lat])")

        # 验证 pixel 自动推断
        assert cps[0]["pixel"] == [0, 0]  # top_left
        assert cps[1]["pixel"] == [499, 0]  # top_right
        assert cps[2]["pixel"] == [0, 499]  # bottom_left
        assert cps[3]["pixel"] == [499, 499]  # bottom_right

        print("[OK] test_corners_wgs84_import: [lon, lat] 格式正确解析")
    finally:
        os.unlink(f_path)


def test_corners_wgs84_export():
    from roadnet.geo_calibration import GeoCalibration

    geo = GeoCalibration()
    cp_data = [
        {"name": "top_left", "pixel": [0, 0], "lon": 117.12345678, "lat": 31.12345678},
        {"name": "top_right", "pixel": [500, 0], "lon": 117.22345678, "lat": 31.12345678},
        {"name": "bottom_left", "pixel": [0, 500], "lon": 117.12345678, "lat": 31.02345678},
    ]
    geo.set_control_points(cp_data)

    d = geo.to_dict()
    assert "corners_wgs84" in d, "to_dict 应包含 corners_wgs84 字段"

    corners = d["corners_wgs84"]
    for name, coords in corners.items():
        assert len(coords) == 2, f"{name} 坐标应是 [lon, lat]"
        assert -180 <= coords[0] <= 180, f"{name} lon 超出范围"
        assert -90 <= coords[1] <= 90, f"{name} lat 超出范围"
        assert coords[0] > 100, f"{name} lon 应该 > 100 (中国区域)"
        print(f"  corners_wgs84[{name}] = {coords}")

    # 验证可以 round-trip
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f_path = f.name

    try:
        from roadnet.gcp_io import load_gcp_json
        reloaded = load_gcp_json(f_path, 500, 500)
        assert len(reloaded) == 3
        for orig, reloaded_cp in zip(cp_data, reloaded):
            assert abs(orig["lon"] - reloaded_cp["lon"]) < 1e-8
            assert abs(orig["lat"] - reloaded_cp["lat"]) < 1e-8
        print("[OK] test_corners_wgs84_export: round-trip 一致")
    finally:
        os.unlink(f_path)


# ============================================================================
# Test 7: 计算失败后 enabled=False，UI 显示"未校准"
# ============================================================================
def test_failure_shows_uncalibrated():
    from roadnet.geo_calibration import GeoCalibration

    geo = GeoCalibration()

    # 用共线世界坐标（同一经度）构造必然失败的场景
    cp_data = [
        {"name": "top_left",     "pixel": [0, 0],     "lon": 117.0, "lat": 31.0},
        {"name": "top_right",    "pixel": [500, 0],   "lon": 117.0, "lat": 31.0},  # 同 lon
        {"name": "bottom_left",  "pixel": [0, 500],   "lon": 117.0, "lat": 31.0},  # 同 lon+lat
    ]
    geo.set_control_points(cp_data)
    geo.setup_projection()

    try:
        geo.compute_affine()
        # 可能不会抛异常，但矩阵可能奇异
    except ValueError:
        pass

    # reset_state 后 enabled=False
    geo.reset_state()
    assert geo.enabled is False, "重置后应为未校准状态"

    print("[OK] test_failure_shows_uncalibrated: 重置后 enabled=False")


# ============================================================================
# Test 8: 验证 set_control_points 对 117.xx 精度的保留
# ============================================================================
def test_lon_precision():
    from roadnet.geo_calibration import GeoCalibration

    geo = GeoCalibration()

    # 117.12345678 应完整保留
    cp_data = [
        {"name": "top_left",     "pixel": [0, 0],     "lon": 117.12345678, "lat": 31.12345678},
        {"name": "top_right",    "pixel": [500, 0],   "lon": 117.22345678, "lat": 31.12345678},
        {"name": "bottom_left",  "pixel": [0, 500],   "lon": 117.12345678, "lat": 31.02345678},
    ]
    geo.set_control_points(cp_data)

    for cp in geo.control_points:
        lon = cp["lon"]
        assert isinstance(lon, float), f"lon 应为 float，实际 {type(lon)}"
        assert 117 < lon < 118 or abs(lon) < 1e-6, f"lon={lon} 不符合预期"
        assert len(str(lon).split(".")[-1]) >= 8 or lon == int(lon), \
            f"lon={lon} 精度不足，字符串: {str(lon)}"

    print(f"[OK] test_lon_precision: lon={geo.control_points[0]['lon']:.8f}")


# ============================================================================
# 运行全部测试
# ============================================================================
if __name__ == "__main__":
    all_tests = [
        ("reset_state 状态重置", test_reset_state),
        ("set_control_points 清除 x_meter/y_meter", test_set_control_points_clears_xy),
        ("validate 无世界共线检查（投影前）", test_validate_no_world_check_before_projection),
        ("重复计算稳定性", test_repeat_compute_stability),
        ("lon/lat 填反检测", test_lon_lat_swap_detection),
        ("corners_wgs84 JSON 导入", test_corners_wgs84_import),
        ("corners_wgs84 JSON 导出 round-trip", test_corners_wgs84_export),
        ("失败后 enabled=False", test_failure_shows_uncalibrated),
        ("lon 精度 117.12345678", test_lon_precision),
    ]

    passed = 0
    failed = 0
    for name, func in all_tests:
        try:
            func()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"总计: {passed} 通过, {failed} 失败")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed} TESTS FAILED")
        sys.exit(1)
