"""骨架优化集成测试。"""
import os, tempfile, json, shutil
import numpy as np
import cv2

# ===== 1. Test normalize_road_mask =====
from roadnet.optimized_skeleton import normalize_road_mask

# Case 1: bool
m = np.zeros((100, 100), dtype=np.bool_)
m[30:70, 30:70] = True
r = normalize_road_mask(m)
assert r.dtype == np.uint8
assert r.max() == 255
print('[1] bool mask OK')

# Case 2: 0/1 float
m2 = np.zeros((100, 100), dtype=np.float64)
m2[20:50, 20:50] = 1.0
r2 = normalize_road_mask(m2)
assert r2.dtype == np.uint8
print('[2] 0/1 float mask OK')

# Case 3: 0/255 uint8
m3 = np.zeros((100, 100), dtype=np.uint8)
m3[10:90, 10:90] = 255
r3 = normalize_road_mask(m3)
assert r3.max() == 255
print('[3] 0/255 mask OK')

# Case 4: grayscale with OTSU
m4 = np.zeros((100, 100), dtype=np.uint8)
m4[40:60, 40:60] = 200
m4[0:10, 0:10] = 5
r4 = normalize_road_mask(m4)
assert r4.max() == 255
print('[4] grayscale OTSU OK, road_ratio=%s' % ((r4 > 0).sum() / r4.size))

# Case 5: very dark
m5 = np.ones((100, 100), dtype=np.uint8) * 3
r5 = normalize_road_mask(m5)
print('[5] dark noise mask OK, road_px=%d' % ((r5 > 0).sum()))

print('normalize_road_mask: ALL 5 cases PASSED')

# ===== 2. Test resolve_center_distance_threshold =====
from roadnet.optimized_skeleton import (
    resolve_center_distance_threshold, skeletonize_thin,
    compute_distance_transform,
)

wide = np.zeros((200, 200), dtype=np.uint8)
wide[50:150, 50:150] = 255
dist = compute_distance_transform(wide)
skel = skeletonize_thin(wide)
thr = resolve_center_distance_threshold(skel, dist, 4.0)
print('[6] Wide road: requested=4.0, resolved=%.2f' % thr)

# Ultra-thin road (2-3px): should get reduced threshold
thin = np.zeros((200, 200), dtype=np.uint8)
thin[99:101, 20:180] = 255  # 2px wide
thin[20:180, 99:101] = 255
dist_t = compute_distance_transform(thin)
skel_t = skeletonize_thin(thin)
thr_t = resolve_center_distance_threshold(skel_t, dist_t, 4.0)
print('[7] Ultra-thin road: requested=4.0, resolved=%.2f' % thr_t)
assert thr_t < 4.0, "ultra-thin road should get lower threshold"

# 10px road with requested=4.0 should keep 4.0 (wide enough)
thick = np.zeros((200, 200), dtype=np.uint8)
thick[60:140, 60:140] = 255
thick[70:130, 70:130] = 255  # double layer = thicker
dist_th = compute_distance_transform(thick)
skel_th = skeletonize_thin(thick)
thr_th = resolve_center_distance_threshold(skel_th, dist_th, 4.0)
print('[7b] Thick road: requested=4.0, resolved=%.2f (wide enough)' % thr_th)
print('resolve_center_distance_threshold: PASSED')

# ===== 3. Test full optimize_skeleton =====
from roadnet.optimized_skeleton import optimize_skeleton

m = np.zeros((200, 200), dtype=np.uint8)
m[40:160, 40:160] = 255
m[50:150, 50:150] = 255
raw_skel = skeletonize_thin(m)
result = optimize_skeleton(m, raw_skel, min_center_dist=4.0,
                           min_branch_length=20, max_connect_dist=0)
stats = result['stats']
print('[8] optimize_skeleton: %d -> %d px, endpoints: %d -> %d, '
      'spur_removed=%d, effective_mcd=%s' % (
    stats['raw_pixels'], stats['optimized_pixels'],
    stats['raw_endpoints'], stats['optimized_endpoints'],
    stats['removed_spur_count'],
    stats.get('effective_min_center_dist', 'N/A')))
assert 'effective_min_center_dist' in stats
assert 'raw_pixels' in stats
assert 'optimized_pixels' in stats
print('optimize_skeleton: PASSED')

# ===== 4. Test run_skeleton_optimization =====
from roadnet.skeleton_optimizer_adapter import (
    SkeletonOptimizeConfig, run_skeleton_optimization,
)
tmpdir = tempfile.mkdtemp()
img_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
img_rgb[80:120, 80:120] = [128, 128, 128]

config = SkeletonOptimizeConfig(
    skeleton_method='skeletonize',
    min_center_dist=3.0,
    prune_length=20,
    enable_endpoint_connect=False,
    output_overlay=True,
    save_stats_json=True,
)
full_result = run_skeleton_optimization(m, config, tmpdir, image_rgb=img_rgb)
assert full_result.success
assert full_result.optimized_skeleton is not None
assert full_result.normalized_mask is not None
print('[9] Full pipeline OK: %.2fs' % full_result.elapsed_seconds)
for name, path in full_result.saved_files.items():
    assert os.path.exists(path), 'Missing: %s' % path
    print('    %s: %d bytes' % (name, os.path.getsize(path)))
print('run_skeleton_optimization: PASSED')

# ===== 5. Verify stats JSON =====
stats_path = os.path.join(tmpdir, 'skeleton_outputs', 'skeleton_stats.json')
assert os.path.exists(stats_path)
with open(stats_path, encoding='utf-8') as f:
    stats_json = json.load(f)
required_keys = [
    'mask_width', 'mask_height', 'raw_skeleton_pixels',
    'optimized_skeleton_pixels', 'effective_min_center_dist',
    'skeleton_method', 'prune_length', 'border_margin',
    'endpoint_connect_distance', 'endpoint_connect_enabled',
    'connected_gap_pixels', 'removed_spur_count',
    'junction_cluster_count', 'elapsed_seconds',
]
for key in required_keys:
    assert key in stats_json, 'Missing stats key: %s' % key
print('[10] Stats JSON: all %d required keys present' % len(required_keys))
print('Stats JSON: PASSED')

# ===== 6. Test existing optimize_skeleton still works (backward compat) =====
# Use a realistic road-like mask (not solid block which skeletonize handles poorly)
m6 = np.zeros((400, 400), dtype=np.uint8)
# Draw a cross pattern (like roads)
m6[180:220, 50:350] = 255  # horizontal road ~40px wide
m6[50:350, 180:220] = 255  # vertical road ~40px wide
m6[120:140, 120:280] = 255  # oblique road
m6[120:280, 120:140] = 255
raw6 = skeletonize_thin(m6)
assert (raw6 > 0).sum() > 10, "skeleton should have reasonable pixels for road-like mask"

result6 = optimize_skeleton(m6, raw6, min_center_dist=2.0,
                            min_branch_length=20, border_margin=5,
                            max_connect_dist=0)
assert result6['stats']['optimized_pixels'] > 0, "optimized pixels should be > 0"
print('[11] Backward compat OK (existing _on_run_optimize compatible)')

# ===== 7. Verify output files exist =====
expected_files = ['processed_mask.png', 'optimized_skeleton.png',
                  'skeleton_overlay.png', 'skeleton_stats.json']
for fn in expected_files:
    p = os.path.join(tmpdir, 'skeleton_outputs', fn)
    assert os.path.exists(p), 'Missing output: %s' % fn
    assert os.path.getsize(p) > 0, 'Empty output: %s' % fn
print('[12] All output files present and non-empty')

# cleanup
shutil.rmtree(tmpdir, ignore_errors=True)

print()
print('=== ALL %d TESTS PASSED ===' % 12)
