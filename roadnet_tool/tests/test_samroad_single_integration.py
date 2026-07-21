"""
端到端测试：SAM-Road 单图推理包集成
测试 runner 命令构造、adapter graph.p 转换、输出探测和加载。
"""
import os, json, pickle, tempfile, shutil, numpy as np

# ===== 1. 测试 runner 命令构造 =====
from roadnet.samroad_single_runner import (
    SAMRoadSingleRunConfig, build_command, create_output_dir,
    validate_config, dict_to_runconfig, runconfig_to_dict,
)

config = SAMRoadSingleRunConfig(
    project_dir='D:/sam_road_single_image_share',
    python_executable='D:/Anaconda/envs/samroad/python.exe',
    infer_script='D:/sam_road_single_image_share/infer_single.py',
    config_path='D:/sam_road_single_image_share/config/toponet_vitb_256_spacenet_4060_night.yaml',
    sam_backbone_ckpt_path='D:/sam_road_single_image_share/sam_ckpts/sam_vit_b_01ec64.pth',
    samroad_model_ckpt_path='D:/sam_road_single_image_share/checkpoints/model.ckpt',
    input_image='outputs/test.png',
    device='cuda',
    inference_mode='tile',
    tile_size=1024,
    overlap=128,
    skip_black_tile=True,
    black_threshold=10,
    min_black_component_area=4096,
    valid_pixel_ratio_threshold=0.1,
    merge_method='max',
)
cmd = build_command(config, 'outputs/test_out')
print('[1] Command OK')
assert '--config' in cmd
assert '--checkpoint' in cmd
assert '--image' in cmd
assert '--output_dir' in cmd
print('[1] All expected args present')

# ===== 2. 测试配置持久化 =====
data = runconfig_to_dict(config)
roundtrip = dict_to_runconfig(data)
assert str(roundtrip.project_dir).endswith('sam_road_single_image_share')
assert roundtrip.inference_mode == 'tile'
assert roundtrip.tile_size == 1024 and roundtrip.overlap == 128
assert roundtrip.skip_black_tile is True
assert roundtrip.min_black_component_area == 4096
assert roundtrip.merge_method == 'max'
print('[2] Config round-trip OK')

# ===== 3. 测试 adapter - graph.p 转换 =====
from roadnet.samroad_single_adapter import (
    _convert_graph_p_to_unified, detect_single_outputs, SAMRoadSingleOutput,
    load_single_output,
)

# 构造 Sat2Graph 格式的 mock graph.p
mock_graph = {
    (10, 20): [(15, 25), (12, 18)],
    (15, 25): [(10, 20), (18, 30)],
    (12, 18): [(10, 20)],
    (18, 30): [(15, 25)],
}
nodes, edges = _convert_graph_p_to_unified(mock_graph)
print(f'[3] Converted: {len(nodes)} nodes, {len(edges)} edges')
assert len(nodes) == 4
assert len(edges) == 3
for n in nodes:
    assert 'id' in n and 'x' in n and 'y' in n
    assert n['type'] == 'junction'
for e in edges:
    assert 'id' in e and 'start' in e and 'end' in e and 'points_pixel' in e
    assert len(e['points_pixel']) == 2
    assert e['source'] == 'auto'
    assert e['enabled'] is True
print('[3] Graph conversion format OK')

# ===== 4. 测试 detect_single_outputs =====
tmpdir = tempfile.mkdtemp()
for fname in ['road_mask.png', 'graph.p']:
    p = os.path.join(tmpdir, fname)
    with open(p, 'wb') as f:
        if fname == 'road_mask.png':
            f.write(b'\x89PNG\r\n\x1a\n')
        elif fname == 'graph.p':
            pickle.dump(mock_graph, f)

detected = detect_single_outputs(tmpdir)
print(f'[4] Detected: is_samroad_single={detected.get("is_samroad_single")}, '
      f'has_road_mask={detected.get("has_road_mask")}, '
      f'has_graph={detected.get("has_graph")}, '
      f'has_itsc_mask={detected.get("has_itsc_mask")}, '
      f'has_viz={detected.get("has_viz")}')
assert detected['is_samroad_single'] == True
assert detected['has_road_mask'] == True
assert detected['has_graph'] == True
print('[4] Detection OK')

# ===== 5. 测试 load_single_output =====
output = load_single_output(tmpdir)
print(f'[5] Output: has_road_mask={output.has_road_mask}, '
      f'has_graph={output.has_graph}, nodes={output.node_count}, edges={output.edge_count}')
assert output.has_graph
assert output.node_count == 4
assert output.edge_count == 3
print('[5] Load OK')

# ===== 6. 测试空目录/缺文件容错 =====
empty_dir = tempfile.mkdtemp()
detected2 = detect_single_outputs(empty_dir)
assert detected2['is_samroad_single'] == False
assert detected2['has_graph'] == False
output2 = load_single_output(empty_dir)
assert not output2.has_graph
assert not output2.has_road_mask
print('[6] Empty dir handling OK')

# cleanup
shutil.rmtree(tmpdir, ignore_errors=True)
shutil.rmtree(empty_dir, ignore_errors=True)

print('=== ALL TESTS PASSED ===')
