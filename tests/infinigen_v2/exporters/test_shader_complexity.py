import bpy
import pytest

from infinigen_v2.exporters.util.render_error_check import (
    SHADER_NODE_COUNT_FAIL,
    ShaderTooComplexError,
    check_scene_shader_complexity,
    count_material_nodes,
)


def _make_material(name, group_size=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_fake_user = True
    if group_size:
        inner = bpy.data.node_groups.new(f"{name}_inner", "ShaderNodeTree")
        for _ in range(group_size):
            inner.nodes.new("ShaderNodeValue")
        grp = mat.node_tree.nodes.new("ShaderNodeGroup")
        grp.node_tree = inner
    return mat


def test_count_material_nodes_includes_group_contents():
    mat = _make_material("with_group", group_size=50)
    assert count_material_nodes(mat) >= 50


def test_check_scene_shader_complexity_passes_under_threshold():
    _make_material("ok_mat", group_size=10)
    counts = check_scene_shader_complexity()
    assert all(c < SHADER_NODE_COUNT_FAIL for c in counts.values())


def test_check_scene_shader_complexity_raises_over_threshold():
    _make_material("too_big", group_size=SHADER_NODE_COUNT_FAIL + 10)
    with pytest.raises(ShaderTooComplexError, match="too_big"):
        check_scene_shader_complexity()
