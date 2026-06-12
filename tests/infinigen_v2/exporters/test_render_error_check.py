import bpy
import numpy as np
import procfunc as pf
import pytest
from procfunc.util.manifest import import_item

from infinigen_v2.exporters.util.render_error_check import (
    ACTIVE_RENDER,
    DEGENERATE,
    MISSING,
    SHADER_NODE_COUNT_FAIL,
    DisplacementCoordError,
    check_material_uv_coords,
    count_material_nodes,
    unsafe_displacement_materials,
)
from infinigen_v2.generators import GENERATORS_MANIFEST
from infinigen_v2.generators.shaders import functionality_lists
from infinigen_v2.generators.shaders.composites import bricks


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


def test_count_material_nodes_under_threshold():
    mat = _make_material("ok_mat", group_size=10)
    assert count_material_nodes(mat) < SHADER_NODE_COUNT_FAIL


def test_count_material_nodes_over_threshold():
    mat = _make_material("too_big", group_size=SHADER_NODE_COUNT_FAIL + 10)
    assert count_material_nodes(mat) >= SHADER_NODE_COUNT_FAIL


def _value_group(name, n_values, child_instances=()):
    """A node group holding `n_values` leaf nodes plus one instance of each
    given child group tree."""
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for _ in range(n_values):
        tree.nodes.new("ShaderNodeValue")
    for child in child_instances:
        grp = tree.nodes.new("ShaderNodeGroup")
        grp.node_tree = child
    return tree


def test_shader_complexity_diamond_dag():
    a, b, c, d = 2, 3, 4, 5
    grp_d = _value_group("diamond_D", d)
    grp_b = _value_group("diamond_B", b, child_instances=[grp_d])
    grp_c = _value_group("diamond_C", c, child_instances=[grp_d])
    mat = bpy.data.materials.new("diamond_mat")
    mat.use_nodes = True
    mat.use_fake_user = True
    base = len(mat.node_tree.nodes)
    for _ in range(a):
        mat.node_tree.nodes.new("ShaderNodeValue")
    for child in (grp_b, grp_c, grp_d):
        grp = mat.node_tree.nodes.new("ShaderNodeGroup")
        grp.node_tree = child
    expected = (base + a) + (b + d) + (c + d) + d
    assert count_material_nodes(mat) == expected


def _composite_material_names():
    mats = pf.util.manifest.filter_manifest(
        GENERATORS_MANIFEST,
        filter={"category": "Material"},
        exclude={"name": ["LATER", "DECLINE"]},
        require_nonempty=["name"],
        min_entries=1,
    )
    return [n for n in mats["name"].values if ".shaders.composites." in n]


def _assert_material_under_node_budget(material_sample, name, seeds=range(11)):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        plane = pf.ops.primitives.mesh_plane(size=1)
        material = material_sample(rng, pf.nodes.shader.coord().object)
        before = {m.name for m in bpy.data.materials}
        pf.ops.object.set_material(plane, material=material)
        new = [m for m in bpy.data.materials if m.name not in before]
        assert new, f"{name} seed {seed} created no material"
        offenders = {
            m.name: c
            for m in new
            if (c := count_material_nodes(m)) >= SHADER_NODE_COUNT_FAIL
        }
        assert not offenders, (
            f"{name} seed {seed} exceeds "
            f"{SHADER_NODE_COUNT_FAIL} flattened nodes: {offenders}"
        )


@pytest.mark.parametrize(
    "pathspec", _composite_material_names(), ids=lambda s: s.rsplit(".", 1)[-1]
)
def test_composite_material_under_node_budget(pathspec):
    _assert_material_under_node_budget(import_item(pathspec), pathspec)


_FUNCTIONALITY_MATERIAL_FUNCS = [
    functionality_lists.table_top_material_distribution,
    functionality_lists.furniture_material_distribution,
    functionality_lists.paint_wall_distribution,
    functionality_lists.paint_flaked_distribution,
    functionality_lists.wall_material_distribution,
    functionality_lists.skirt_material_distribution,
    functionality_lists.floor_material_distribution,
    functionality_lists.ceiling_material_distribution,
]


@pytest.mark.parametrize(
    "material_sample", _FUNCTIONALITY_MATERIAL_FUNCS, ids=lambda f: f.__name__
)
def test_functionality_material_under_node_budget(material_sample):
    _assert_material_under_node_budget(material_sample, material_sample.__name__)


def _bricks_material(vector):
    pf.ops.object.clear_scene()
    obj = pf.ops.primitives.mesh_plane(size=2)
    uvs = pf.ops.attr.uv_coords(obj)
    pf.ops.attr.write_uv_coords(obj, uvs * 2)
    mat = bricks.bricks_distribution(np.random.default_rng(0), vector)
    pf.ops.object.set_material(obj, surface=mat.surface, displacement=mat.displacement)
    return obj.item().active_material


def _raise_if_unsafe(materials):
    """Mirror of the render_cycles displacement gate."""
    unsafe = unsafe_displacement_materials(materials)
    if unsafe:
        raise DisplacementCoordError(str(unsafe))


def test_uv_map_displacement_flagged():
    mat = _bricks_material(pf.nodes.shader.uv_map(uv_map="UVMap"))
    assert unsafe_displacement_materials([mat]) == {mat.name: "ShaderNodeUVMap"}


def test_texcoord_displacement_safe():
    mat = _bricks_material(pf.nodes.shader.coord().uv)
    assert unsafe_displacement_materials([mat]) == {}


def test_gate_raises_on_named_attribute():
    mat = _bricks_material(pf.nodes.shader.uv_map(uv_map="UVMap"))
    with pytest.raises(DisplacementCoordError, match=mat.name):
        _raise_if_unsafe([mat])


def test_attribute_node_displacement_flagged():
    mat = _bricks_material(pf.nodes.shader.attribute("UVMap").vector)
    assert unsafe_displacement_materials([mat]) == {mat.name: "ShaderNodeAttribute"}
    with pytest.raises(DisplacementCoordError, match=mat.name):
        _raise_if_unsafe([mat])


def test_gate_passes_on_safe_coords():
    mat = _bricks_material(pf.nodes.shader.coord().uv)
    _raise_if_unsafe([mat])  # must not raise


def test_uv_map_nested_in_group_displacement_flagged():
    mat = _bricks_material(pf.nodes.shader.uv_map(uv_map="UVMap"))
    groups_with_uv = [
        n
        for n in mat.node_tree.nodes
        if n.bl_idname == "ShaderNodeGroup"
        and n.node_tree
        and any(x.bl_idname == "ShaderNodeUVMap" for x in n.node_tree.nodes)
    ]
    assert groups_with_uv, "expected uv_map nested inside a node group"
    assert unsafe_displacement_materials([mat]) == {mat.name: "ShaderNodeUVMap"}


def test_uv_map_only_on_surface_not_flagged():
    pf.ops.object.clear_scene()
    obj = pf.ops.primitives.mesh_plane(size=2)
    uvs = pf.ops.attr.uv_coords(obj)
    pf.ops.attr.write_uv_coords(obj, uvs * 2)
    surface = bricks.bricks_distribution(
        np.random.default_rng(0), pf.nodes.shader.uv_map(uv_map="UVMap")
    )
    displacement = bricks.bricks_distribution(
        np.random.default_rng(1), pf.nodes.shader.coord().uv
    )
    pf.ops.object.set_material(
        obj, surface=surface.surface, displacement=displacement.displacement
    )
    mat = obj.item().active_material
    assert unsafe_displacement_materials([mat]) == {}


def _plane_with_material(coord_node, valid_uv=True):
    pf.ops.object.clear_scene()
    obj = pf.ops.primitives.mesh_plane(size=2)
    if valid_uv:
        uvs = pf.ops.attr.uv_coords(obj)
        pf.ops.attr.write_uv_coords(obj, uvs * 2)
    mat = bricks.bricks_distribution(np.random.default_rng(0), coord_node)
    pf.ops.object.set_material(obj, surface=mat.surface, displacement=mat.displacement)
    return obj.item()


def _remove_uv_layers(obj):
    layers = obj.data.uv_layers
    while len(layers):
        layers.remove(layers[0])


def _collapse_active_uv(obj):
    layer = obj.data.uv_layers.active
    layer.data.foreach_set("uv", np.zeros(len(obj.data.loops) * 2))


def test_texcoord_uv_with_valid_layer_no_issue():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    assert check_material_uv_coords(obj) == []


def test_texcoord_uv_no_layer_missing():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _remove_uv_layers(obj)
    issues = check_material_uv_coords(obj)
    assert len(issues) == 1
    assert issues[0].severity == MISSING
    assert issues[0].layer == ACTIVE_RENDER
    assert issues[0].object_name == obj.name


def test_texcoord_uv_degenerate_layer():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _collapse_active_uv(obj)
    issues = check_material_uv_coords(obj)
    assert len(issues) == 1
    assert issues[0].severity == DEGENERATE


def test_non_uv_coord_no_false_positive():
    obj = _plane_with_material(pf.nodes.shader.coord().generated, valid_uv=True)
    _remove_uv_layers(obj)
    assert check_material_uv_coords(obj) == []


def test_geometry_position_no_false_positive():
    obj = _plane_with_material(pf.nodes.shader.geometry().position, valid_uv=True)
    _remove_uv_layers(obj)
    assert check_material_uv_coords(obj) == []


def test_named_uv_map_missing():
    obj = _plane_with_material(pf.nodes.shader.uv_map(uv_map="MyUV"), valid_uv=True)
    issues = check_material_uv_coords(obj)
    missing = [i for i in issues if i.severity == MISSING]
    assert len(missing) == 1
    assert missing[0].layer == "MyUV"


def test_accept_by_material_index():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _remove_uv_layers(obj)
    issues = check_material_uv_coords(obj, mat_index=0)
    assert len(issues) == 1
    assert issues[0].severity == MISSING


def test_uv_issues_flag_missing():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _remove_uv_layers(obj)
    issues = check_material_uv_coords(obj)
    assert any(i.severity == MISSING for i in issues)


def test_uv_issues_degenerate_not_missing():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _collapse_active_uv(obj)
    issues = check_material_uv_coords(obj)
    assert issues
    assert all(i.severity != MISSING for i in issues)
    assert any(i.severity == DEGENERATE for i in issues)
