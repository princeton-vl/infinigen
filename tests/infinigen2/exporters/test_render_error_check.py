# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import bpy
import imageio.v3 as iio
import numpy as np
import procfunc as pf
import pytest
from procfunc.util.manifest import import_item

from infinigen2 import GENERATORS_MANIFEST, context
from infinigen2.exporters.render_error_check import (
    SHADER_NODE_COUNT_FAIL,
    AdaptiveSamplingError,
    DisplacementCoordError,
    FrameCheckError,
    HiddenRenderObjectError,
    MissingAttributeError,
    NonFiniteGeometryError,
    SingularTransformError,
    assert_adaptive_sampling_converged,
    assert_frames_not_black,
    assert_geometry_finite,
    assert_material_attributes_present,
    assert_render_objects_visible,
    assert_transforms_nonsingular,
    check_material_uv_coords,
    configure_sample_count_output,
    count_material_nodes,
    detect_cycles_errors,
    hidden_render_objects,
    material_node_issues,
    missing_attribute_issues,
    nonfinite_vertex_counts,
    singular_transform_objects,
    unsafe_displacement_materials,
)
from infinigen2.exporters.render_error_check.adaptive_sampling import (
    _frames_at_sample_cap,
)
from infinigen2.shaders import functionality_lists
from infinigen2.shaders.composites import bricks


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
    functionality_lists.table_top_material_rand,
    functionality_lists.furniture_material_rand,
    functionality_lists.paint_wall_rand,
    functionality_lists.paint_flaked_rand,
    functionality_lists.wall_material_rand,
    functionality_lists.skirt_material_rand,
    functionality_lists.floor_material_rand,
    functionality_lists.ceiling_material_rand,
]


@pytest.mark.parametrize(
    "material_sample", _FUNCTIONALITY_MATERIAL_FUNCS, ids=lambda f: f.__name__
)
def test_functionality_material_under_node_budget(material_sample):
    _assert_material_under_node_budget(
        material_sample, material_sample.__name__, seeds=range(10)
    )


def _bricks_material(vector):
    pf.ops.object.clear_scene()
    obj = pf.ops.primitives.mesh_plane(size=2)
    uvs = pf.ops.attr.uv_coords(obj)
    pf.ops.attr.write_uv_coords(obj, uvs * 2)
    mat = bricks.bricks_rand(np.random.default_rng(0), vector)
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


@pf.nodes.node_function
def _uv_map_nodegroup():
    return pf.nodes.shader.uv_map(uv_map="UVMap")


def test_uv_map_nested_in_group_displacement_flagged():
    mat = _bricks_material(_uv_map_nodegroup())
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
    surface = bricks.bricks_rand(
        np.random.default_rng(0), pf.nodes.shader.uv_map(uv_map="UVMap")
    )
    displacement = bricks.bricks_rand(
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
    mat = bricks.bricks_rand(np.random.default_rng(0), coord_node)
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
    assert "no active-render UV layer" in issues[0]
    assert obj.name in issues[0]


def test_texcoord_uv_degenerate_layer():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _collapse_active_uv(obj)
    issues = check_material_uv_coords(obj)
    assert len(issues) == 1
    assert "degenerate" in issues[0]


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
    assert len(issues) == 1
    assert "MyUV" in issues[0]


def test_accept_by_material_index():
    obj = _plane_with_material(pf.nodes.shader.coord().uv, valid_uv=True)
    _remove_uv_layers(obj)
    issues = check_material_uv_coords(obj, mat_index=0)
    assert len(issues) == 1


def _node_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_fake_user = True
    return mat


def test_normal_map_node_flagged():
    mat = _node_material("normal_map")
    mat.node_tree.nodes.new("ShaderNodeNormalMap")
    issues = material_node_issues(mat)
    assert any("ShaderNodeNormalMap" in i for i in issues)


def test_linked_normal_input_flagged():
    mat = _node_material("linked_normal")
    tree = mat.node_tree
    bsdf = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    tree.links.new(coord.outputs["Normal"], bsdf.inputs["Normal"])
    issues = material_node_issues(mat)
    assert any("'Normal' input set" in i for i in issues)


def test_implicit_texture_vector_flagged():
    mat = _node_material("implicit_vector")
    mat.node_tree.nodes.new("ShaderNodeTexNoise")
    issues = material_node_issues(mat)
    assert any("Vector input unlinked" in i for i in issues)


def test_explicit_texture_vector_ok():
    mat = _node_material("explicit_vector")
    tree = mat.node_tree
    noise = tree.nodes.new("ShaderNodeTexNoise")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    tree.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    assert material_node_issues(mat) == []


def test_baked_constant_vector_still_flagged():
    """A constant baked into the Vector default (unlinked) is ignored by Cycles
    for texture nodes, so it must still be flagged."""
    mat = _node_material("baked_constant_vector")
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Vector"].default_value = (1.0, 1.0, 1.0)
    issues = material_node_issues(mat)
    assert any("Vector input unlinked" in i for i in issues)


def test_one_d_noise_no_implicit_vector():
    mat = _node_material("one_d_noise")
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "1D"
    assert material_node_issues(mat) == []


def test_floating_output_in_group_flagged():
    mat = _node_material("floating_output")
    inner = bpy.data.node_groups.new("inner", "ShaderNodeTree")
    inner.nodes.new("ShaderNodeOutputMaterial")
    grp = mat.node_tree.nodes.new("ShaderNodeGroup")
    grp.node_tree = inner
    issues = material_node_issues(mat)
    assert any("floating output node" in i for i in issues)


def test_top_level_output_not_flagged():
    mat = _node_material("clean_default")
    assert material_node_issues(mat) == []


def test_real_material_no_issues():
    mat = _bricks_material(pf.nodes.shader.coord().object)
    assert material_node_issues(mat) == []


def test_black_frame_check(tmp_path):
    black = tmp_path / "black.png"
    bright = tmp_path / "bright.png"
    iio.imwrite(black, np.zeros((8, 8, 3), dtype=np.uint8))
    iio.imwrite(bright, np.full((8, 8, 3), 128, dtype=np.uint8))

    assert assert_frames_not_black([bright], mean_pixel_thresh=0.02) == {}
    assert assert_frames_not_black([black], mean_pixel_thresh=None) == {}

    with context.override_globals(error_mode_black_frame="warn"):
        dark = assert_frames_not_black([black, bright], mean_pixel_thresh=0.02)
    assert set(dark) == {"black.png"}

    with context.override_globals(error_mode_black_frame="ignore"):
        assert assert_frames_not_black([black], mean_pixel_thresh=0.02) == {}

    with context.override_globals(error_mode_black_frame="error"):
        with pytest.raises(FrameCheckError):
            assert_frames_not_black([black], mean_pixel_thresh=0.02)


def test_override_globals_restores():
    orig = context.globals.error_mode_black_frame
    with context.override_globals(error_mode_black_frame="ignore"):
        assert context.globals.error_mode_black_frame == "ignore"
    assert context.globals.error_mode_black_frame == orig


def test_frames_at_sample_cap_parsing():
    log = (
        "Fra:1 Mem:10M | Scene, ViewLayer | Sample 0/64\n"
        "Fra:1 Mem:12M | Scene, ViewLayer | Sample 64/64\n"
        "Fra:2 Mem:10M | Scene, ViewLayer | Sample 0/64\n"
        "Fra:2 Mem:12M | Scene, ViewLayer | Sample 48/64\n"
    )
    assert _frames_at_sample_cap(log, 64) == {1}
    assert _frames_at_sample_cap(log, 256) == set()


def _adaptive_test_render(tmp_path, max_samples, threshold):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_monkey_add()
    bpy.ops.object.camera_add(location=(0, -4, 0), rotation=(1.5708, 0, 0))
    bpy.context.scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="AREA", location=(2, -2, 3))
    bpy.context.object.data.energy = 200

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 96
    scene.render.resolution_y = 64
    scene.render.filepath = str(tmp_path / "rgb_")
    scene.cycles.device = "CPU"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.samples = max_samples
    scene.cycles.adaptive_min_samples = 0
    scene.cycles.adaptive_threshold = threshold

    configure_sample_count_output(scene.view_layers["ViewLayer"], tmp_path)
    with detect_cycles_errors(replay=False) as render_log:
        bpy.ops.render.render(animation=True)
    return render_log["text"], tmp_path


def test_adaptive_sampling_unconverged_raises(tmp_path):
    log, folder = _adaptive_test_render(tmp_path, max_samples=8, threshold=0.0001)
    with (
        context.override_globals(error_mode_unconverged_samples="error"),
        pytest.raises(AdaptiveSamplingError),
    ):
        assert_adaptive_sampling_converged(
            log, folder, max_samples=8, fail_fraction=0.05
        )


def test_adaptive_sampling_converged_passes(tmp_path):
    log, folder = _adaptive_test_render(tmp_path, max_samples=4096, threshold=0.25)
    fractions = assert_adaptive_sampling_converged(
        log, folder, max_samples=4096, fail_fraction=0.05
    )
    assert all(v <= 0.05 for v in fractions.values())


def test_adaptive_sampling_unconverged_warn_mode(tmp_path):
    log, folder = _adaptive_test_render(tmp_path, max_samples=8, threshold=0.0001)
    with context.override_globals(error_mode_unconverged_samples="warn"):
        fractions = assert_adaptive_sampling_converged(
            log, folder, max_samples=8, fail_fraction=0.05
        )
    assert max(fractions.values()) > 0.05


def _plane():
    pf.ops.object.clear_scene()
    return pf.ops.primitives.mesh_plane(size=1)


def test_nonfinite_geometry_flagged():
    mo = _plane()
    obj = mo.item()
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co[0] = np.nan
    obj.data.vertices.foreach_set("co", co)
    obj.data.update()
    bpy.context.view_layer.update()
    assert nonfinite_vertex_counts([mo]).get(obj.name, 0) >= 1
    with pytest.raises(NonFiniteGeometryError):
        assert_geometry_finite([mo])


def test_finite_geometry_ok():
    mo = _plane()
    assert nonfinite_vertex_counts([mo]) == {}
    assert_geometry_finite([mo])


def test_singular_transform_flagged():
    mo = _plane()
    obj = mo.item()
    obj.scale = (1, 1, 0)
    bpy.context.view_layer.update()
    assert obj.name in singular_transform_objects([mo])
    with pytest.raises(SingularTransformError):
        assert_transforms_nonsingular([mo])


def test_nonsingular_transform_ok():
    mo = _plane()
    assert singular_transform_objects([mo]) == {}
    assert_transforms_nonsingular([mo])


def test_hidden_render_object_flagged():
    mo = _plane()
    mo.item().hide_render = True
    assert hidden_render_objects([mo]) == {mo.item().name: "hide_render"}
    with context.override_globals(error_mode_hidden_render_object="error"):
        with pytest.raises(HiddenRenderObjectError):
            assert_render_objects_visible([mo])


def test_visible_camera_off_flagged():
    mo = _plane()
    mo.item().visible_camera = False
    assert hidden_render_objects([mo]) == {mo.item().name: "visible_camera=False"}


def test_holdout_object_not_flagged():
    mo = _plane()
    mo.item().is_holdout = True
    assert hidden_render_objects([mo]) == {}


def test_visible_object_ok():
    mo = _plane()
    assert hidden_render_objects([mo]) == {}
    assert_render_objects_visible([mo])


def _attribute_material(name, attribute_name):
    mat = _node_material(name)
    node = mat.node_tree.nodes.new("ShaderNodeAttribute")
    node.attribute_type = "GEOMETRY"
    node.attribute_name = attribute_name
    bsdf = next(
        n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    mat.node_tree.links.new(node.outputs["Fac"], bsdf.inputs["Metallic"])
    return mat


def test_missing_attribute_flagged():
    mo = _plane()
    obj = mo.item()
    obj.data.materials.append(_attribute_material("attr_missing", "does_not_exist"))
    bpy.context.view_layer.update()
    assert any("does_not_exist" in i for i in missing_attribute_issues(obj))
    with context.override_globals(error_mode_missing_attribute="error"):
        with pytest.raises(MissingAttributeError):
            assert_material_attributes_present([mo])


def test_present_attribute_ok():
    mo = _plane()
    obj = mo.item()
    obj.data.attributes.new("myattr", "FLOAT", "POINT")
    obj.data.materials.append(_attribute_material("attr_present", "myattr"))
    bpy.context.view_layer.update()
    assert missing_attribute_issues(obj) == []
