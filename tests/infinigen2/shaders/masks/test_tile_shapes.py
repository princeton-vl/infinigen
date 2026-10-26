# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import tempfile
from collections.abc import Callable
from pathlib import Path

import bpy
import numpy as np
import procfunc as pf
import pytest
from procfunc.nodes import types as t
from procfunc.nodes.util import bpy_node_info

from infinigen2.shaders.masks import tile_shapes

pytestmark = pytest.mark.render_gpu

AOV_NAME = "tile_coordinates"
SHAPE_NAMES = (
    "triangle",
    "star",
    "spanish_bound",
    "shell",
    "hexagon",
    "herringbone",
    "diamond",
    "chevron",
    "brick",
    "basket_weave",
    "square",
)
CENTER_ROUNDTRIP_SHAPES = SHAPE_NAMES


def _nodegroup(fn: Callable[[], object]) -> bpy.types.NodeTree:
    graph = pf.nodes.function_to_compute_graph(fn)
    return pf.nodes.as_nodegroup(graph, bpy_node_info.NodeGroupType.SHADER)


def _shape_result(
    name: str,
    shape_count: int,
    vector: t.SocketOrVal[pf.Vector] | None = None,
):
    if vector is None:
        coordinate = pf.nodes.shader.coord().generated
        local_x = (coordinate.x * shape_count) % 1.0
        vector = pf.nodes.math.combine_xyz(
            x=local_x * 12.0 - 6.0,
            y=coordinate.y * 12.0 - 6.0,
        )
    return getattr(tile_shapes, name)(
        vector=vector,
        subtiles_number=2.0,
        aspect_ratio=3.0,
        border=0.05,
        flatness=0.9,
    )


def _material(field: str, shape_names: tuple[str, ...]) -> bpy.types.Material:
    def build() -> dict[str, pf.ProcNode]:
        outputs = {}
        for name in shape_names:
            initial = _shape_result(name, len(shape_names))
            if field == "center_idx_error":
                centered = _shape_result(
                    name, len(shape_names), initial.cell_center_pos
                )
                outputs[name] = centered.cell_center_idx - initial.cell_center_idx
            elif field == "center_vector":
                centered = _shape_result(
                    name, len(shape_names), initial.cell_center_pos
                )
                outputs[name] = centered.vector
            elif field == "center_distance":
                centered = _shape_result(
                    name, len(shape_names), initial.cell_center_pos
                )
                outputs[name] = centered.distance_from_edge
            else:
                outputs[name] = getattr(initial, field)
        return outputs

    return _output_material(build, field)


def _output_material(
    build: Callable[[], dict[str, pf.ProcNode]], name: str
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    group = nodes.new("ShaderNodeGroup")
    group.node_tree = _nodegroup(build)
    selected = group.outputs[0]
    coordinate = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    material.node_tree.links.new(coordinate.outputs["Generated"], separate.inputs[0])

    for index, socket in enumerate(group.outputs[1:], start=1):
        compare = nodes.new("ShaderNodeMath")
        compare.operation = "GREATER_THAN"
        compare.inputs[1].default_value = index / len(group.outputs)
        material.node_tree.links.new(separate.outputs["X"], compare.inputs[0])
        mix = nodes.new("ShaderNodeMixRGB")
        material.node_tree.links.new(compare.outputs[0], mix.inputs[0])
        material.node_tree.links.new(selected, mix.inputs[1])
        material.node_tree.links.new(socket, mix.inputs[2])
        selected = mix.outputs[0]

    aov = nodes.new("ShaderNodeOutputAOV")
    aov.aov_name = AOV_NAME
    material.node_tree.links.new(selected, aov.inputs["Color"])
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _scene(material: bpy.types.Material, width: int, height: int) -> bpy.types.Scene:
    scene = bpy.data.scenes.new("tile_coordinates")
    mesh = bpy.data.meshes.new("tile_coordinates")
    mesh.from_pydata(
        [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
        [],
        [(0, 1, 2, 3)],
    )
    plane = bpy.data.objects.new("tile_coordinates", mesh)
    mesh.materials.append(material)
    scene.collection.objects.link(plane)

    camera_data = bpy.data.cameras.new("tile_coordinates")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2
    camera = bpy.data.objects.new("tile_coordinates", camera_data)
    camera.location.z = 1
    scene.collection.objects.link(camera)
    scene.camera = camera

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 1
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    active_scene = bpy.context.window.scene
    bpy.context.window.scene = scene
    aov = scene.view_layers[0].aovs.add()
    aov.name = AOV_NAME
    aov.type = "COLOR"
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs[AOV_NAME], composite.inputs["Image"])
    bpy.context.window.scene = active_scene
    return scene


def _render(
    field: str,
    shape_names: tuple[str, ...] = SHAPE_NAMES,
    tile_size: int = 64,
    material: bpy.types.Material | None = None,
) -> dict[str, np.ndarray]:
    width = len(shape_names) * tile_size
    if material is None:
        material = _material(field, shape_names)
    scene = _scene(material, width, tile_size)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "field.exr"
        scene.render.filepath = str(path.with_suffix(""))
        bpy.ops.render.render(scene=scene.name, write_still=True)
        image = bpy.data.images.load(str(path), check_existing=False)
        buffer = np.empty(
            image.size[0] * image.size[1] * image.channels, dtype=np.float32
        )
        image.pixels.foreach_get(buffer)
        array = buffer.reshape(image.size[1], image.size[0], image.channels)
        array = np.flipud(array.copy())
        bpy.data.images.remove(image)
    return {
        name: array[:, index * tile_size : (index + 1) * tile_size, :3]
        for index, name in enumerate(shape_names)
    }


def test_tile_coordinates_are_integer_and_constant_per_cell():
    indices = _render("cell_center_idx")
    positions = _render("cell_center_pos")
    center_errors = _render("center_idx_error", CENTER_ROUNDTRIP_SHAPES)

    for name in SHAPE_NAMES:
        index = indices[name].reshape(-1, 3)
        position = positions[name].reshape(-1, 3)
        rounded_index = np.round(index)
        np.testing.assert_allclose(index, rounded_index, atol=5e-3, err_msg=name)
        np.testing.assert_allclose(index[:, 2], 0.0, atol=5e-3, err_msg=name)
        np.testing.assert_allclose(position[:, 2], 0.0, atol=5e-3, err_msg=name)

        unique_indices, inverse = np.unique(rounded_index, axis=0, return_inverse=True)
        cell_positions = []
        for cell_number in range(len(unique_indices)):
            samples = position[inverse == cell_number]
            center_error = np.max(np.abs(samples - samples.mean(axis=0)))
            assert center_error <= 5e-3, name
            cell_positions.append(samples.mean(axis=0))

        quantized_positions = np.round(np.asarray(cell_positions) / 0.05)
        unique_positions = np.unique(quantized_positions, axis=0)
        assert len(unique_indices) == len(unique_positions), name
        assert len(unique_indices) >= 8, name

    for name, error in center_errors.items():
        np.testing.assert_allclose(error, 0.0, atol=5e-3, err_msg=name)

    spanish_indices = _render("cell_center_idx", ("spanish_bound",))["spanish_bound"]
    spanish_variants = np.round(spanish_indices[..., 0]).astype(int) % 3
    np.testing.assert_array_equal(np.unique(spanish_variants), (0, 1, 2))


def test_random_shape_centers_round_trip_through_layout_transform() -> None:
    def build() -> dict[str, pf.ProcNode]:
        coordinate = pf.nodes.shader.coord().generated
        vector = pf.nodes.math.combine_xyz(
            x=coordinate.x * 12.0 - 6.0,
            y=coordinate.y * 12.0 - 6.0,
        )
        outputs = {}
        for name in SHAPE_NAMES:
            shape_rand = getattr(tile_shapes, f"{name}_rand")
            params = dict(
                subtiles_number=2.0,
                aspect_ratio=3.0,
                border=0.05,
                flatness=0.9,
            )
            initial = shape_rand(np.random.default_rng(5), vector, **params)
            centered = shape_rand(
                np.random.default_rng(5), initial.cell_center_pos, **params
            )
            outputs[name] = centered.cell_center_idx - initial.cell_center_idx
        return outputs

    errors = _render(
        "random_center_idx_error",
        material=_output_material(build, "random_center_idx_error"),
    )
    for name, error in errors.items():
        np.testing.assert_allclose(error, 0.0, atol=5e-3, err_msg=name)


def test_random_shape_vector_uses_caller_coordinate_space() -> None:
    def build() -> dict[str, pf.ProcNode]:
        vector = pf.nodes.math.combine_xyz(x=0.37, y=0.61)
        result = tile_shapes.square_rand(
            np.random.default_rng(5),
            vector,
            subtiles_number=1.0,
            aspect_ratio=3.0,
            border=0.05,
            flatness=0.9,
        )
        return {"square": result.vector - (vector - result.cell_center_pos)}

    errors = _render(
        "random_vector_error",
        ("square",),
        tile_size=4,
        material=_output_material(build, "random_vector_error"),
    )["square"]
    np.testing.assert_allclose(errors, 0.0, atol=5e-3)


def test_square_default_has_finite_center() -> None:
    def build() -> dict[str, pf.ProcNode]:
        result = tile_shapes.square(vector=(0.25, 0.25, 0.0))
        return {"square": result.cell_center_pos}

    center = _render(
        "square_default_center",
        ("square",),
        tile_size=4,
        material=_output_material(build, "square_default_center"),
    )["square"]
    assert np.all(np.isfinite(center))
    expected = np.broadcast_to((0.5, 0.5, 0.0), center.shape)
    np.testing.assert_allclose(center, expected, atol=5e-3)


@pytest.mark.parametrize("subtiles", (1, 3))
@pytest.mark.parametrize("offset", (0.0, -3.0))
def test_basket_weave_coordinates_agree_with_physical_tiles(
    subtiles: int, offset: float
) -> None:
    short_center = 0.5 / subtiles
    cases = {
        "horizontal": ((2.5, 1.5 + short_center, 0.0), (0.0, 0.15 / subtiles, 0.0)),
        "vertical": ((0.5 + short_center, 2.0, 0.0), (0.15 / subtiles, 0.0, 0.0)),
    }

    def build() -> dict[str, pf.ProcNode]:
        outputs = {}
        for name, (center, delta) in cases.items():
            expected = np.array(center) + (offset, offset, 0.0)
            points = [
                tile_shapes.basket_weave(
                    vector=tuple(expected + sign * np.array(delta)),
                    aspect_ratio=2.0,
                    subtiles_number=subtiles,
                    border=0.05,
                    flatness=0.9,
                )
                for sign in (-1, 1)
            ]
            errors = [
                pf.nodes.math.vector_length(point.cell_center_pos - tuple(expected))
                for point in points
            ]
            outputs[name] = pf.nodes.math.combine_xyz(
                x=errors[0] + errors[1],
                y=pf.nodes.math.vector_length(
                    points[0].cell_center_idx - points[1].cell_center_idx
                ),
                z=points[0].mask.astype(dtype=float)
                * points[1].mask.astype(dtype=float),
            )
        return outputs

    material = _output_material(build, "basket_weave_physical_tiles")
    rendered = _render("basket_weave", tuple(cases), tile_size=4, material=material)
    for name, values in rendered.items():
        np.testing.assert_allclose(
            values[..., :2],
            0.0,
            atol=5e-3,
            err_msg=name,
        )
        assert np.all(values[..., 2] > 0.5), name


def test_tile_vectors_are_centered_in_local_shape_space():
    vectors = _render("vector")
    centered_vectors = _render("center_vector")

    for name in SHAPE_NAMES:
        assert np.all(np.isfinite(vectors[name])), name
        assert np.ptp(vectors[name][..., :2]) > 0.1, name
        np.testing.assert_allclose(vectors[name][..., 2], 0.0, atol=5e-3, err_msg=name)

    for name in CENTER_ROUNDTRIP_SHAPES:
        np.testing.assert_allclose(centered_vectors[name], 0.0, atol=5e-3, err_msg=name)


@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_tile_vectors_ignore_input_z(name: str) -> None:
    def build() -> dict[str, pf.ProcNode]:
        result = _shape_result(name, 1, vector=(0.37, 0.61, 7.0))
        return {name: result.vector}

    material = _output_material(build, "tile_vector_input_z")
    vector = _render(name, (name,), tile_size=4, material=material)[name]
    np.testing.assert_allclose(vector[..., 2], 0.0, atol=5e-3, err_msg=name)


def test_tile_edge_distances_vary_and_peak_at_centers():
    distances = _render("distance_from_edge")
    center_distances = _render("center_distance")

    for name in SHAPE_NAMES:
        distance = distances[name][..., 0]
        center_distance = center_distances[name][..., 0]
        assert np.all(np.isfinite(distance)), name
        assert np.ptp(distance) > 5e-3, name
        assert np.min(center_distance) > 5e-3, name
        assert np.max(distance - center_distance) < 5e-3, name
