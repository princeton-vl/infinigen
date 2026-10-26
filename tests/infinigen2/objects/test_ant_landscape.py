# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import ast
from collections.abc import Callable

import bpy
import numpy as np
import procfunc as pf
import pytest
from procfunc.codegen import to_python
from procfunc.tracer import TraceLevel

from infinigen2.objects import ant_landscape

pytestmark = pytest.mark.skip(reason="needs a Blender extension; excluded for now")

_LANDSCAPE_GENERATORS = [
    ant_landscape.landscape_canyon_rand,
    ant_landscape.landscape_cliff_rand,
    ant_landscape.landscape_generalized_rand,
    ant_landscape.landscape_mesa_rand,
    ant_landscape.landscape_mountain_rand,
    ant_landscape.landscape_rand,
    ant_landscape.landscape_river_rand,
    ant_landscape.landscape_volcano_rand,
]


def _vertex_positions(mesh: pf.MeshObject) -> np.ndarray:
    return np.array([vertex.co[:] for vertex in mesh.item().data.vertices])


def test_landscape_rejects_water_plane() -> None:
    with pytest.raises(ValueError, match="cannot create a water plane"):
        ant_landscape.landscape(water_plane=True)


def test_landscape_returns_only_new_mesh() -> None:
    before = set(bpy.data.objects)
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    mesh = ant_landscape.landscape(
        subdivision_x=8,
        subdivision_y=8,
        random_seed=3,
        material=material,
    )

    assert isinstance(mesh, pf.MeshObject)
    assert mesh.item().type == "MESH"
    assert mesh.item().mode == "OBJECT"
    assert mesh.item().data.materials[0] is material.item()
    assert set(bpy.data.objects) - before == {mesh.item()}


def test_landscape_cleans_multiple_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = set(bpy.data.objects)

    def create_two(**_kwargs: object) -> set[str]:
        bpy.ops.mesh.primitive_cube_add()
        bpy.ops.mesh.primitive_cube_add()
        return {"FINISHED"}

    monkeypatch.setattr(ant_landscape, "_execute_landscape", create_two)

    with pytest.raises(RuntimeError, match="created 2 objects"):
        ant_landscape.landscape()
    assert set(bpy.data.objects) == before


def test_landscape_cleans_up_after_operator_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = set(bpy.data.objects)

    def fail_after_create(**_kwargs: object) -> set[str]:
        bpy.ops.mesh.primitive_cube_add()
        raise RuntimeError("operator failed")

    monkeypatch.setattr(ant_landscape, "_execute_landscape", fail_after_create)

    with pytest.raises(RuntimeError, match="operator failed"):
        ant_landscape.landscape()
    assert set(bpy.data.objects) == before


def test_landscape_maps_dimensions_to_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = pf.ops.primitives.mesh_plane()

    def capture(**kwargs: object) -> pf.MeshObject:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(ant_landscape, "landscape", capture)
    result = ant_landscape.landscape_rand_from_params(
        np.random.default_rng(4),
        dimensions=pf.Vector((4, 6, 2)),
        mesh_resolution=0.5,
    )

    assert isinstance(result, ant_landscape.LandscapeResult)
    assert result.mesh is expected
    assert captured["subdivision_x"] == 8
    assert captured["subdivision_y"] == 12
    assert captured["mesh_size_x"] == 4
    assert captured["mesh_size_y"] == 6
    assert "mesh_size" not in captured


def test_landscape_rand_from_params_forwards_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> pf.MeshObject:
        captured.update(kwargs)
        return pf.ops.primitives.mesh_plane()

    monkeypatch.setattr(ant_landscape, "landscape", capture)
    ant_landscape.landscape_rand_from_params(
        np.random.default_rng(4),
        dimensions=pf.Vector((4, 4, 1)),
        mesh_resolution=1,
        noise_type="marble_noise",
        strata=7.0,
    )

    assert captured["noise_type"] == "marble_noise"
    assert captured["strata"] == 7.0


@pytest.mark.parametrize("generator", _LANDSCAPE_GENERATORS)
def test_isotropic_dimensions_are_a_similarity_transform(
    generator: Callable[..., ant_landscape.LandscapeResult],
) -> None:
    native = generator(
        np.random.default_rng(0),
        dimensions=pf.Vector((2, 2, 2)),
        mesh_resolution=0.02,
    )
    scaled = generator(
        np.random.default_rng(0),
        dimensions=pf.Vector((50, 50, 50)),
        mesh_resolution=0.5,
    )
    expected = _vertex_positions(native.mesh) * 25.0
    actual = _vertex_positions(scaled.mesh)
    deviation = np.abs(expected - actual).max() / max(np.abs(actual).max(), 1.0)
    assert deviation < 1e-3


def test_noise_offsets_stay_within_float32_precision() -> None:
    subdivisions = 400
    result = ant_landscape.landscape_rand_from_params(
        np.random.default_rng(0),
        dimensions=pf.Vector((120, 120, 120)),
        mesh_resolution=0.3,
        edge_falloff="0",
        maximum=1e6,
        minimum=-1e6,
    )
    heights = _vertex_positions(result.mesh)[:, 2].reshape(subdivisions, subdivisions)
    interior = heights[80:320, 80:320]
    for axis in (0, 1):
        assert np.mean(np.diff(interior, axis=axis) == 0.0) < 0.01


def test_landscape_params_scale_with_dimensions() -> None:
    extents = {}
    for width in [10, 50, 200]:
        result = ant_landscape.landscape_mountain_rand(
            np.random.default_rng(0),
            dimensions=pf.Vector((width, width, 10)),
            mesh_resolution=width / 100.0,
        )
        positions = _vertex_positions(result.mesh)
        extents[width] = (positions.max(0) - positions.min(0))[2]

    assert np.allclose(list(extents.values()), extents[10], rtol=1e-3)

    taller = ant_landscape.landscape_mountain_rand(
        np.random.default_rng(0),
        dimensions=pf.Vector((50, 50, 40)),
        mesh_resolution=0.5,
    )
    positions = _vertex_positions(taller.mesh)
    assert (positions.max(0) - positions.min(0))[2] == pytest.approx(
        extents[50] * 4, rel=1e-3
    )


def test_landscape_rand_from_params_repeats_for_one_seed() -> None:
    kwargs = {"dimensions": pf.Vector((4, 4, 1)), "mesh_resolution": 1}
    first = ant_landscape.landscape_rand_from_params(np.random.default_rng(3), **kwargs)
    second = ant_landscape.landscape_rand_from_params(
        np.random.default_rng(3), **kwargs
    )

    assert np.array_equal(_vertex_positions(first.mesh), _vertex_positions(second.mesh))


@pytest.mark.parametrize("generator", _LANDSCAPE_GENERATORS)
def test_landscape_generators_return_mesh(
    generator: Callable[..., ant_landscape.LandscapeResult],
) -> None:
    result = generator(
        np.random.default_rng(5),
        dimensions=pf.Vector((4, 4, 1)),
        mesh_resolution=1,
    )

    assert isinstance(result, ant_landscape.LandscapeResult)
    assert isinstance(result.mesh, pf.MeshObject)
    assert len(result.mesh.item().data.vertices) == 16
    assert len(result.mesh.item().data.materials) == 1


def test_landscape_mesa_is_deterministic() -> None:
    kwargs = {"dimensions": pf.Vector((4, 4, 1)), "mesh_resolution": 1}
    first = ant_landscape.landscape_mesa_rand(
        np.random.default_rng(9),
        **kwargs,
    )
    first_positions = _vertex_positions(first.mesh)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    second = ant_landscape.landscape_mesa_rand(
        np.random.default_rng(9),
        **kwargs,
    )

    np.testing.assert_allclose(_vertex_positions(second.mesh), first_positions)


def test_landscape_generator_traces_without_executing_operator() -> None:
    graph = pf.trace(
        ant_landscape.landscape_generalized_rand,
        trace_level=TraceLevel.RANDOM_CONTROL,
        rng=np.random.default_rng(12),
        dimensions=pf.Vector((4, 4, 1)),
        mesh_resolution=1,
    )
    code = to_python(graph, toplevel_as_maincall=False)

    ast.parse(code)
    assert "landscape(" in code


@pytest.mark.parametrize("generator", _LANDSCAPE_GENERATORS)
def test_landscape_generators_respect_material_override(
    generator: Callable[..., ant_landscape.LandscapeResult],
) -> None:
    material = pf.Material(
        surface=pf.nodes.shader.principled_bsdf(base_color=(0.1, 0.2, 0.3, 1.0))
    )

    result = generator(
        np.random.default_rng(5),
        dimensions=pf.Vector((4, 4, 1)),
        mesh_resolution=1,
        material=material,
    )

    assert result.mesh.item().data.materials[0] is material.item()
