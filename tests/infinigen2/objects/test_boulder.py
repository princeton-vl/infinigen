# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import ast
from collections.abc import Callable

import numpy as np
import procfunc as pf
import pytest
from procfunc.codegen import to_python
from procfunc.tracer import TraceLevel

from infinigen2.objects import boulder

pytestmark = pytest.mark.skip(reason="needs a Blender extension; excluded for now")

_BOULDER_STYLE_GENERATORS = [
    boulder.boulder_abstract_rand,
    boulder.boulder_cauliflower_rand,
    boulder.boulder_crystalline_rand,
    boulder.boulder_flatstones_rand,
    boulder.boulder_ridged_rand,
    boulder.boulder_rock_rand,
]

_BOULDER_GENERATORS = [
    *_BOULDER_STYLE_GENERATORS,
    boulder.boulder_rand,
]


def _vertex_positions(mesh: pf.MeshObject) -> np.ndarray:
    return np.array([vertex.co[:] for vertex in mesh.item().data.vertices])


@pytest.mark.parametrize("generator", _BOULDER_GENERATORS)
def test_boulder_generators_fit_dimensions_and_ground(
    generator: Callable[..., boulder.BoulderResult],
) -> None:
    expected = np.array((1.2, 0.9, 0.6))
    result = generator(
        np.random.default_rng(7),
        dimensions=pf.Vector(expected),
    )
    positions = _vertex_positions(result.mesh)

    np.testing.assert_allclose(np.ptp(positions, axis=0), expected, rtol=1e-5)
    assert np.isclose(positions[:, 2].min(), 0.0)
    assert len(result.mesh.item().data.materials) == 1


def test_boulder_applies_explicit_material() -> None:
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    result = boulder.boulder(
        material=material,
        mesh_resolution=0.25,
    )

    assert len(result.mesh.item().data.materials) == 1


def test_boulder_rand_spans_every_style_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: set[str] = set()
    mesh = pf.ops.primitives.mesh_cube()
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    def capture(name: str) -> Callable[..., boulder.BoulderResult]:
        def generator(
            _rng: pf.RNG,
            dimensions: pf.Vector | None = None,
            material: pf.Material | None = None,
            mesh_resolution: float = 0.02,
        ) -> boulder.BoulderResult:
            assert dimensions is not None
            assert material is not None
            seen.add(name)
            return boulder.BoulderResult(mesh=mesh)

        return generator

    for generator in _BOULDER_STYLE_GENERATORS:
        monkeypatch.setattr(boulder, generator.__name__, capture(generator.__name__))
    for seed in range(256):
        boulder.boulder_rand(
            np.random.default_rng(seed),
            dimensions=pf.Vector((1.0, 1.0, 1.0)),
            material=material,
        )

    assert seen == {generator.__name__ for generator in _BOULDER_STYLE_GENERATORS}


def test_boulder_rand_traces_and_generates_python() -> None:
    graph = pf.trace(
        boulder.boulder_rand,
        trace_level=TraceLevel.RANDOM_CONTROL,
        rng=np.random.default_rng(12),
    )
    code = to_python(graph, toplevel_as_maincall=False)

    ast.parse(code)
    assert "boulder" in code
