# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from collections.abc import Callable

import bpy
import procfunc as pf
import pytest

from infinigen2.util.polycount import estimated_eval_tricount


def quads() -> pf.MeshObject:
    return pf.ops.primitives.mesh_cube(size=1.0)


def tris() -> pf.MeshObject:
    return pf.ops.primitives.mesh_icosphere(subdivisions=2, radius=1.0)


def ngon_caps() -> pf.MeshObject:
    return pf.ops.primitives.mesh_cylinder(
        vertices=7, radius=1.0, depth=2.0, end_fill_type="NGON"
    )


def pentagon() -> pf.MeshObject:
    return pf.ops.primitives.mesh_circle(vertices=5, radius=1.0, fill_type="NGON")


def hendecagon() -> pf.MeshObject:
    return pf.ops.primitives.mesh_circle(vertices=11, radius=1.0, fill_type="NGON")


def open_grid() -> pf.MeshObject:
    return pf.ops.primitives.mesh_grid(x_subdivisions=3, y_subdivisions=4)


def mixed() -> pf.MeshObject:
    return pf.ops.primitives.mesh_monkey()


def _evaluated_tris(obj: pf.MeshObject) -> int:
    item = obj.item()
    evaluated = item.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        return len(mesh.loop_triangles)
    finally:
        evaluated.to_mesh_clear()


@pytest.mark.parametrize(
    "build", [quads, tris, ngon_caps, pentagon, hendecagon, open_grid, mixed]
)
@pytest.mark.parametrize("levels", [0, 1, 2, 3])
@pytest.mark.parametrize("subdivision_type", ["CATMULL_CLARK", "SIMPLE"])
def test_estimate_matches_evaluated_mesh(
    build: Callable[[], pf.MeshObject], levels: int, subdivision_type: str
) -> None:
    obj = build()
    rendered = build()
    if levels:
        pf.ops.modifier.subdivide_surface(
            obj,
            levels=levels,
            subdivision_type=subdivision_type,
            _skip_apply=True,
        )
        pf.ops.modifier.subdivide_surface(
            rendered, levels=levels, subdivision_type=subdivision_type
        )

    assert estimated_eval_tricount(obj) == _evaluated_tris(rendered)


def test_stacked_subsurf_levels_sum() -> None:
    obj = quads()
    rendered = quads()
    for levels in (1, 2):
        pf.ops.modifier.subdivide_surface(obj, levels=levels, _skip_apply=True)
        pf.ops.modifier.subdivide_surface(rendered, levels=levels)

    assert estimated_eval_tricount(obj) == _evaluated_tris(rendered)


def test_render_levels_not_viewport_levels() -> None:
    obj = quads()
    pf.ops.modifier.subdivide_surface(obj, levels=3, _skip_apply=True)
    mod = obj.item().modifiers[-1]
    mod.levels = 0

    assert estimated_eval_tricount(obj) == 768
    assert _evaluated_tris(obj) == 12


def test_non_mesh_objects_ignored() -> None:
    obj = pf.ops.primitives.curve_bezier()

    assert estimated_eval_tricount(obj) == 0
