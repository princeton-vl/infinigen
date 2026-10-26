# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import (
    fabric_general_rand,
    fabric_sturdy_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "CushionResult",
    "cushion_bed_pillow_rand",
    "cushion_box_geometry",
    "cushion_throw_pillow_rand",
    "optional_piping_radius_rand",
    "support_loop_offset",
]


class CushionResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def support_loop_offset(
    radius: t.SocketOrVal[float],
    size: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return pf.nodes.math.vector_minimum(
        pf.nodes.math.combine_xyz(x=radius, y=radius, z=radius), size * 0.4
    )


@pf.nodes.node_function
def _piped_and_anchored(
    body: t.SocketOrVal[pf.MeshObject],
    seam: t.SocketOrVal[bool],
    material: t.SocketOrVal[pf.Material],
    piping_radius: t.SocketOrVal[float],
    size: t.SocketOrVal[pf.Vector],
    anchor: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.MeshObject]:
    # a fully creased seam and a tube swept along it subdivide to the same B-spline
    loops = pf.nodes.geo.mesh_to_curve(body, selection=seam)
    loops = pf.nodes.geo.remove_attribute(loops, name="crease_edge")
    profile = pf.nodes.geo.curve_circle(resolution=6, radius=piping_radius * 1.2)
    piping = curve_to_mesh_with_uv(curve=loops, profile=profile).mesh
    piping = pf.nodes.func.switch(switch=piping_radius > 1e-5, b=piping)
    cushion = pf.nodes.geo.join_geometry([body, piping])
    cushion = pf.nodes.geo.set_material(cushion, material)
    return pf.nodes.geo.transform(
        geometry=cushion,
        translation=size * 0.5 - anchor * size,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )


@pf.nodes.node_function
def cushion_box_geometry(
    size: t.SocketOrVal[pf.Vector],
    material: t.SocketOrVal[pf.Material],
    anchor: t.SocketOrVal[pf.Vector],
    edge_radius: t.SocketOrVal[float],
    crown: t.SocketOrVal[float],
    piping_radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    boxing = size.z - crown * 2.0
    body_size = pf.nodes.math.combine_xyz(x=size.x, y=size.y, z=boxing)
    body = mesh_util.box_with_support_loops(
        size=body_size,
        location=(0, 0, 0),
        anchor=(0.5, 0.5, 0.5),
        vertices_x=7,
        vertices_y=7,
        vertices_z=4,
        support_loop_offset=support_loop_offset(edge_radius, body_size),
    )

    edge = pf.nodes.geo.input_mesh_edge_vertices()
    p1 = pf.nodes.math.vector_absolute(edge.position_1)
    p2 = pf.nodes.math.vector_absolute(edge.position_2)
    limit = body_size * 0.5 - (1e-5, 1e-5, 1e-5)
    on_cap = pf.nodes.func.boolean_and(a=p1.z > limit.z, b=p2.z > limit.z)
    along_x = pf.nodes.func.boolean_and(a=p1.x > limit.x, b=p2.x > limit.x)
    along_y = pf.nodes.func.boolean_and(a=p1.y > limit.y, b=p2.y > limit.y)
    seam = pf.nodes.func.boolean_and(
        a=on_cap, b=pf.nodes.func.boolean_or(a=along_x, b=along_y)
    )
    body = pf.nodes.geo.store_named_attribute(
        geometry=body,
        domain="EDGE",
        name="crease_edge",
        selection=seam,
        value=(piping_radius > 1e-5).astype(dtype=float),
    )

    position = pf.nodes.geo.input_position()
    dome_x = pf.nodes.math.cos(position.x / size.x * math.pi)
    dome_y = pf.nodes.math.cos(position.y / size.y * math.pi)
    on_panel = pf.nodes.math.absolute(position.z) > boxing * 0.5 - 1e-5
    lift = dome_x * dome_y * crown * on_panel.astype(dtype=float)
    body = pf.nodes.geo.set_position(
        geometry=body,
        offset=pf.nodes.math.combine_xyz(z=pf.nodes.math.sign(position.z) * lift),
    )
    return _piped_and_anchored(body, seam, material, piping_radius, size, anchor)


@pf.nodes.node_function
def _knife_edge_cushion_geometry(
    size: t.SocketOrVal[pf.Vector],
    material: t.SocketOrVal[pf.Material],
    fullness: t.SocketOrVal[float],
    corner_pinch: t.SocketOrVal[float],
    piping_radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    # superellipse panels meet the seam convex; grid rows bunch toward the seam
    grid = pf.nodes.geo.mesh_grid(size_x=2.0, size_y=2.0, vertices_x=7, vertices_y=7)
    position = pf.nodes.geo.input_position()
    edge_u = 1.0 - pf.nodes.math.absolute(position.x)
    edge_v = 1.0 - pf.nodes.math.absolute(position.y)
    u = pf.nodes.math.sign(position.x) * (1.0 - edge_u * edge_u * edge_u)
    v = pf.nodes.math.sign(position.y) * (1.0 - edge_v * edge_v * edge_v)
    bulge_u = 1.0 - pf.nodes.math.power(pf.nodes.math.absolute(u), fullness)
    bulge_v = 1.0 - pf.nodes.math.power(pf.nodes.math.absolute(v), fullness)
    bulge = pf.nodes.math.power(
        pf.nodes.math.maximum(bulge_u * bulge_v, 0.0), 1.0 / fullness
    )
    x = u * size.x * 0.5 * (1.0 - corner_pinch * (1.0 - v * v))
    y = v * size.y * 0.5 * (1.0 - corner_pinch * (1.0 - u * u))
    top = pf.nodes.geo.set_position(
        geometry=grid.mesh,
        position=pf.nodes.math.combine_xyz(x=x, y=y, z=bulge * size.z * 0.5),
    )
    top = pf.nodes.geo.store_named_attribute(
        geometry=top,
        name="UVMap",
        value=pf.nodes.math.combine_xyz(x=x, y=y),
        domain="CORNER",
        data_type="FLOAT2",
    )
    bottom = pf.nodes.geo.transform(
        geometry=top, translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, -1)
    )
    body = pf.nodes.geo.join_geometry([top, pf.nodes.geo.flip_faces(bottom)])
    body = pf.nodes.geo.merge_by_distance(body, distance=1e-5)

    edge = pf.nodes.geo.input_mesh_edge_vertices()
    seam = pf.nodes.func.boolean_and(
        a=pf.nodes.math.absolute(edge.position_1.z) < 1e-6,
        b=pf.nodes.math.absolute(edge.position_2.z) < 1e-6,
    )
    body = pf.nodes.geo.store_named_attribute(
        geometry=body, domain="EDGE", name="crease_edge", selection=seam, value=1.0
    )
    return _piped_and_anchored(
        body, seam, material, piping_radius, size, (0.5, 0.5, 0.0)
    )


def _no_piping_rand(rng: pf.RNG) -> float:
    return 0.0


def _piping_radius_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.0025, 0.0055)


def optional_piping_radius_rand(rng: pf.RNG, piped_weight: float) -> float:
    rng_choice, rng_radius = rng.spawn(2)
    piping_fn = pf.control.choice(
        rng_choice,
        [(_piping_radius_rand, piped_weight), (_no_piping_rand, 1.0 - piped_weight)],
    )
    return piping_fn(rng_radius)


def _pillow(
    size: pf.Vector,
    material: pf.Material,
    fullness: float,
    corner_pinch: float,
    piping_radius: float,
) -> CushionResult:
    geometry = _knife_edge_cushion_geometry(
        size=size,
        material=material,
        fullness=fullness,
        corner_pinch=corner_pinch,
        piping_radius=piping_radius,
    )
    obj = pf.nodes.to_mesh_object(geometry)
    pf.ops.modifier.subdivide_surface(obj, levels=6, _skip_apply=True)
    return CushionResult(mesh=obj)


def cushion_throw_pillow_rand(
    rng: pf.RNG,
    size: pf.Vector | None = None,
    piping_radius: float | None = None,
    material: pf.Material | None = None,
    fullness: float | None = None,
) -> CushionResult:
    """Square throw pillow, 16-24 in (0.40-0.60 m) across."""
    rng, rng_piping, rng_fabric = rng.spawn(3)
    if size is None:
        side = pf.random.uniform(rng, 0.4, 0.6)
        thickness = side * pf.random.uniform(rng, 0.25, 0.33)
        depth = side * pf.random.uniform(rng, 0.97, 1.03)
        size = pf.Vector((side, depth, thickness))
    if fullness is None:
        fullness = pf.random.uniform(rng, 1.4, 3.0)
    corner_pinch = pf.random.uniform(rng, 0.02, 0.07)
    if piping_radius is None:
        piping_radius = optional_piping_radius_rand(rng_piping, 0.4)
    if material is None:
        material = fabric_sturdy_rand(rng_fabric, pf.nodes.shader.coord().uv)
    return _pillow(size, material, fullness, corner_pinch, piping_radius)


def cushion_bed_pillow_rand(
    rng: pf.RNG,
    size: pf.Vector | None = None,
    fullness: float | None = None,
    material: pf.Material | None = None,
) -> CushionResult:
    """Sleeping pillow, 19-21 in deep, 26-36 in long (standard to king), 6-9 in loft."""
    rng, rng_fabric = rng.spawn(2)
    if size is None:
        short_side = pf.random.uniform(rng, 0.48, 0.53)
        long_side = pf.random.uniform(rng, 0.66, 0.92)
        loft = pf.random.uniform(rng, 0.15, 0.22)
        size = pf.Vector((short_side, long_side, loft))
    if fullness is None:
        fullness = pf.random.uniform(rng, 1.4, 3.0)
    corner_pinch = pf.random.uniform(rng, 0.02, 0.07)
    if material is None:
        material = fabric_general_rand(rng_fabric, pf.nodes.shader.coord().uv)
    return _pillow(size, material, fullness, corner_pinch, 0.0)
