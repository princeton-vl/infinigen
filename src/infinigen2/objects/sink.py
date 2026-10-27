# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: original Infinigen v1 sink geometry (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/table_decorations/sink.py)
# - Alexander Raistrick: added the V1 placeholder, detail optimization, and revised cutter (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/table_decorations/sink.py)
# - Alexander Raistrick: refactor for Infinigen2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import decorative_material_rand

__all__ = ["SinkResult", "sink", "sink_rand"]


class SinkResult(NamedTuple):
    mesh: pf.MeshObject
    cutter: pf.MeshObject


class _SinkGeometryResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    cutter: pf.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def _sink_geometry(
    material: t.SocketOrVal[pf.Material],
    width: t.SocketOrVal[float] = 0.6,
    depth: t.SocketOrVal[float] = 0.45,
    curvature: t.SocketOrVal[float] = 1.0,
    upper_height: t.SocketOrVal[float] = 0.3,
    lower_height: t.SocketOrVal[float] = 0.005,
    hole_radius: t.SocketOrVal[float] = 0.035,
    margin: t.SocketOrVal[float] = 0.035,
    water_tap_margin: t.SocketOrVal[float] = 0.11,
) -> _SinkGeometryResult:
    outer_width = depth + margin + water_tap_margin
    outer = pf.nodes.geo.curve_quadrilateral(
        width=outer_width,
        height=width + margin,
    )
    outer = pf.nodes.geo.transform(
        geometry=outer,
        translation=pf.nodes.math.combine_xyz(water_tap_margin * -0.5),
    )
    fillet_radius = pf.nodes.math.minimum(a=depth, b=width) * 0.1
    outer = pf.nodes.geo.fillet_curve_poly(
        curve=outer,
        radius=fillet_radius,
        count=10,
    )

    inner_curve = pf.nodes.geo.curve_quadrilateral(width=depth, height=width)
    inner = pf.nodes.geo.fillet_curve_poly(
        curve=inner_curve,
        radius=fillet_radius,
        count=50,
    )
    rim = pf.nodes.geo.fill_curve(pf.nodes.geo.join_geometry([outer, inner]))
    rim = pf.nodes.geo.extrude_mesh(
        mesh=rim,
        offset_scale=upper_height - lower_height,
    )
    rim = pf.nodes.geo.transform(
        geometry=rim.mesh,
        translation=pf.nodes.math.combine_xyz(z=lower_height),
    )

    inner_scaled = pf.nodes.geo.transform(
        geometry=inner,
        scale=(0.99, 0.99, 1.0),
    )
    bowl = pf.nodes.geo.fill_curve(pf.nodes.geo.join_geometry([inner, inner_scaled]))
    bowl = pf.nodes.geo.extrude_mesh(mesh=bowl, offset_scale=lower_height)
    position = pf.nodes.geo.input_position()
    warped_position = pf.nodes.math.combine_xyz(
        x=position.x * curvature,
        y=position.y * curvature,
        z=position.z,
    )
    bowl = pf.nodes.geo.set_position(
        geometry=bowl.mesh,
        position=warped_position,
        selection=(position.z < 0.0).astype(dtype=bool),
    )

    drain_line = pf.nodes.geo.curve_line(
        start=(0.0, 0.0, 0.0),
        end=pf.nodes.math.combine_xyz(z=lower_height - 0.01),
    )
    drain_circle = pf.nodes.geo.curve_circle(radius=hole_radius)
    drain_wall = pf.nodes.geo.curve_to_mesh(
        curve=drain_line,
        profile_curve=drain_circle,
    )
    drain_wall = pf.nodes.geo.transform(
        geometry=drain_wall,
        translation=pf.nodes.math.combine_xyz(z=lower_height),
    )

    drain_inner = pf.nodes.geo.transform(
        geometry=drain_circle,
        scale=(0.7, 0.7, 1.0),
    )
    drain_ring = pf.nodes.geo.fill_curve(
        pf.nodes.geo.join_geometry([drain_inner, drain_circle])
    )
    drain_ring = pf.nodes.geo.transform(
        geometry=drain_ring,
        translation=pf.nodes.math.combine_xyz(z=lower_height - 0.01),
    )
    drain_ring = pf.nodes.geo.extrude_mesh(
        mesh=drain_ring,
        offset_scale=lower_height,
        individual=False,
    )

    bowl_outline = pf.nodes.geo.transform(
        geometry=inner,
        scale=pf.nodes.math.combine_xyz(x=curvature, y=curvature),
    )
    bowl_cap = pf.nodes.geo.fill_curve(
        pf.nodes.geo.join_geometry([drain_circle, bowl_outline])
    )
    bowl_cap = pf.nodes.geo.transform(
        geometry=bowl_cap,
        translation=pf.nodes.math.combine_xyz(z=lower_height),
    )
    bowl_lip = pf.nodes.geo.extrude_mesh(
        mesh=bowl_cap,
        offset_scale=-0.01,
        individual=False,
    )

    geometry = pf.nodes.geo.join_geometry(
        [rim, bowl, drain_wall, drain_ring.mesh, bowl_cap, bowl_lip.mesh]
    )
    geometry = pf.nodes.geo.set_material(geometry=geometry, material=material)
    geometry_offset = pf.nodes.math.combine_xyz((water_tap_margin + margin) / 2.56)
    geometry = pf.nodes.geo.set_position(
        geometry=geometry,
        offset=geometry_offset,
    )

    cutter_curve = pf.nodes.geo.fillet_curve_poly(
        curve=inner_curve,
        radius=fillet_radius,
        count=3,
    )
    cutter_curve = pf.nodes.geo.transform(
        geometry=cutter_curve,
        scale=(1.01, 1.01, 1.0),
    )
    cutter = pf.nodes.geo.fill_curve(curve=cutter_curve, mode="NGONS")
    cutter = pf.nodes.geo.extrude_mesh(
        mesh=cutter,
        offset_scale=lower_height + upper_height + 0.05,
    )
    cutter = pf.nodes.geo.set_position(
        geometry=cutter.mesh,
        offset=geometry_offset,
    )
    return _SinkGeometryResult(geometry=geometry, cutter=cutter)


def _finish(result: _SinkGeometryResult) -> SinkResult:
    bounds = pf.nodes.geo.bound_box(result.geometry)
    ground = pf.nodes.math.combine_xyz(z=bounds.min.z * -1.0)
    geometry = pf.nodes.geo.transform(
        geometry=result.geometry,
        translation=ground,
    )
    cutter = pf.nodes.geo.transform(
        geometry=result.cutter,
        translation=ground,
    )
    obj = pf.nodes.to_mesh_object(pf.nodes.geo.realize_instances(geometry))
    cutter_obj = pf.nodes.to_mesh_object(pf.nodes.geo.realize_instances(cutter))
    pf.ops.uv.smart_project(obj, uv_name="UVMap", island_margin=0.01)
    cutter_obj.item().hide_viewport = True
    cutter_obj.item().hide_render = True
    return SinkResult(mesh=obj, cutter=cutter_obj)


@pf.tracer.generator
def sink(
    material: pf.Material | None = None,
    width: float = 0.6,
    depth: float = 0.45,
    curvature: float = 1.0,
    upper_height: float = 0.3,
    lower_height: float = 0.005,
    hole_radius: float = 0.035,
    margin: float = 0.035,
    water_tap_margin: float = 0.11,
) -> SinkResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf(metallic=1.0))
    result = _sink_geometry(
        material=material,
        width=width,
        depth=depth,
        curvature=curvature,
        upper_height=upper_height,
        lower_height=lower_height,
        hole_radius=hole_radius,
        margin=margin,
        water_tap_margin=water_tap_margin,
    )
    return _finish(result)


def sink_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    width: float | None = None,
    depth: float | None = None,
    upper_height: float | None = None,
    lower_height: float | None = None,
    hole_radius: float | None = None,
    margin: float | None = None,
    water_tap_margin: float | None = None,
) -> SinkResult:
    (
        rng_material,
        rng_width,
        rng_depth,
        rng_upper,
        rng_lower,
        rng_hole,
        rng_margin,
        rng_tap_margin,
    ) = rng.spawn(8)
    coord = pf.nodes.shader.coord()
    if material is None:
        material = decorative_material_rand(rng_material, coord.uv)
    if width is None:
        width = pf.random.uniform(rng_width, 0.4, 1.0)
    if depth is None:
        depth = pf.random.uniform(rng_depth, 0.4, 0.5)
    if upper_height is None:
        upper_height = pf.random.uniform(rng_upper, 0.2, 0.4)
    if lower_height is None:
        lower_height = pf.random.uniform(rng_lower, 0.0, 0.01)
    if hole_radius is None:
        hole_radius = pf.random.uniform(rng_hole, 0.02, 0.05)
    if margin is None:
        margin = pf.random.uniform(rng_margin, 0.02, 0.05)
    if water_tap_margin is None:
        water_tap_margin = pf.random.uniform(rng_tap_margin, 0.1, 0.12)
    return sink(
        material=material,
        width=width,
        depth=depth,
        curvature=1.0,
        upper_height=upper_height,
        lower_height=lower_height,
        hole_radius=hole_radius,
        margin=margin,
        water_tap_margin=water_tap_margin,
    )
