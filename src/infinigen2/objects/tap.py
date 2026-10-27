# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Stamatis Alexandropoulos: original Infinigen tap (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/table_decorations/sink.py)
# - Alexander Raistrick: refactor for Infinigen2

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.shaders.functionality_lists import decorative_material_rand
from infinigen2.util import curve, mesh

__all__ = ["TapResult", "tap_arc", "tap_curved", "tap_rand", "tap_vessel"]


class TapResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _tube(
    path: t.SocketOrVal[pf.CurveObject],
    radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    profile = pf.nodes.geo.curve_circle(resolution=16, radius=radius)
    mesh_result = curve.curve_to_mesh_with_uv(path, profile, fill_caps=True)
    return mesh_result.mesh


@pf.nodes.node_function
def _tap_base(
    width: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    outline = pf.nodes.geo.curve_quadrilateral(width=width, height=length)
    outline = pf.nodes.geo.fillet_curve_poly(outline, radius=radius, count=6)
    face = pf.nodes.geo.fill_curve(outline)
    base = pf.nodes.geo.extrude_mesh(face, offset_scale=height).mesh
    return mesh.metric_box_uv(base)


@pf.nodes.node_function
def _mount(
    base_height: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    height = radius * 5.0
    mount = mesh.quad_cylinder(radius=radius * 1.65, depth=height, resolution=24)
    mount = pf.nodes.geo.transform(
        mount,
        translation=pf.nodes.math.combine_xyz(z=base_height + height * 0.5),
    )
    return mesh.metric_box_uv(mount)


@pf.nodes.node_function
def _arc_spout(
    base_height: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    reach: t.SocketOrVal[float],
    drop: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    rotation_z: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    rise = height - base_height
    path = pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(z=base_height),
        start_handle=pf.nodes.math.combine_xyz(z=base_height + rise * 0.75),
        end_handle=pf.nodes.math.combine_xyz(x=reach, z=height),
        end=pf.nodes.math.combine_xyz(x=reach, z=height - drop),
        resolution=32,
    )
    spout = _tube(path, radius)
    rotation = pf.nodes.math.combine_xyz(z=rotation_z).astype(dtype=pf.Euler)
    return pf.nodes.geo.transform(spout, rotation=rotation)


@pf.nodes.node_function
def _curved_spout(
    base_height: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    reach: t.SocketOrVal[float],
    drop: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    rotation_z: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    rise = height - base_height
    path = pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(z=base_height),
        start_handle=pf.nodes.math.combine_xyz(
            x=reach * 0.2,
            z=base_height + rise * 0.45,
        ),
        end_handle=pf.nodes.math.combine_xyz(x=reach * 1.1, z=height),
        end=pf.nodes.math.combine_xyz(x=reach, z=height - drop),
        resolution=32,
    )
    spout = _tube(path, radius)
    rotation = pf.nodes.math.combine_xyz(z=rotation_z).astype(dtype=pf.Euler)
    return pf.nodes.geo.transform(spout, rotation=rotation)


@pf.nodes.node_function
def _lever_handle() -> pf.ProcNode[pf.MeshObject]:
    path = pf.nodes.geo.curve_bezier_segment(
        start=(0.0, 0.0, 0.0),
        start_handle=(0.0, 0.0, 0.28),
        end_handle=(0.08, 0.0, 0.28),
        end=(0.4, 0.0, 0.36),
    )
    spline_parameter = pf.nodes.geo.spline_parameter()
    radius = pf.nodes.math.float_curve(
        factor=1.0,
        value=spline_parameter.factor,
        curve=np.array([[0.0, 0.975], [1.0, 0.1625]], dtype=np.float64),
    )
    path = pf.nodes.geo.set_curve_radius(path, radius=radius * 1.3)
    profile = pf.nodes.geo.curve_circle(resolution=16, radius=0.08)
    handle = curve.curve_to_mesh_with_uv(path, profile, fill_caps=True).mesh
    position = pf.nodes.geo.input_position()
    y_scale = pf.nodes.math.map_range(
        value=position.x,
        from_min=0.08,
        from_max=0.4,
        to_max=2.5,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    position = pf.nodes.math.combine_xyz(
        x=position.x,
        y=position.y * y_scale,
        z=position.z,
    )
    handle = pf.nodes.geo.set_position(handle, position=position)
    handle = pf.nodes.geo.subdivision_surface(handle)
    return pf.nodes.geo.set_shade_smooth(handle)


@pf.nodes.node_function
def _lever_handles(
    _hands_length_x: t.SocketOrVal[float],
    _hands_length_y: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    lever = _lever_handle()
    left = pf.nodes.geo.transform(
        lever,
        translation=(0.0, 0.08, 0.0),
        rotation=(0.0, 0.0, 2.618),
        scale=(0.3, 0.3, 0.3),
    )
    right = pf.nodes.geo.transform(
        lever,
        translation=(0.0, -0.08, 0.0),
        rotation=(0.0, 0.0, 3.6652),
        scale=(0.3, 0.3, 0.3),
    )
    handles = pf.nodes.geo.join_geometry([left, right])
    return pf.nodes.geo.transform(handles, rotation=(0.0, 0.0, 3.14159265))


@pf.nodes.node_function
def _bar_handles(
    hands_length_x: t.SocketOrVal[float],
    hands_length_y: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    thin_handle = pf.nodes.geo.mesh_cylinder(
        vertices=8,
        side_segments=1,
        radius=0.002,
        depth=0.04,
    )
    thin_handle = mesh.metric_box_uv(thin_handle.mesh)
    thin_right = pf.nodes.geo.transform(
        thin_handle,
        translation=(0.0, -0.032, 0.06),
        rotation=(0.0, 0.0, 0.0855),
        scale=(1.0, 1.0, 1.1),
    )
    thin_left = pf.nodes.geo.transform(
        thin_handle,
        translation=(0.0, 0.032, 0.06),
        scale=(1.0, 1.0, 1.1),
    )
    thick_handle = pf.nodes.geo.mesh_cylinder(
        vertices=16,
        side_segments=1,
        radius=0.012,
        depth=0.04,
    )
    thick_handle = mesh.metric_box_uv(thick_handle.mesh)
    thick_right = pf.nodes.geo.transform(
        thick_handle,
        translation=(0.0, -0.02, 0.04),
        rotation=(1.5708, 0.0, 0.0),
    )
    thick_left = pf.nodes.geo.transform(
        thick_handle,
        translation=(0.0, 0.02, 0.04),
        rotation=(1.5708, 0.0, 0.0),
    )
    handles = pf.nodes.geo.join_geometry(
        [thin_right, thin_left, thick_right, thick_left]
    )
    handles = pf.nodes.geo.transform(
        handles,
        scale=pf.nodes.math.combine_xyz(
            x=hands_length_x,
            y=hands_length_y,
            z=1.0,
        ),
    )
    return pf.nodes.geo.transform(handles, rotation=(0.0, 0.0, 3.14159265))


@pf.nodes.node_function
def _tap_geometry(
    material: t.SocketOrVal[pf.Material],
    base_width: t.SocketOrVal[float],
    base_length: t.SocketOrVal[float],
    base_height: t.SocketOrVal[float],
    base_radius: t.SocketOrVal[float],
    spout_radius: t.SocketOrVal[float],
    handle_geometry: t.SocketOrVal[pf.MeshObject],
    spout: t.SocketOrVal[pf.MeshObject],
) -> pf.ProcNode[pf.MeshObject]:
    base = _tap_base(base_width, base_length, base_height, base_radius)
    mount = _mount(base_height, spout_radius)
    body = pf.nodes.geo.join_geometry([base, mount, handle_geometry, spout])
    return pf.nodes.geo.set_material(body, material)


@pf.nodes.node_function
def _vessel_spout(
    base_height: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    reach: t.SocketOrVal[float],
    drop: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    rotation_z: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    rise = height - base_height
    path = pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(z=base_height),
        start_handle=pf.nodes.math.combine_xyz(
            x=reach * 0.35,
            z=base_height + rise * 0.35,
        ),
        end_handle=pf.nodes.math.combine_xyz(
            x=reach * 0.85,
            z=height - drop * 0.25,
        ),
        end=pf.nodes.math.combine_xyz(x=reach, z=height - drop),
        resolution=24,
    )
    spout = _tube(path, radius)
    rotation = pf.nodes.math.combine_xyz(z=rotation_z).astype(dtype=pf.Euler)
    return pf.nodes.geo.transform(spout, rotation=rotation)


def _finish(
    material: pf.Material,
    base_width: float,
    base_length: float,
    base_height: float,
    base_radius: float,
    spout_radius: float,
    spout: pf.ProcNode[pf.MeshObject],
    hands_length_x: float,
    hands_length_y: float,
    bar_handles: bool,
) -> TapResult:
    handle_func = _bar_handles if bar_handles else _lever_handles
    handles = handle_func(hands_length_x, hands_length_y)
    geometry = _tap_geometry(
        material,
        base_width,
        base_length,
        base_height,
        base_radius,
        spout_radius,
        handles,
        spout,
    )
    geometry = mesh.metric_box_uv(geometry)
    obj = pf.nodes.to_mesh_object(pf.nodes.geo.realize_instances(geometry))
    return TapResult(mesh=obj)


@pf.tracer.generator
def tap_arc(
    material: pf.Material,
    base_width: float,
    base_length: float,
    base_height: float,
    base_radius: float,
    spout_height: float,
    spout_reach: float,
    spout_drop: float,
    spout_radius: float,
    rotation_z: float,
    hands_length_x: float,
    hands_length_y: float,
    bar_handles: bool,
) -> TapResult:
    spout = _arc_spout(
        base_height, spout_height, spout_reach, spout_drop, spout_radius, rotation_z
    )
    return _finish(
        material,
        base_width,
        base_length,
        base_height,
        base_radius,
        spout_radius,
        spout,
        hands_length_x,
        hands_length_y,
        bar_handles,
    )


@pf.tracer.generator
def tap_curved(
    material: pf.Material,
    base_width: float,
    base_length: float,
    base_height: float,
    base_radius: float,
    spout_height: float,
    spout_reach: float,
    spout_drop: float,
    spout_radius: float,
    rotation_z: float,
    hands_length_x: float,
    hands_length_y: float,
    bar_handles: bool,
) -> TapResult:
    spout = _curved_spout(
        base_height, spout_height, spout_reach, spout_drop, spout_radius, rotation_z
    )
    return _finish(
        material,
        base_width,
        base_length,
        base_height,
        base_radius,
        spout_radius,
        spout,
        hands_length_x,
        hands_length_y,
        bar_handles,
    )


@pf.tracer.generator
def tap_vessel(
    material: pf.Material,
    base_width: float,
    base_length: float,
    base_height: float,
    base_radius: float,
    spout_height: float,
    spout_reach: float,
    spout_drop: float,
    spout_radius: float,
    rotation_z: float,
    hands_length_x: float,
    hands_length_y: float,
    bar_handles: bool,
) -> TapResult:
    spout = _vessel_spout(
        base_height, spout_height, spout_reach, spout_drop, spout_radius, rotation_z
    )
    return _finish(
        material,
        base_width,
        base_length,
        base_height,
        base_radius,
        spout_radius,
        spout,
        hands_length_x,
        hands_length_y,
        bar_handles,
    )


def _tap_material_rand(
    rng: pf.RNG,
    material: pf.Material | None,
) -> pf.Material:
    if material is not None:
        return material
    coord = pf.nodes.shader.coord()
    return decorative_material_rand(rng, coord.uv)


def _standard_tap_rand(
    rng: pf.RNG,
    material: pf.Material | None,
    base_width: float | None,
    base_length: float | None,
    base_height: float | None,
    base_radius: float | None,
    spout_height: float | None,
    spout_reach: float | None,
    spout_drop: float | None,
    spout_radius: float | None,
    rotation_z: float | None,
    hands_length_x: float | None,
    hands_length_y: float | None,
) -> TapResult:
    rng_spout, rng_handles, rng_dimensions, rng_material = rng.spawn(4)
    tap_func = pf.control.choice(
        rng_spout,
        [(tap_arc, 1.0), (tap_curved, 1.0)],
    )
    bar_handles_fn = pf.control.choice(
        rng_handles,
        [(lambda: False, 4.0), (lambda: True, 1.0)],
    )
    material = _tap_material_rand(rng_material, material)
    if base_width is None:
        base_width = pf.random.uniform(rng_dimensions, 0.08, 0.12)
    if base_length is None:
        base_length = pf.random.uniform(rng_dimensions, 0.24, 0.32)
    if base_height is None:
        base_height = pf.random.uniform(rng_dimensions, 0.015, 0.025)
    if base_radius is None:
        base_radius = pf.random.uniform(rng_dimensions, 0.0, base_width * 0.4)
    if spout_height is None:
        spout_height = pf.random.uniform(rng_dimensions, 0.18, 0.3)
    if spout_reach is None:
        spout_reach = pf.random.uniform(rng_dimensions, 0.07, 0.11)
    if spout_drop is None:
        spout_drop = pf.random.uniform(rng_dimensions, 0.02, 0.06)
    if spout_radius is None:
        spout_radius = pf.random.uniform(rng_dimensions, 0.009, 0.014)
    if rotation_z is None:
        rotation_z = pf.random.uniform(rng_dimensions, -0.08, 0.08)
    if hands_length_x is None:
        hands_length_x = pf.random.uniform(rng_dimensions, 0.75, 1.25)
    if hands_length_y is None:
        hands_length_y = pf.random.uniform(rng_dimensions, 0.95, 1.55)
    bar_handles = bar_handles_fn()
    return tap_func(
        material=material,
        base_width=base_width,
        base_length=base_length,
        base_height=base_height,
        base_radius=base_radius,
        spout_height=spout_height,
        spout_reach=spout_reach,
        spout_drop=spout_drop,
        spout_radius=spout_radius,
        rotation_z=rotation_z,
        hands_length_x=hands_length_x,
        hands_length_y=hands_length_y,
        bar_handles=bar_handles,
    )


def _vessel_tap_rand(
    rng: pf.RNG,
    material: pf.Material | None,
    base_width: float | None,
    base_length: float | None,
    base_height: float | None,
    base_radius: float | None,
    spout_height: float | None,
    spout_reach: float | None,
    spout_drop: float | None,
    spout_radius: float | None,
    rotation_z: float | None,
    hands_length_x: float | None,
    hands_length_y: float | None,
) -> TapResult:
    rng_handles, rng_dimensions, rng_material = rng.spawn(3)
    bar_handles_fn = pf.control.choice(
        rng_handles,
        [(lambda: False, 4.0), (lambda: True, 1.0)],
    )
    material = _tap_material_rand(rng_material, material)
    if base_width is None:
        base_width = pf.random.uniform(rng_dimensions, 0.08, 0.12)
    if base_length is None:
        base_length = pf.random.uniform(rng_dimensions, 0.24, 0.32)
    if base_height is None:
        base_height = pf.random.uniform(rng_dimensions, 0.015, 0.025)
    if base_radius is None:
        base_radius = pf.random.uniform(rng_dimensions, 0.0, base_width * 0.4)
    if spout_height is None:
        spout_height = pf.random.uniform(rng_dimensions, 0.14, 0.22)
    if spout_reach is None:
        spout_reach = pf.random.uniform(rng_dimensions, 0.09, 0.14)
    if spout_drop is None:
        spout_drop = pf.random.uniform(rng_dimensions, 0.0, 0.025)
    if spout_radius is None:
        spout_radius = pf.random.uniform(rng_dimensions, 0.005, 0.009)
    if rotation_z is None:
        rotation_z = pf.random.uniform(rng_dimensions, -0.08, 0.08)
    if hands_length_x is None:
        hands_length_x = pf.random.uniform(rng_dimensions, 0.75, 1.25)
    if hands_length_y is None:
        hands_length_y = pf.random.uniform(rng_dimensions, 0.95, 1.55)
    bar_handles = bar_handles_fn()
    return tap_vessel(
        material=material,
        base_width=base_width,
        base_length=base_length,
        base_height=base_height,
        base_radius=base_radius,
        spout_height=spout_height,
        spout_reach=spout_reach,
        spout_drop=spout_drop,
        spout_radius=spout_radius,
        rotation_z=rotation_z,
        hands_length_x=hands_length_x,
        hands_length_y=hands_length_y,
        bar_handles=bar_handles,
    )


def tap_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    base_width: float | None = None,
    base_length: float | None = None,
    base_height: float | None = None,
    base_radius: float | None = None,
    spout_height: float | None = None,
    spout_reach: float | None = None,
    spout_drop: float | None = None,
    spout_radius: float | None = None,
    rotation_z: float | None = None,
    hands_length_x: float | None = None,
    hands_length_y: float | None = None,
) -> TapResult:
    rng_choice, rng_tap = rng.spawn(2)
    producer = pf.control.choice(
        rng_choice,
        [(_standard_tap_rand, 4.0), (_vessel_tap_rand, 1.0)],
    )
    return producer(
        rng_tap,
        material,
        base_width,
        base_length,
        base_height,
        base_radius,
        spout_height,
        spout_reach,
        spout_drop,
        spout_radius,
        rotation_z,
        hands_length_x,
        hands_length_y,
    )
