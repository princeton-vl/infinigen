# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Anna Calveri: primary author of the original Infinigen v1 lever, rose, and lock system (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/sim_objects/door_handle.py)
# - Max Gonzalez Saez-Diez, Abhishek Joshi: Infinigen v1 simulation updates to the lever, rose, and lock system (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/sim_objects/door_handle.py)
# - Abhishek Joshi: primary author of the original Infinigen v1 drawer and appliance pull-handle nodegroup (https://github.com/princeton-vl/infinigen_internal/blob/c6345652e65b2ee05f76e2403756531976046d90/src/infinigen/assets/sim_objects/drawer.py)
# - Max Gonzalez Saez-Diez: Infinigen v1 simulation updates to the drawer and appliance pull-handle nodegroup (https://github.com/princeton-vl/infinigen_internal/blob/c6345652e65b2ee05f76e2403756531976046d90/src/infinigen/assets/sim_objects/drawer.py)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.shaders.functionality_lists import decorative_material_rand
from infinigen2.util.curve import curve_to_mesh_with_uv
from infinigen2.util.mesh import quad_cap, quad_cylinder

# the knob's rims are its silhouette, so they get a tight radius not a razor edge
_KNOB_RIM_CREASE = 0.5

__all__ = [
    "HandleResult",
    "handle_bar_pull",
    "handle_bar_pull_rand",
    "handle_curved_pull",
    "handle_curved_pull_rand",
    "handle_knob",
    "handle_knob_rand",
    "handle_lever",
    "handle_lever_rand",
    "handle_rand",
]


class HandleResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _rounded_prism(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    count: t.SocketOrVal[int] = 6,
) -> pf.ProcNode[pf.MeshObject]:
    quad = pf.nodes.geo.curve_quadrilateral(width=width, height=height)
    profile = pf.nodes.geo.fillet_curve_poly(
        curve=quad, radius=radius, count=count, limit_radius=True
    )
    line = pf.nodes.geo.curve_line(
        start=(0.0, 0.0, 0.0), end=pf.nodes.math.combine_xyz(z=depth)
    )
    walls = curve_to_mesh_with_uv(curve=line, profile=profile, fill_caps=False).mesh
    cap_start = pf.nodes.geo.flip_faces(quad_cap(profile, insets=3, scale=0.5))
    cap_end = pf.nodes.geo.transform(
        geometry=quad_cap(profile, insets=3, scale=0.5),
        translation=pf.nodes.math.combine_xyz(z=depth),
    )
    solid = pf.nodes.geo.join_geometry([walls, cap_start, cap_end])
    return pf.nodes.geo.merge_by_distance(solid, distance=1e-5)


@pf.nodes.node_function
def _cylinder_with_uv(
    radius: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    vertices: t.SocketOrVal[int] = 16,
) -> pf.ProcNode[pf.MeshObject]:
    cyl = pf.nodes.geo.mesh_cylinder(vertices=vertices, radius=radius, depth=depth)
    # mesh_cylinder packs the wall into v 0.5..1 and each cap into a uv disc of radius 0.225
    wall = pf.nodes.math.combine_xyz(
        x=(cyl.uv_map.y - 0.5) * depth * 2.0,
        y=cyl.uv_map.x * radius * 2.0 * math.pi,
    )
    cap = pf.nodes.math.combine_xyz(
        x=cyl.uv_map.x * radius / 0.225, y=cyl.uv_map.y * radius / 0.225
    )
    is_cap = pf.nodes.math.absolute(pf.nodes.geo.input_normal().z) > 0.5
    uv = pf.nodes.func.switch(
        switch=is_cap, a=wall, b=cap, data_type=NodeDataType.FLOAT_VECTOR
    )
    return pf.nodes.geo.store_named_attribute(
        geometry=cyl.mesh, name="UVMap", value=uv, domain="CORNER", data_type="FLOAT2"
    )


@pf.nodes.node_function
def _uv_sphere_with_uv(
    radius: t.SocketOrVal[float],
    z_scale: t.SocketOrVal[float] = 1.0,
    segments: t.SocketOrVal[int] = 24,
    rings: t.SocketOrVal[int] = 16,
) -> pf.ProcNode[pf.MeshObject]:
    sphere = pf.nodes.geo.mesh_uv_sphere(segments=segments, rings=rings, radius=radius)
    # z_scale shortens the meridian by that factor at the equator and not at all at the poles
    uv = pf.nodes.math.combine_xyz(
        x=sphere.uv_map.y * radius * math.pi * z_scale,
        y=sphere.uv_map.x * radius * 2.0 * math.pi,
    )
    mesh = pf.nodes.geo.store_named_attribute(
        geometry=sphere.mesh,
        name="UVMap",
        value=uv,
        domain="CORNER",
        data_type="FLOAT2",
    )
    return pf.nodes.geo.transform(
        geometry=mesh, scale=pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=z_scale)
    )


@pf.nodes.node_function
def handle_part_rose(
    rose_height: t.SocketOrVal[float] = 0.06,
    rose_radius: t.SocketOrVal[float] = 0.01,
    rose_depth: t.SocketOrVal[float] = 0.01,
) -> pf.ProcNode[pf.MeshObject]:
    prism = _rounded_prism(
        width=rose_height, height=rose_height, depth=rose_depth, radius=rose_radius
    )
    return pf.nodes.geo.transform(geometry=prism, rotation=(0.0, 1.5708, 0.0))


@pf.nodes.node_function
def handle_part_lever(
    stub_height: t.SocketOrVal[float] = 0.012,
    stub_radius: t.SocketOrVal[float] = 0.03,
    stub_depth: t.SocketOrVal[float] = 0.006,
    lever_length: t.SocketOrVal[float] = 0.12,
    lever_width: t.SocketOrVal[float] = 0.006,
) -> pf.ProcNode[pf.MeshObject]:
    outer_sleeve_height = stub_height * 1.5
    outer_sleeve_radius = stub_depth * 1.0

    lever_mesh = _rounded_prism(
        width=outer_sleeve_height,
        height=lever_length,
        depth=lever_width,
        radius=outer_sleeve_radius,
    )
    lever_translation = pf.nodes.math.combine_xyz(
        x=stub_radius,
        y=(lever_length * -0.5) + (outer_sleeve_height * 0.5),
    )
    lever = pf.nodes.geo.transform(
        geometry=lever_mesh,
        translation=lever_translation,
        rotation=(0.0, 1.5708, 0.0),
    )

    sleeve_mesh = _rounded_prism(
        width=outer_sleeve_height,
        height=outer_sleeve_height,
        depth=stub_radius,
        radius=outer_sleeve_radius,
    )
    sleeve = pf.nodes.geo.transform(geometry=sleeve_mesh, rotation=(0.0, 1.5708, 0.0))
    return pf.nodes.geo.join_geometry([lever, sleeve])


@pf.nodes.node_function
def handle_part_lock(
    value: t.SocketOrVal[float] = 0.012,
    turn_lock: t.SocketOrVal[bool] = False,
    button_depth: t.SocketOrVal[float] = 0.007,
    mini_lock_depth: t.SocketOrVal[float] = 0.002,
) -> pf.ProcNode[pf.MeshObject]:
    cylinder = _cylinder_with_uv(radius=value * 0.5, depth=button_depth)
    base = pf.nodes.geo.transform(geometry=cylinder, rotation=(0.0, 1.5708, 0.0))

    mini_mesh = _rounded_prism(
        width=0.002, height=value * 0.8, depth=mini_lock_depth, radius=0.25
    )
    mini = pf.nodes.geo.transform(
        geometry=mini_mesh,
        translation=(0.002, 0.0, 0.0),
        rotation=(0.0, 1.5708, 0.0),
    )
    base_box = pf.nodes.geo.bound_box(base)
    mini_box = pf.nodes.geo.bound_box(mini)
    mini = pf.nodes.geo.transform(
        geometry=mini,
        translation=pf.nodes.math.combine_xyz(base_box.max.x - mini_box.min.x),
    )
    with_mini = pf.nodes.geo.join_geometry([base, mini])
    return pf.nodes.func.switch(
        switch=turn_lock, a=base, b=with_mini, data_type=NodeDataType.GEOMETRY
    )


@pf.nodes.node_function
def _handle_lever_geometry(
    rose_height: t.SocketOrVal[float] = 0.06,
    rose_radius: t.SocketOrVal[float] = 0.01,
    rose_depth: t.SocketOrVal[float] = 0.01,
    stub_height: t.SocketOrVal[float] = 0.012,
    stub_radius: t.SocketOrVal[float] = 0.006,
    stub_depth: t.SocketOrVal[float] = 0.03,
    lever_length: t.SocketOrVal[float] = 0.12,
    lever_width: t.SocketOrVal[float] = 0.006,
    button_depth: t.SocketOrVal[float] = 0.007,
    mini_lock_depth: t.SocketOrVal[float] = 0.002,
    turn_lock: t.SocketOrVal[bool] = True,
    has_lock: t.SocketOrVal[bool] = True,
) -> pf.ProcNode[pf.MeshObject]:
    rose = handle_part_rose(
        rose_height=rose_height, rose_radius=rose_radius, rose_depth=rose_depth
    )
    lever = handle_part_lever(
        stub_height=stub_height,
        stub_radius=stub_depth,
        stub_depth=stub_radius,
        lever_length=lever_length,
        lever_width=lever_width,
    )
    lever = pf.nodes.geo.transform(
        geometry=lever, translation=pf.nodes.math.combine_xyz(x=rose_depth)
    )
    lever_and_rose = pf.nodes.geo.join_geometry([rose, lever])

    lock = handle_part_lock(
        value=stub_height,
        turn_lock=turn_lock,
        button_depth=button_depth,
        mini_lock_depth=mini_lock_depth,
    )
    lock_x = (rose_depth / 2.0) + (button_depth / 2.0) + lever_width + stub_depth
    lock = pf.nodes.geo.transform(
        geometry=lock, translation=pf.nodes.math.combine_xyz(lock_x)
    )
    with_lock = pf.nodes.geo.join_geometry([lever_and_rose, lock])

    geo = pf.nodes.func.switch(
        switch=has_lock,
        a=lever_and_rose,
        b=with_lock,
        data_type=NodeDataType.GEOMETRY,
    )
    box = pf.nodes.geo.bound_box(geo)
    return pf.nodes.geo.transform(
        geometry=geo,
        translation=pf.nodes.math.combine_xyz(z=(box.min.z + box.max.z) * -0.5),
    )


@pf.nodes.node_function
def _handle_knob_geometry(
    base_radius: t.SocketOrVal[float] = 0.016,
    base_depth: t.SocketOrVal[float] = 0.004,
    stem_radius: t.SocketOrVal[float] = 0.005,
    stem_length: t.SocketOrVal[float] = 0.014,
    head_radius: t.SocketOrVal[float] = 0.013,
) -> pf.ProcNode[pf.MeshObject]:
    base = quad_cylinder(radius=base_radius, depth=base_depth, resolution=12)
    base = pf.nodes.geo.transform(
        geometry=base, translation=pf.nodes.math.combine_xyz(z=base_depth * 0.5)
    )
    stem_depth = stem_length + base_depth * 0.5
    stem = quad_cylinder(radius=stem_radius, depth=stem_depth, resolution=8)
    stem = pf.nodes.geo.transform(
        geometry=stem,
        translation=pf.nodes.math.combine_xyz(
            z=base_depth + stem_length - stem_depth * 0.5
        ),
    )
    head = _uv_sphere_with_uv(radius=head_radius, z_scale=0.8, segments=12, rings=8)
    head = pf.nodes.geo.transform(
        geometry=head,
        translation=pf.nodes.math.combine_xyz(z=base_depth + stem_length),
    )
    geo = pf.nodes.geo.join_geometry([base, stem, head])
    return pf.nodes.geo.transform(geometry=geo, rotation=(0.0, 1.5708, 0.0))


@pf.nodes.node_function
def _standoff_post(
    length: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    z: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    post = _cylinder_with_uv(radius=radius, depth=length)
    return pf.nodes.geo.transform(
        geometry=post,
        rotation=(-1.5708, 0.0, 0.0),
        translation=pf.nodes.math.combine_xyz(y=length * 0.5, z=z),
    )


@pf.nodes.node_function
def _handle_bar_pull_geometry(
    grip_length: t.SocketOrVal[float] = 0.16,
    grip_radius: t.SocketOrVal[float] = 0.008,
    standoff_length: t.SocketOrVal[float] = 0.035,
    standoff_radius: t.SocketOrVal[float] = 0.006,
) -> pf.ProcNode[pf.MeshObject]:
    grip = _cylinder_with_uv(radius=grip_radius, depth=grip_length)
    grip = pf.nodes.geo.transform(
        geometry=grip, translation=pf.nodes.math.combine_xyz(y=standoff_length)
    )
    post_z = grip_length * 0.5 - grip_radius
    top = _standoff_post(standoff_length, standoff_radius, post_z)
    bottom = _standoff_post(standoff_length, standoff_radius, -post_z)
    geo = pf.nodes.geo.join_geometry([grip, top, bottom])
    return pf.nodes.geo.transform(geometry=geo, rotation=(0.0, 0.0, -1.5708))


@pf.nodes.node_function
def _handle_curved_pull_geometry(
    arc_radius: t.SocketOrVal[float] = 0.05,
    profile_radius: t.SocketOrVal[float] = 0.01,
    span_scale: t.SocketOrVal[float] = 2.0,
    depth_scale: t.SocketOrVal[float] = 1.5,
) -> pf.ProcNode[pf.MeshObject]:
    arc = pf.nodes.geo.curve_arc(radius=arc_radius, sweep_angle=3.1416, resolution=24)
    profile = pf.nodes.geo.curve_circle(radius=profile_radius, resolution=16)
    tube = curve_to_mesh_with_uv(curve=arc, profile=profile, fill_caps=True).mesh
    scale = pf.nodes.math.combine_xyz(x=span_scale, y=depth_scale, z=1.0)
    tube = pf.nodes.geo.transform(
        geometry=tube, scale=scale, rotation=(0.0, 0.0, -1.5708)
    )
    tube = pf.nodes.geo.transform(geometry=tube, rotation=(1.5708, 0.0, 0.0))
    box = pf.nodes.geo.bound_box(tube)
    return pf.nodes.geo.transform(
        geometry=tube,
        translation=pf.nodes.math.combine_xyz(z=(box.min.z + box.max.z) * -0.5),
    )


def _finish(
    geo: pf.ProcNode, material: pf.Material, crease_value: float = 1.0
) -> HandleResult:
    geo = pf.nodes.geo.set_material(geo, material=material)
    geo = pf.nodes.geo.set_shade_smooth(geometry=geo, shade_smooth=True)
    # round sections tessellate at <=30 deg; genuine shoulders are cap rims at ~90
    sharp = pf.nodes.geo.input_mesh_edge_angle().unsigned_angle > math.radians(60.0)
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo,
        name="crease_edge",
        domain="EDGE",
        value=sharp.astype(dtype=float) * crease_value,
        data_type="FLOAT",
    )
    obj = pf.nodes.to_mesh_object(geo)
    return HandleResult(mesh=obj)


def _subdivided(obj: pf.MeshObject) -> HandleResult:
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return HandleResult(mesh=obj)


def _default_material(material: pf.Material | None) -> pf.Material:
    if material is None:
        return pf.Material(surface=pf.nodes.shader.principled_bsdf())
    return material


@pf.tracer.generator
def handle_lever(
    material: pf.Material | None = None,
    rose_height: float = 0.06,
    rose_radius: float = 0.01,
    rose_depth: float = 0.01,
    stub_height: float = 0.012,
    stub_radius: float = 0.006,
    stub_depth: float = 0.03,
    lever_length: float = 0.12,
    lever_width: float = 0.006,
    button_depth: float = 0.007,
    mini_lock_depth: float = 0.002,
    turn_lock: bool = True,
    has_lock: bool = True,
) -> HandleResult:
    geo = _handle_lever_geometry(
        rose_height=rose_height,
        rose_radius=rose_radius,
        rose_depth=rose_depth,
        stub_height=stub_height,
        stub_radius=stub_radius,
        stub_depth=stub_depth,
        lever_length=lever_length,
        lever_width=lever_width,
        button_depth=button_depth,
        mini_lock_depth=mini_lock_depth,
        turn_lock=turn_lock,
        has_lock=has_lock,
    )
    result = _finish(geo, _default_material(material))
    result.mesh.item().name = handle_lever.__name__
    return result


def handle_lever_rand(rng: pf.RNG, material: pf.Material | None = None) -> HandleResult:
    rng, rng_turn_lock, rng_has_lock, rng_mat = rng.spawn(4)
    rose_height = pf.random.uniform(rng, 0.05, 0.1)
    rose_depth = pf.random.uniform(rng, 0.008, 0.02)
    stub_height = pf.random.uniform(rng, 0.008, 0.016)
    stub_radius = pf.random.uniform(rng, 0.0, 0.01)
    rose_radius = pf.random.uniform(rng, 0.0, rose_height / 2)
    stub_depth = pf.random.uniform(rng, 0.02, 0.05)
    lever_length = pf.random.uniform(rng, 0.08, 0.16)
    lever_width = pf.random.uniform(rng, 0.01, 0.04)
    button_depth = pf.random.uniform(rng, 0.005, 0.01)
    mini_lock_depth = pf.random.uniform(rng, 0.005, 0.01)
    turn_lock = pf.control.choice(rng_turn_lock, [(True, 1.0), (False, 1.0)])
    has_lock = pf.control.choice(rng_has_lock, [(True, 1.0), (False, 1.0)])
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    obj = handle_lever(
        material=material,
        rose_height=rose_height,
        rose_radius=rose_radius,
        rose_depth=rose_depth,
        stub_height=stub_height,
        stub_radius=stub_radius,
        stub_depth=stub_depth,
        lever_length=lever_length,
        lever_width=lever_width,
        button_depth=button_depth,
        mini_lock_depth=mini_lock_depth,
        turn_lock=turn_lock,
        has_lock=has_lock,
    ).mesh
    return _subdivided(obj)


@pf.tracer.generator
def handle_knob(
    material: pf.Material | None = None,
    base_radius: float = 0.016,
    base_depth: float = 0.004,
    stem_radius: float = 0.005,
    stem_length: float = 0.014,
    head_radius: float = 0.013,
) -> HandleResult:
    geo = _handle_knob_geometry(
        base_radius=base_radius,
        base_depth=base_depth,
        stem_radius=stem_radius,
        stem_length=stem_length,
        head_radius=head_radius,
    )
    result = _finish(geo, _default_material(material), crease_value=_KNOB_RIM_CREASE)
    result.mesh.item().name = handle_knob.__name__
    return result


def handle_knob_rand(rng: pf.RNG, material: pf.Material | None = None) -> HandleResult:
    rng, rng_mat = rng.spawn(2)
    stem_radius = pf.random.uniform(rng, 0.004, 0.007)
    head_radius = pf.random.uniform(rng, stem_radius * 1.8, 0.018)
    base_radius = pf.random.uniform(rng, 0.012, 0.02)
    base_depth = pf.random.uniform(rng, 0.003, 0.006)
    stem_length = pf.random.uniform(rng, head_radius, 0.022)
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    obj = handle_knob(
        material=material,
        base_radius=base_radius,
        base_depth=base_depth,
        stem_radius=stem_radius,
        stem_length=stem_length,
        head_radius=head_radius,
    ).mesh
    return _subdivided(obj)


@pf.tracer.generator
def handle_curved_pull(
    material: pf.Material | None = None,
    arc_radius: float = 0.05,
    profile_radius: float = 0.01,
    span_scale: float = 2.0,
    depth_scale: float = 1.5,
) -> HandleResult:
    geo = _handle_curved_pull_geometry(
        arc_radius=arc_radius,
        profile_radius=profile_radius,
        span_scale=span_scale,
        depth_scale=depth_scale,
    )
    result = _finish(geo, _default_material(material))
    result.mesh.item().name = handle_curved_pull.__name__
    return result


def handle_curved_pull_rand(
    rng: pf.RNG, material: pf.Material | None = None
) -> HandleResult:
    rng, rng_mat = rng.spawn(2)
    arc_radius = pf.random.uniform(rng, 0.035, 0.06)
    profile_radius = pf.random.uniform(rng, 0.006, 0.012)
    span_scale = pf.random.uniform(rng, 1.4, 2.4)
    depth_scale = pf.random.uniform(rng, 1.0, 1.8)
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    obj = handle_curved_pull(
        material=material,
        arc_radius=arc_radius,
        profile_radius=profile_radius,
        span_scale=span_scale,
        depth_scale=depth_scale,
    ).mesh
    return _subdivided(obj)


@pf.tracer.generator
def handle_bar_pull(
    material: pf.Material | None = None,
    grip_length: float = 0.16,
    grip_radius: float = 0.008,
    standoff_length: float = 0.035,
    standoff_radius: float = 0.006,
) -> HandleResult:
    geo = _handle_bar_pull_geometry(
        grip_length=grip_length,
        grip_radius=grip_radius,
        standoff_length=standoff_length,
        standoff_radius=standoff_radius,
    )
    result = _finish(geo, _default_material(material))
    result.mesh.item().name = handle_bar_pull.__name__
    return result


def handle_bar_pull_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    grip_length: float | None = None,
) -> HandleResult:
    rng, rng_mat = rng.spawn(2)
    if grip_length is None:
        grip_length = pf.random.uniform(rng, 0.1, 0.25)
    grip_radius = pf.random.uniform(rng, 0.006, 0.012)
    standoff_length = pf.random.uniform(rng, 0.025, 0.045)
    standoff_radius = pf.random.uniform(rng, 0.004, 0.008)
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    obj = handle_bar_pull(
        material=material,
        grip_length=grip_length,
        grip_radius=grip_radius,
        standoff_length=standoff_length,
        standoff_radius=standoff_radius,
    ).mesh
    return _subdivided(obj)


def handle_rand(rng: pf.RNG, material: pf.Material | None = None) -> HandleResult:
    rng_choice, rng_func = rng.spawn(2)
    handle_func = pf.control.choice(
        rng_choice,
        [
            (handle_lever_rand, 2.0),
            (handle_bar_pull_rand, 1.5),
            (handle_curved_pull_rand, 1.5),
            (handle_knob_rand, 1.0),
        ],
    )
    return handle_func(rng_func, material)
