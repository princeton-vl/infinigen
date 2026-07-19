# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# The lever handle geometry (rose / lever / lock) is extracted (sim articulation
# stripped) and ported to infinigen2 from infinigen/assets/sim_objects/door_handle.py
# Authors: Anna Calveri (primary), Max Gonzalez Saez-Diez, Abhishek Joshi
# The ball knob and the bar (D) pull are authored directly in infinigen2.

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.shaders.functionality_lists import decorative_material_rand
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "HandleResult",
    "bar_pull_handle",
    "bar_pull_handle_rand",
    "knob_handle",
    "knob_handle_rand",
    "lever_handle",
    "lever_handle_rand",
]


class HandleResult(NamedTuple):
    mesh: pf.MeshObject


def _quad_cap(
    profile: pf.ProcNode, insets: int = 3, scale: float = 0.5
) -> pf.ProcNode[pf.MeshObject]:
    cap = pf.nodes.geo.fill_curve(profile, mode="NGONS")
    for _ in range(insets):
        extruded = pf.nodes.geo.extrude_mesh(
            cap, offset_scale=0.0, individual=False, mode="FACES"
        )
        cap = pf.nodes.geo.scale_elements(
            extruded.mesh, scale=scale, selection=extruded.top
        )
    position = pf.nodes.geo.input_position()
    uv = pf.nodes.math.combine_xyz(x=position.x, y=position.y)
    return pf.nodes.geo.store_named_attribute(
        geometry=cap, name="UVMap", value=uv, domain="CORNER", data_type="FLOAT2"
    )


@pf.nodes.node_function
def _rounded_prism(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    count: t.SocketOrVal[int] = 8,
) -> pf.ProcNode[pf.MeshObject]:
    quad = pf.nodes.geo.curve_quadrilateral(width=width, height=height)
    profile = pf.nodes.geo.fillet_curve_poly(
        curve=quad, radius=radius, count=count, limit_radius=True
    )
    line = pf.nodes.geo.curve_line(
        start=(0.0, 0.0, 0.0), end=pf.nodes.math.combine_xyz(z=depth)
    )
    walls = curve_to_mesh_with_uv(curve=line, profile=profile, fill_caps=False).mesh
    cap_start = pf.nodes.geo.flip_faces(_quad_cap(profile))
    cap_end = pf.nodes.geo.transform(
        geometry=_quad_cap(profile), translation=pf.nodes.math.combine_xyz(z=depth)
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
    uv = pf.nodes.math.combine_xyz(
        x=cyl.uv_map.y * depth, y=cyl.uv_map.x * radius * 6.283185307179586
    )
    return pf.nodes.geo.store_named_attribute(
        geometry=cyl.mesh, name="UVMap", value=uv, domain="CORNER", data_type="FLOAT2"
    )


@pf.nodes.node_function
def handle_rose(
    rose_height: t.SocketOrVal[float] = 0.06,
    rose_radius: t.SocketOrVal[float] = 0.01,
    rose_depth: t.SocketOrVal[float] = 0.01,
) -> pf.ProcNode[pf.MeshObject]:
    prism = _rounded_prism(
        width=rose_height, height=rose_height, depth=rose_depth, radius=rose_radius
    )
    return pf.nodes.geo.transform(geometry=prism, rotation=(0.0, 1.5708, 0.0))


@pf.nodes.node_function
def handle_lever(
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
def handle_lock(
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
def _lever_handle_geometry(
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
    rose = handle_rose(
        rose_height=rose_height, rose_radius=rose_radius, rose_depth=rose_depth
    )
    lever = handle_lever(
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

    lock = handle_lock(
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
def _knob_handle_geometry(
    base_radius: t.SocketOrVal[float] = 0.016,
    base_depth: t.SocketOrVal[float] = 0.004,
    stem_radius: t.SocketOrVal[float] = 0.005,
    stem_length: t.SocketOrVal[float] = 0.014,
    head_radius: t.SocketOrVal[float] = 0.013,
) -> pf.ProcNode[pf.MeshObject]:
    base = pf.nodes.geo.mesh_cylinder(vertices=32, radius=base_radius, depth=base_depth)
    base = pf.nodes.geo.transform(
        geometry=base.mesh, translation=pf.nodes.math.combine_xyz(z=base_depth * 0.5)
    )
    stem = pf.nodes.geo.mesh_cylinder(
        vertices=16, radius=stem_radius, depth=stem_length
    )
    stem = pf.nodes.geo.transform(
        geometry=stem.mesh,
        translation=pf.nodes.math.combine_xyz(z=base_depth + stem_length * 0.5),
    )
    head = pf.nodes.geo.mesh_uv_sphere(segments=24, rings=16, radius=head_radius)
    head = pf.nodes.geo.transform(
        geometry=head.mesh,
        scale=(1.0, 1.0, 0.8),
        translation=pf.nodes.math.combine_xyz(z=base_depth + stem_length),
    )
    geo = pf.nodes.geo.join_geometry([base, stem, head])
    position = pf.nodes.geo.input_position()
    uv = pf.nodes.math.combine_xyz(x=position.x, y=position.y)
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo,
        name="UVMap",
        value=uv,
        domain="CORNER",
        data_type="FLOAT2",
    )
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
def _bar_pull_handle_geometry(
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


def _finish(geo: pf.ProcNode, material: pf.Material) -> HandleResult:
    geo = pf.nodes.geo.set_material(geo, material=material)
    geo = pf.nodes.geo.set_shade_smooth(geometry=geo, shade_smooth=True)
    sharp = pf.nodes.geo.input_mesh_edge_angle().unsigned_angle > 0.5
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo,
        name="crease_edge",
        domain="EDGE",
        value=sharp.astype(dtype=float),
        data_type="FLOAT",
    )
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.subdivide_surface(obj, levels=4, _skip_apply=True)
    return HandleResult(mesh=obj)


def lever_handle(material: pf.Material | None = None) -> HandleResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    geo = _lever_handle_geometry()
    return _finish(geo, material)


def lever_handle_rand(rng: pf.RNG, material: pf.Material | None = None) -> HandleResult:
    rng, rng_mat = rng.spawn(2)
    rose_height = pf.random.uniform(rng, 0.05, 0.1)
    rose_depth = pf.random.uniform(rng, 0.008, 0.02)
    stub_height = pf.random.uniform(rng, 0.008, 0.016)
    stub_radius = pf.random.uniform(rng, 0.0, 0.01)
    geo = _lever_handle_geometry(
        rose_height=rose_height,
        rose_radius=pf.random.uniform(rng, 0.0, rose_height / 2),
        rose_depth=rose_depth,
        stub_height=stub_height,
        stub_radius=stub_radius,
        stub_depth=pf.random.uniform(rng, 0.02, 0.05),
        lever_length=pf.random.uniform(rng, 0.08, 0.16),
        lever_width=pf.random.uniform(rng, 0.01, 0.04),
        button_depth=pf.random.uniform(rng, 0.005, 0.01),
        mini_lock_depth=pf.random.uniform(rng, 0.005, 0.01),
        turn_lock=pf.control.choice(rng, [(True, 1.0), (False, 1.0)]),
        has_lock=pf.control.choice(rng, [(True, 1.0), (False, 1.0)]),
    )
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    return _finish(geo, material)


def knob_handle(material: pf.Material | None = None) -> HandleResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    geo = _knob_handle_geometry()
    return _finish(geo, material)


def knob_handle_rand(rng: pf.RNG, material: pf.Material | None = None) -> HandleResult:
    rng, rng_mat = rng.spawn(2)
    stem_radius = pf.random.uniform(rng, 0.004, 0.007)
    geo = _knob_handle_geometry(
        base_radius=pf.random.uniform(rng, 0.012, 0.02),
        base_depth=pf.random.uniform(rng, 0.003, 0.006),
        stem_radius=stem_radius,
        stem_length=pf.random.uniform(rng, 0.01, 0.022),
        head_radius=pf.random.uniform(rng, stem_radius * 1.8, 0.018),
    )
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    return _finish(geo, material)


def bar_pull_handle(material: pf.Material | None = None) -> HandleResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    geo = _bar_pull_handle_geometry()
    return _finish(geo, material)


def bar_pull_handle_rand(
    rng: pf.RNG, material: pf.Material | None = None
) -> HandleResult:
    rng, rng_mat = rng.spawn(2)
    geo = _bar_pull_handle_geometry(
        grip_length=pf.random.uniform(rng, 0.1, 0.25),
        grip_radius=pf.random.uniform(rng, 0.006, 0.012),
        standoff_length=pf.random.uniform(rng, 0.025, 0.045),
        standoff_radius=pf.random.uniform(rng, 0.004, 0.008),
    )
    if material is None:
        material = decorative_material_rand(rng_mat, pf.nodes.shader.coord().object)
    return _finish(geo, material)
