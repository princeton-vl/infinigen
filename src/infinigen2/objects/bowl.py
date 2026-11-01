# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import bpy
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import vase
from infinigen2.util import mesh as mesh_util

__all__ = ["BowlResult", "PotResult", "bowl", "bowl_rand", "pot"]


class BowlResult(NamedTuple):
    mesh: pf.MeshObject


class PotResult(NamedTuple):
    mesh: pf.MeshObject
    inside_volume: pf.MeshObject


class _PotGeometryResult(NamedTuple):
    mesh: t.ProcNode[pf.MeshObject]
    inside_volume: t.ProcNode[pf.MeshObject]


@pf.tracer.primitive
def _discard_inside_volume(
    mesh: pf.MeshObject, inside_volume: pf.MeshObject
) -> pf.MeshObject:
    bpy.data.objects.remove(inside_volume.item(), do_unlink=True)
    return mesh


@pf.nodes.node_function
def _pot_geometry(
    diameter: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    inside_height: t.SocketOrVal[float],
    base_scale: t.SocketOrVal[float],
    profile_fullness: t.SocketOrVal[float],
    profile_slope: t.SocketOrVal[float],
    rib_count: t.SocketOrVal[int],
    rib_depth: t.SocketOrVal[float],
    rib_sharpness: t.SocketOrVal[float],
    twist: t.SocketOrVal[float],
    u_resolution: t.SocketOrVal[int],
    v_resolution: t.SocketOrVal[int],
) -> _PotGeometryResult:
    radius = diameter * 0.5
    base_radius = radius * base_scale
    start = pf.nodes.math.combine_xyz(x=base_radius)
    middle_radius = base_radius + (radius - base_radius) * profile_fullness
    middle = pf.nodes.math.combine_xyz(x=middle_radius, z=height * profile_slope)
    end = pf.nodes.math.combine_xyz(x=radius, z=height)
    section_resolution = v_resolution.astype(dtype=float) * 4.0
    section = pf.nodes.geo.curve_bezier(
        start=start,
        middle=middle,
        end=end,
        resolution=section_resolution.astype(dtype=int),
    )
    side_segments = v_resolution.astype(dtype=float) - 1.0
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=u_resolution,
        side_segments=side_segments.astype(dtype=int),
        radius=1.0,
        depth=2.0,
    )
    geo = pf.nodes.geo.store_named_attribute(
        geometry=cylinder.mesh,
        name="UVMap",
        value=cylinder.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )
    side = pf.nodes.geo.delete_geometry(geo, selection=cylinder.top, domain="FACE")
    side = pf.nodes.geo.delete_geometry(side, selection=cylinder.bottom, domain="FACE")
    profile = pf.nodes.geo.curve_circle(radius=1.0, resolution=u_resolution)
    cap = mesh_util.quad_cap(profile, insets=3, scale=0.5)
    bottom = pf.nodes.geo.transform(
        geometry=pf.nodes.geo.flip_faces(cap), translation=(0.0, 0.0, -1.0)
    )
    top = pf.nodes.geo.transform(geometry=cap, translation=(0.0, 0.0, 1.0))
    geo = pf.nodes.geo.join_geometry([side, bottom])
    geo = pf.nodes.geo.merge_by_distance(geo, distance=1e-6)
    solid = pf.nodes.geo.join_geometry([side, bottom, top])
    solid = pf.nodes.geo.merge_by_distance(solid, distance=1e-6)
    position = pf.nodes.geo.input_position()
    section_position = pf.nodes.geo.sample_curve(
        curves=section, factor=(position.z + 1.0) * 0.5, data_type="FLOAT"
    ).position
    elevation = section_position.z / height
    angle = pf.nodes.math.atan2(position.y, position.x)
    radial_factor = pf.nodes.math.sqrt(
        position.x * position.x + position.y * position.y
    )
    phase = angle * rib_count.astype(dtype=float)
    phase_cosine = pf.nodes.math.cos(phase)
    smooth_rib = (1.0 + phase_cosine) * 0.5
    sharp_rib = 1.0 - pf.nodes.math.acos(phase_cosine) / 3.141592653589793
    rib = smooth_rib + rib_sharpness * (sharp_rib - smooth_rib)
    fade = elevation * (2.0 - elevation)
    r = section_position.x * (1.0 - rib_depth * fade * rib)
    angle = angle + twist * elevation
    x = r * radial_factor * pf.nodes.math.cos(angle)
    y = r * radial_factor * pf.nodes.math.sin(angle)
    surface_position = pf.nodes.math.combine_xyz(x=x, y=y, z=height * elevation)
    geo = pf.nodes.geo.set_position(geo, position=surface_position)
    inside_min = thickness / height
    inside_max = inside_height / height
    inside = pf.nodes.geo.transform(
        solid,
        translation=pf.nodes.math.combine_xyz(z=inside_min + inside_max - 1.0),
        scale=pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=inside_max - inside_min),
    )
    inside_radius = pf.nodes.math.maximum(r - thickness, 0.0) * radial_factor
    inside_start = pf.nodes.geo.sample_curve(
        curves=section, factor=inside_min, data_type="FLOAT"
    ).position.z
    inside_end = pf.nodes.geo.sample_curve(
        curves=section, factor=inside_max, data_type="FLOAT"
    ).position.z
    inside_elevation = (section_position.z - inside_start) / (inside_end - inside_start)
    inside_z = thickness + (inside_height - thickness) * inside_elevation
    inside_position = pf.nodes.math.combine_xyz(
        x=inside_radius * pf.nodes.math.cos(angle),
        y=inside_radius * pf.nodes.math.sin(angle),
        z=inside_z,
    )
    inside = pf.nodes.geo.set_position(inside, position=inside_position)
    return _PotGeometryResult(geo, inside)


@pf.tracer.generator
def pot(
    diameter: float = 0.24,
    height: float = 0.09,
    base_scale: float = 0.4,
    profile_fullness: float = 0.85,
    profile_slope: float = 0.0,
    rib_count: int = 12,
    rib_depth: float = 0.0,
    rib_sharpness: float = 0.0,
    twist: float = 0.0,
    thickness: float = 0.003,
    inside_height: float | None = None,
    u_resolution: int = 192,
    v_resolution: int = 32,
    material: pf.Material | None = None,
) -> PotResult:
    if inside_height is None:
        inside_height = height
    if material is None:
        surface = pf.nodes.shader.principled_bsdf()
        material = pf.Material(surface=surface)
    geo = _pot_geometry(
        diameter=diameter,
        height=height,
        thickness=thickness,
        inside_height=inside_height,
        base_scale=base_scale,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        rib_sharpness=rib_sharpness,
        twist=twist,
        u_resolution=u_resolution,
        v_resolution=v_resolution,
    )
    mesh = pf.nodes.geo.set_material(geo.mesh, material)
    obj = pf.nodes.to_mesh_object(mesh)
    obj.item().name = pot.__name__
    inside_volume = pf.nodes.to_mesh_object(geo.inside_volume)
    inside_volume.item().name = f"{pot.__name__}_inside_volume"
    mesh_util.metric_cylinder_uv(obj)
    pf.ops.modifier.solidify(obj, thickness=thickness, offset=-1.0)
    pf.ops.object.shade_smooth(obj)
    return PotResult(
        mesh=obj,
        inside_volume=inside_volume,
    )


@pf.tracer.generator
def bowl(
    diameter: float = 0.24,
    height: float = 0.09,
    base_scale: float = 0.4,
    profile_fullness: float = 0.85,
    profile_slope: float = 0.0,
    rib_count: int = 12,
    rib_depth: float = 0.0,
    rib_sharpness: float = 0.0,
    twist: float = 0.0,
    thickness: float = 0.003,
    u_resolution: int = 192,
    v_resolution: int = 32,
    material: pf.Material | None = None,
) -> BowlResult:
    result = pot(
        diameter=diameter,
        height=height,
        base_scale=base_scale,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        rib_sharpness=rib_sharpness,
        twist=twist,
        thickness=thickness,
        u_resolution=u_resolution,
        v_resolution=v_resolution,
        material=material,
    )
    mesh = _discard_inside_volume(result.mesh, result.inside_volume)
    mesh.item().name = bowl.__name__
    return BowlResult(mesh=mesh)


def bowl_rand(rng: pf.RNG) -> BowlResult:
    (
        rng_diameter,
        rng_decorated_depth,
        rng_ribs,
        rng_broad_lobes,
        rng_fine_ribs,
        rng_count,
        rng_sharp,
        rng_twist_angle,
        rng_twist,
        rng_material,
        rng_height,
        rng_base_scale,
        rng_profile_fullness,
        rng_profile_slope,
        rng_thickness,
    ) = rng.spawn(15)
    diameter = pf.random.uniform(rng_diameter, 0.14, 0.34)
    decorated_depth = pf.random.uniform(rng_decorated_depth, 0.025, 0.1)
    rib_depth = pf.control.choice(rng_ribs, [(0.0, 0.5), (decorated_depth, 0.5)])
    broad_lobes = pf.random.randint(rng_broad_lobes, 3, 7)
    fine_ribs = pf.random.randint(rng_fine_ribs, 8, 25)
    rib_count = pf.control.choice(rng_count, [(broad_lobes, 0.4), (fine_ribs, 0.6)])
    rib_sharpness = pf.random.uniform(rng_sharp, 0.0, 1.0)
    twist_angle = pf.random.uniform(rng_twist_angle, -0.8, 0.8)
    twist = pf.control.choice(rng_twist, [(0.0, 0.65), (twist_angle, 0.35)])
    uv = pf.nodes.shader.coord().uv
    material = vase.vase_material_rand(rng_material, uv)
    height = diameter * pf.random.uniform(rng_height, 0.22, 0.48)
    base_scale = pf.random.uniform(rng_base_scale, 0.3, 0.55)
    profile_fullness = pf.random.uniform(rng_profile_fullness, 0.5, 1.0)
    profile_slope = pf.random.uniform(rng_profile_slope, 0.0, 0.16)
    thickness = diameter * pf.random.uniform(rng_thickness, 0.01, 0.018)
    result = bowl(
        diameter=diameter,
        height=height,
        base_scale=base_scale,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        rib_sharpness=rib_sharpness,
        twist=twist,
        thickness=thickness,
        material=material,
    )
    return result
