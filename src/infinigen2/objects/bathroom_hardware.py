# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen bathroom hardware (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/bathroom/hardware.py)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import metal_brushed, metal_hammered
from infinigen2.util import curve, mesh

__all__ = [
    "BathroomHardwareResult",
    "hardware_bar",
    "hardware_bar_rand",
    "hardware_bathroom_rand",
    "hardware_holder",
    "hardware_holder_rand",
    "hardware_hook",
    "hardware_hook_rand",
    "hardware_ring",
    "hardware_ring_rand",
]


class BathroomHardwareResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _prism(
    square: t.SocketOrVal[bool],
    radius: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    cylinder = mesh.quad_cylinder(radius=radius, depth=length, resolution=24)
    side = pf.nodes.math.absolute(pf.nodes.geo.input_normal().z) < 0.5
    cylinder = pf.nodes.geo.set_shade_smooth(cylinder, selection=side)
    size = pf.nodes.math.combine_xyz(radius * 2.0, radius * 2.0, length)
    cube = pf.nodes.geo.mesh_cube(size=size).mesh
    cube = mesh.metric_box_uv(cube)
    return pf.nodes.func.switch(square, cylinder, cube)


@pf.nodes.node_function
def _attachment(
    square: t.SocketOrVal[bool],
    attachment_radius: t.SocketOrVal[float],
    attachment_depth: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    base = _prism(square, attachment_radius, attachment_depth)
    base = pf.nodes.geo.transform(
        base,
        translation=pf.nodes.math.combine_xyz(attachment_depth / 2.0, 0, 0),
        rotation=(0, math.pi / 2, 0),
    )
    rod = _prism(square, radius, depth)
    rod = pf.nodes.geo.transform(
        rod,
        translation=pf.nodes.math.combine_xyz(depth / 2.0, 0, 0),
        rotation=(0, math.pi / 2, 0),
    )
    return pf.nodes.geo.join_geometry([base, rod])


@pf.nodes.node_function
def _rail(
    square: t.SocketOrVal[bool],
    radius: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    center: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    rail = _prism(square, radius * 1.001, length * 1.001)
    return pf.nodes.geo.transform(
        rail,
        translation=pf.nodes.math.combine_xyz(depth, center * 1.001, 0),
        rotation=(math.pi / 2, 0, 0),
    )


@pf.nodes.node_function
def _ring(
    attachment_depth: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    ring_radius: t.SocketOrVal[float],
    ring_minor_scale: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    ring_path = pf.nodes.geo.curve_circle(resolution=48, radius=ring_radius * 1.001)
    ring_profile = pf.nodes.geo.curve_circle(
        resolution=12, radius=radius * ring_minor_scale * 1.001
    )
    ring = curve.curve_to_mesh_with_uv(ring_path, ring_profile).mesh
    ring = pf.nodes.geo.transform(
        ring,
        translation=pf.nodes.math.combine_xyz(
            depth - attachment_depth * 1.001, 0, -ring_radius * 1.001
        ),
        rotation=(0, math.pi / 2, 0),
    )
    return pf.nodes.geo.set_shade_smooth(ring)


def _result(
    material: pf.Material, geometry: pf.ProcNode[pf.MeshObject]
) -> BathroomHardwareResult:
    geometry = pf.nodes.geo.set_material(geometry, material)
    return BathroomHardwareResult(mesh=pf.nodes.to_mesh_object(geometry))


def _hook_geometry(
    square: bool,
    attachment_radius: float,
    attachment_depth: float,
    radius: float,
    depth: float,
    hook_length: float,
) -> pf.ProcNode[pf.MeshObject]:
    attachment = _attachment(square, attachment_radius, attachment_depth, radius, depth)
    hook = _prism(square, radius * 1.001, hook_length * 1.001)
    hook = pf.nodes.geo.transform(
        hook, translation=pf.nodes.math.combine_xyz(depth, 0, 0)
    )
    return pf.nodes.geo.join_geometry([attachment, hook])


def _holder_geometry(
    square: bool,
    attachment_radius: float,
    attachment_depth: float,
    radius: float,
    depth: float,
    holder_length: float,
    extension_length: float,
) -> pf.ProcNode[pf.MeshObject]:
    attachment = _attachment(square, attachment_radius, attachment_depth, radius, depth)
    rail = _rail(
        square,
        radius,
        depth,
        holder_length + extension_length,
        (holder_length - extension_length) / 2.0,
    )
    return pf.nodes.geo.join_geometry([attachment, rail])


def _bar_geometry(
    square: bool,
    attachment_radius: float,
    attachment_depth: float,
    radius: float,
    depth: float,
    bar_length: float,
    extension_length: float,
) -> pf.ProcNode[pf.MeshObject]:
    attachment = _attachment(square, attachment_radius, attachment_depth, radius, depth)
    rail = _rail(
        square,
        radius,
        depth,
        bar_length + extension_length * 2.0,
        bar_length / 2.0,
    )
    second = pf.nodes.geo.transform(
        attachment, translation=pf.nodes.math.combine_xyz(0, bar_length, 0)
    )
    return pf.nodes.geo.join_geometry([attachment, rail, second])


def _ring_geometry(
    square: bool,
    attachment_radius: float,
    attachment_depth: float,
    radius: float,
    depth: float,
    ring_radius: float,
    ring_minor_scale: float,
) -> pf.ProcNode[pf.MeshObject]:
    attachment = _attachment(square, attachment_radius, attachment_depth, radius, depth)
    ring = _ring(attachment_depth, radius, depth, ring_radius, ring_minor_scale)
    return pf.nodes.geo.join_geometry([attachment, ring])


@pf.tracer.generator
def hardware_hook(
    material: pf.Material,
    square: bool = False,
    attachment_radius: float = 0.025,
    attachment_depth: float = 0.0125,
    radius: float = 0.0125,
    depth: float = 0.08,
    hook_length: float = 0.075,
) -> BathroomHardwareResult:
    geometry = _hook_geometry(
        square, attachment_radius, attachment_depth, radius, depth, hook_length
    )
    result = _result(material, geometry)
    result.mesh.item().name = hardware_hook.__name__
    return result


@pf.tracer.generator
def hardware_holder(
    material: pf.Material,
    square: bool = False,
    attachment_radius: float = 0.025,
    attachment_depth: float = 0.0125,
    radius: float = 0.0125,
    depth: float = 0.08,
    holder_length: float = 0.2,
    extension_length: float = 0.06,
) -> BathroomHardwareResult:
    geometry = _holder_geometry(
        square,
        attachment_radius,
        attachment_depth,
        radius,
        depth,
        holder_length,
        extension_length,
    )
    result = _result(material, geometry)
    result.mesh.item().name = hardware_holder.__name__
    return result


@pf.tracer.generator
def hardware_bar(
    material: pf.Material,
    square: bool = False,
    attachment_radius: float = 0.025,
    attachment_depth: float = 0.0125,
    radius: float = 0.0125,
    depth: float = 0.08,
    bar_length: float = 0.6,
    extension_length: float = 0.06,
) -> BathroomHardwareResult:
    geometry = _bar_geometry(
        square,
        attachment_radius,
        attachment_depth,
        radius,
        depth,
        bar_length,
        extension_length,
    )
    result = _result(material, geometry)
    result.mesh.item().name = hardware_bar.__name__
    return result


@pf.tracer.generator
def hardware_ring(
    material: pf.Material,
    square: bool = False,
    attachment_radius: float = 0.025,
    attachment_depth: float = 0.0125,
    radius: float = 0.0125,
    depth: float = 0.08,
    ring_radius: float = 0.1,
    ring_minor_scale: float = 0.55,
) -> BathroomHardwareResult:
    geometry = _ring_geometry(
        square,
        attachment_radius,
        attachment_depth,
        radius,
        depth,
        ring_radius,
        ring_minor_scale,
    )
    result = _result(material, geometry)
    result.mesh.item().name = hardware_ring.__name__
    return result


def _hardware_material_rand(rng: pf.RNG) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_rand = pf.control.choice(
        rng_choice,
        [
            (metal_brushed.metal_brushed_linear_rand, 2.0),
            (metal_brushed.metal_brushed_radial_rand, 2.0),
            (metal_hammered.metal_hammered_rand, 1.0),
        ],
    )
    coord = pf.nodes.shader.coord()
    return material_rand(rng_material, coord.object)


def hardware_hook_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    attachment_radius: float | None = None,
    attachment_depth: float | None = None,
    radius: float | None = None,
    depth: float | None = None,
    square: bool | None = None,
    hook_length: float | None = None,
) -> BathroomHardwareResult:
    rng, rng_material, rng_square = rng.spawn(3)
    if material is None:
        material = _hardware_material_rand(rng_material)
    if square is None:
        square = pf.control.choice(rng_square, [(False, 1.0), (True, 1.0)])
    if attachment_radius is None:
        attachment_radius = pf.random.uniform(rng, 0.02, 0.03)
    if attachment_depth is None:
        attachment_depth = pf.random.uniform(rng, 0.01, 0.015)
    if radius is None:
        radius = pf.random.uniform(rng, 0.01, 0.015)
    if depth is None:
        depth = pf.random.uniform(rng, 0.06, 0.1)
    if hook_length is None:
        hook_length = attachment_radius * pf.random.uniform(rng, 2.0, 4.0)
    return hardware_hook(
        material=material,
        square=square,
        attachment_radius=attachment_radius,
        attachment_depth=attachment_depth,
        radius=radius,
        depth=depth,
        hook_length=hook_length,
    )


def hardware_holder_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    attachment_radius: float | None = None,
    attachment_depth: float | None = None,
    radius: float | None = None,
    depth: float | None = None,
    square: bool | None = None,
    holder_length: float | None = None,
    extension_length: float | None = None,
) -> BathroomHardwareResult:
    rng, rng_material, rng_square = rng.spawn(3)
    if material is None:
        material = _hardware_material_rand(rng_material)
    if square is None:
        square = pf.control.choice(rng_square, [(False, 1.0), (True, 1.0)])
    if attachment_radius is None:
        attachment_radius = pf.random.uniform(rng, 0.02, 0.03)
    if attachment_depth is None:
        attachment_depth = pf.random.uniform(rng, 0.01, 0.015)
    if radius is None:
        radius = pf.random.uniform(rng, 0.01, 0.015)
    if depth is None:
        depth = pf.random.uniform(rng, 0.06, 0.1)
    if holder_length is None:
        holder_length = pf.random.uniform(rng, 0.15, 0.25)
    if extension_length is None:
        extension_length = attachment_radius * pf.random.uniform(rng, 2.0, 3.0)
    return hardware_holder(
        material=material,
        square=square,
        attachment_radius=attachment_radius,
        attachment_depth=attachment_depth,
        radius=radius,
        depth=depth,
        holder_length=holder_length,
        extension_length=extension_length,
    )


def hardware_bar_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    attachment_radius: float | None = None,
    attachment_depth: float | None = None,
    radius: float | None = None,
    depth: float | None = None,
    square: bool | None = None,
    bar_length: float | None = None,
    extension_length: float | None = None,
) -> BathroomHardwareResult:
    rng, rng_material, rng_square = rng.spawn(3)
    if material is None:
        material = _hardware_material_rand(rng_material)
    if square is None:
        square = pf.control.choice(rng_square, [(False, 1.0), (True, 1.0)])
    if attachment_radius is None:
        attachment_radius = pf.random.uniform(rng, 0.02, 0.03)
    if attachment_depth is None:
        attachment_depth = pf.random.uniform(rng, 0.01, 0.015)
    if radius is None:
        radius = pf.random.uniform(rng, 0.01, 0.015)
    if depth is None:
        depth = pf.random.uniform(rng, 0.06, 0.1)
    if bar_length is None:
        bar_length = pf.random.uniform(rng, 0.4, 0.8)
    if extension_length is None:
        extension_length = attachment_radius * pf.random.uniform(rng, 2.0, 3.0)
    return hardware_bar(
        material=material,
        square=square,
        attachment_radius=attachment_radius,
        attachment_depth=attachment_depth,
        radius=radius,
        depth=depth,
        bar_length=bar_length,
        extension_length=extension_length,
    )


def hardware_ring_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    attachment_radius: float | None = None,
    attachment_depth: float | None = None,
    radius: float | None = None,
    depth: float | None = None,
    square: bool | None = None,
    ring_radius: float | None = None,
    ring_minor_scale: float | None = None,
) -> BathroomHardwareResult:
    rng, rng_material, rng_square = rng.spawn(3)
    if material is None:
        material = _hardware_material_rand(rng_material)
    if square is None:
        square = pf.control.choice(rng_square, [(False, 1.0), (True, 1.0)])
    if attachment_radius is None:
        attachment_radius = pf.random.uniform(rng, 0.02, 0.03)
    if attachment_depth is None:
        attachment_depth = pf.random.uniform(rng, 0.01, 0.015)
    if radius is None:
        radius = pf.random.uniform(rng, 0.01, 0.015)
    if depth is None:
        depth = pf.random.uniform(rng, 0.06, 0.1)
    if ring_radius is None:
        ring_radius = attachment_radius * pf.random.log_uniform(rng, 2.0, 6.0)
    if ring_minor_scale is None:
        ring_minor_scale = pf.random.uniform(rng, 0.4, 0.7)
    return hardware_ring(
        material=material,
        square=square,
        attachment_radius=attachment_radius,
        attachment_depth=attachment_depth,
        radius=radius,
        depth=depth,
        ring_radius=ring_radius,
        ring_minor_scale=ring_minor_scale,
    )


def hardware_bathroom_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> BathroomHardwareResult:
    rng_choice, rng_hardware = rng.spawn(2)
    hardware_func = pf.control.choice(
        rng_choice,
        [
            (hardware_hook_rand, 1.0),
            (hardware_holder_rand, 1.0),
            (hardware_bar_rand, 1.0),
            (hardware_ring_rand, 1.0),
        ],
    )
    return hardware_func(rng_hardware, material=material)
