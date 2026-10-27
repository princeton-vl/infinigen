# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen table (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/tables/dining_table.py)
# - Alexander Raistrick: refactor for Infinigen2

import math

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import storage
from infinigen2.objects.furniture_bases import (
    TableResult,
    base_pedestal_rand,
    base_square_rand,
    base_straight_rand,
    table_coffee_dimensions_rand,
    table_dimensions_rand,
    table_side_dimensions_rand,
)
from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
    table_top_material_rand,
)
from infinigen2.util import mesh

__all__ = [
    "TableResult",
    "table_cocktail_circular_rand",
    "table_cocktail_rand",
    "table_coffee_circular_rand",
    "table_coffee_dimensions_rand",
    "table_coffee_rand",
    "table_coffee_storage_rand",
    "table_dimensions_rand",
    "table_dining_circular_rand",
    "table_dining_rand",
    "table_side_circular_rand",
    "table_side_dimensions_rand",
    "table_side_rand",
    "table_top",
    "table_top_circular",
    "table_top_rectangular",
]


@pf.nodes.node_function
def table_top(
    size: t.SocketOrVal[pf.Vector] = (1.4, 0.8, 0.05),
    support_loop_offset: t.SocketOrVal[pf.Vector] = (0.02, 0.02, 0.01),
) -> t.ProcNode[pf.MeshObject]:
    """Slab spanning z in [0, size.z]; support_loop_offset sets edge roundness.
    Subdivision is left to an unapplied modifier on the final object."""
    box = mesh.box_with_support_loops(
        size=size,
        vertices_x=4,
        vertices_y=4,
        vertices_z=4,
        support_loop_offset=support_loop_offset,
    )
    smooth = pf.nodes.geo.set_shade_smooth(geometry=box, shade_smooth=True)

    translation = pf.nodes.math.combine_xyz(z=size.z * 0.5)
    return pf.nodes.geo.transform(
        geometry=smooth,
        translation=translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )


@pf.nodes.node_function
def _table_top_circular(
    diameter: t.SocketOrVal[float] = 1.2,
    thickness: t.SocketOrVal[float] = 0.05,
) -> t.ProcNode[pf.MeshObject]:
    top = mesh.quad_cylinder(
        radius=diameter * 0.5,
        depth=thickness,
        resolution=32,
        insets=3,
    )
    top = mesh.crease_sharp(top, threshold_degrees=40.0)
    top = pf.nodes.geo.set_shade_smooth(geometry=top, shade_smooth=True)
    return pf.nodes.geo.transform(
        geometry=top,
        translation=pf.nodes.math.combine_xyz(z=thickness * 0.5),
    )


def _table_straight_base_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    footprint = min(dimensions[0], dimensions[1])
    return base_straight_rand(
        rng,
        dimensions,
        leg_diameter_range=(0.02, 0.02 + 0.16 * footprint),
        leg_inset_range=(0.05 * footprint, 0.15 * footprint),
        leg_placement_top_scale=1.0,
    )


def _table_square_base_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    footprint = min(dimensions[0], dimensions[1])
    return base_square_rand(
        rng,
        dimensions,
        leg_diameter_range=(0.02, 0.02 + 0.12 * footprint),
    )


def _table_pedestal_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    return base_pedestal_rand(
        rng,
        dimensions,
        top_radius_range=(0.012 * dimensions[0], 0.05 * dimensions[0]),
        bottom_radius_range=(0.14 * dimensions[0], 0.29 * dimensions[0]),
    )


def _inscribed_base_dimensions(
    dimensions: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        dimensions[0] / math.sqrt(2.0),
        dimensions[1] / math.sqrt(2.0),
        dimensions[2],
    )


def _table_circular_straight_base_rand(
    rng: pf.RNG, dimensions: tuple[float, float, float]
) -> TableResult:
    dimensions = _inscribed_base_dimensions(dimensions)
    footprint = min(dimensions[0], dimensions[1])
    return base_straight_rand(
        rng,
        dimensions,
        leg_diameter_range=(0.02, 0.02 + 0.16 * footprint),
        leg_placement_bottom_scale=1.0,
        leg_inset_range=(0.0, 0.0),
        leg_placement_top_scale=1.0,
    )


def _top_object(
    top: pf.ProcNode[pf.MeshObject], top_height: float, top_material: pf.Material
) -> TableResult:
    top = pf.nodes.geo.transform(top, translation=(0, 0, top_height))
    top = pf.nodes.to_mesh_object(top)
    pf.ops.object.set_material(
        top, surface=top_material.surface, displacement=top_material.displacement
    )
    return TableResult(mesh=top)


def _assemble_table(
    top: pf.MeshObject, base: pf.MeshObject, leg_material: pf.Material
) -> TableResult:
    pf.ops.object.set_material(
        base, surface=leg_material.surface, displacement=leg_material.displacement
    )
    pf.ops.object.join(top, base)
    pf.ops.modifier.subdivide_surface(top, levels=3, _skip_apply=True)
    return TableResult(mesh=top)


@pf.tracer.generator
def table_top_rectangular(
    x: float,
    y: float,
    top_thickness: float,
    top_height: float,
    top_support_loop_offset: pf.Vector,
    top_material: pf.Material,
) -> TableResult:
    top = table_top(
        size=(x, y, top_thickness),
        support_loop_offset=top_support_loop_offset,
    )
    return _top_object(top, top_height, top_material)


@pf.tracer.generator
def table_top_circular(
    diameter: float,
    top_thickness: float,
    top_height: float,
    top_material: pf.Material,
) -> TableResult:
    top = _table_top_circular(diameter=diameter, thickness=top_thickness)
    return _top_object(top, top_height, top_material)


def table_dining_rand(
    rng: pf.RNG,
    dimensions: tuple[float, float, float] | None = None,
    base: pf.MeshObject | None = None,
    top_thickness: float | None = None,
    top_support_loop_offset: pf.Vector | None = None,
    top_material: pf.Material | None = None,
    leg_material: pf.Material | None = None,
) -> TableResult:
    rng, rng_dims, rng_top_mat, rng_leg_mat, rng_base_choice, rng_base = rng.spawn(6)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)

    x = dimensions[0]
    y = dimensions[1]
    z = dimensions[2]

    if top_thickness is None:
        top_thickness = pf.random.uniform(rng, 0.03, 0.08)

    top_height = z - top_thickness

    vec = pf.nodes.shader.coord().uv
    if top_material is None:
        top_material = table_top_material_rand(rng_top_mat, vec)
    if leg_material is None:
        leg_material = furniture_material_rand(rng_leg_mat, vec)

    if top_support_loop_offset is None:
        corner_frac_x = pf.random.uniform(rng, 0.1, 0.5)
        corner_frac_y = pf.random.uniform(rng, 0.1, 0.5)
        edge_frac = pf.random.uniform(rng, 0.1, 0.5)
        corner_shrink = pf.random.uniform(rng, 0.0, 1.0) ** 2
        top_support_loop_offset = pf.Vector(
            (
                corner_frac_x * corner_shrink * x,
                corner_frac_y * corner_shrink * y,
                edge_frac * top_thickness * (1.0 - corner_shrink),
            )
        )
    if base is None:
        base_options = [
            (_table_straight_base_rand, 2.0),
            (_table_pedestal_rand, 1.0),
            (_table_square_base_rand, 0.6),
        ]
        base_fn = pf.control.choice(rng_base_choice, base_options)
        res = base_fn(rng=rng_base, dimensions=(x, y, top_height))
        base = res.mesh
    top = table_top_rectangular(
        x, y, top_thickness, top_height, top_support_loop_offset, top_material
    ).mesh
    return _assemble_table(top, base, leg_material)


def table_dining_circular_rand(
    rng: pf.RNG,
    diameter: float | None = None,
    height: float | None = None,
    base: pf.MeshObject | None = None,
    top_thickness: float | None = None,
    top_material: pf.Material | None = None,
    leg_material: pf.Material | None = None,
) -> TableResult:
    """Circular dining table sized for four chairs."""
    (
        rng_diameter,
        rng_height,
        rng_thickness,
        rng_top_mat,
        rng_leg_mat,
        rng_base_choice,
        rng_base,
    ) = rng.spawn(7)
    if diameter is None:
        diameter = pf.random.clip_gaussian(rng_diameter, 1.15, 0.2, 0.95, 1.5)
    if height is None:
        height = pf.random.uniform(rng_height, 0.72, 0.76)
    if top_thickness is None:
        top_thickness = pf.random.uniform(rng_thickness, 0.03, 0.08)
    vec = pf.nodes.shader.coord().uv
    if top_material is None:
        top_material = table_top_material_rand(rng_top_mat, vec)
    if leg_material is None:
        leg_material = furniture_material_rand(rng_leg_mat, vec)
    top_height = height - top_thickness
    if base is None:
        base_fn = pf.control.choice(
            rng_base_choice,
            [
                (_table_circular_straight_base_rand, 1.0),
                (_table_pedestal_rand, 1.0),
            ],
        )
        base = base_fn(rng_base, (diameter, diameter, top_height)).mesh
    top = table_top_circular(diameter, top_thickness, top_height, top_material).mesh
    return _assemble_table(top, base, leg_material)


def table_side_rand(rng: pf.RNG) -> TableResult:
    """Side table."""
    rng, rng_dims, rng_table = rng.spawn(3)
    dimensions = table_side_dimensions_rand(rng_dims)
    top_thickness = pf.random.uniform(rng, 0.01, 0.04)
    return table_dining_rand(rng_table, dimensions, top_thickness=top_thickness)


def _table_coffee_legged_rand(rng: pf.RNG) -> TableResult:
    """Low rectangular coffee table on legs."""
    rng, rng_dims, rng_table = rng.spawn(3)
    dimensions = table_coffee_dimensions_rand(rng_dims)
    top_thickness = pf.random.uniform(rng, 0.02, 0.04)
    return table_dining_rand(rng_table, dimensions, top_thickness=top_thickness)


def table_coffee_storage_rand(rng: pf.RNG) -> TableResult:
    result = storage.storage_table_coffee_rand(rng)
    mesh.center_footprint(result.mesh)
    return TableResult(mesh=result.mesh)


def table_coffee_circular_rand(rng: pf.RNG) -> TableResult:
    """Round coffee table, 30-42 in (0.76-1.07 m) across and 16-19 in high."""
    rng, rng_table = rng.spawn(2)
    return table_dining_circular_rand(
        rng_table,
        diameter=pf.random.uniform(rng, 0.76, 1.07),
        height=pf.random.uniform(rng, 0.4, 0.48),
        top_thickness=pf.random.uniform(rng, 0.02, 0.05),
    )


def table_side_circular_rand(rng: pf.RNG) -> TableResult:
    """Round end table, 16-24 in (0.4-0.6 m) across, near sofa arm height."""
    rng, rng_table = rng.spawn(2)
    return table_dining_circular_rand(
        rng_table,
        diameter=pf.random.uniform(rng, 0.4, 0.6),
        height=pf.random.uniform(rng, 0.5, 0.65),
        top_thickness=pf.random.uniform(rng, 0.015, 0.04),
    )


def table_coffee_rand(rng: pf.RNG) -> TableResult:
    """Rectangular, round or storage coffee table."""
    func = pf.control.choice(
        rng,
        [
            (_table_coffee_legged_rand, 2.0),
            (table_coffee_circular_rand, 1.0),
            (table_coffee_storage_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


def _table_cocktail_pedestal_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    x = dimensions[0]
    return base_pedestal_rand(
        rng,
        dimensions,
        top_radius_range=(0.012 * x, 0.05 * x),
        bottom_radius_range=(0.325 * x, 0.52 * x),
    )


def table_cocktail_circular_rand(rng: pf.RNG) -> TableResult:
    """Circular cocktail/bar table, usually with a wide pedestal base."""
    rng_dims, rng_thickness, rng_base_choice, rng_base, rng_table = rng.spawn(5)
    diameter = pf.random.uniform(rng_dims, 0.5, 0.8)
    height = pf.random.uniform(rng_dims, 1.0, 1.1)
    top_thickness = pf.random.uniform(rng_thickness, 0.03, 0.08)
    base_fn = pf.control.choice(
        rng_base_choice,
        [
            (_table_cocktail_pedestal_rand, 2.0),
            (_table_circular_straight_base_rand, 1.0),
        ],
    )
    base = base_fn(rng_base, (diameter, diameter, height - top_thickness)).mesh
    return table_dining_circular_rand(
        rng_table,
        diameter=diameter,
        height=height,
        base=base,
        top_thickness=top_thickness,
    )


def _table_cocktail_square_rand(rng: pf.RNG) -> TableResult:
    rng_dims, rng_thickness, rng_base_choice, rng_base, rng_table = rng.spawn(5)
    size = pf.random.uniform(rng_dims, 0.5, 0.8)
    height = pf.random.uniform(rng_dims, 1.0, 1.1)
    top_thickness = pf.random.uniform(rng_thickness, 0.03, 0.08)
    base_fn = pf.control.choice(
        rng_base_choice,
        [
            (_table_cocktail_pedestal_rand, 2.0),
            (_table_straight_base_rand, 1.0),
            (_table_square_base_rand, 1.0),
        ],
    )
    base = base_fn(rng_base, (size, size, height - top_thickness)).mesh
    return table_dining_rand(
        rng_table,
        dimensions=(size, size, height),
        base=base,
        top_thickness=top_thickness,
    )


def table_cocktail_rand(rng: pf.RNG) -> TableResult:
    """Circular or square cocktail/bar table."""
    rng_choice, rng_table = rng.spawn(2)
    table_fn = pf.control.choice(
        rng_choice,
        [
            (table_cocktail_circular_rand, 1.0),
            (_table_cocktail_square_rand, 1.0),
        ],
    )
    return table_fn(rng_table)
