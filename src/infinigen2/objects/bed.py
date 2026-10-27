# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import chair, furniture_bases, storage, table
from infinigen2.shaders.composites import fabric_wrinkled
from infinigen2.shaders.functionality_lists import (
    fabric_general_rand,
    fabric_sturdy_rand,
    furniture_material_rand,
)
from infinigen2.util import mesh

__all__ = [
    "BedPartResult",
    "BedResult",
    "bed",
    "bed_board_solid",
    "bed_mattress",
    "bed_mattress_rand",
    "bed_platform",
    "bed_rand",
]


class BedResult(NamedTuple):
    mesh: pf.MeshObject
    mattress_child: pf.MeshObject


class BedPartResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.generator
def bed_platform(
    dimensions: pf.Vector, frame_height: float, frame_material: pf.Material
) -> BedPartResult:
    platform = mesh.box_with_support_loops(
        size=(dimensions.x + 0.08, dimensions.y + 0.08, 0.10),
        vertices_x=4,
        vertices_y=4,
        vertices_z=4,
        support_loop_offset=(0.01, 0.01, 0.01),
    )
    platform = pf.nodes.to_mesh_object(platform)
    pf.ops.object.set_transform(platform, location=(0, 0, frame_height - 0.05))
    pf.ops.object.set_material(
        platform,
        surface=frame_material.surface,
        displacement=frame_material.displacement,
    )
    return BedPartResult(platform)


@pf.tracer.generator
def bed(
    platform: pf.MeshObject,
    base: pf.MeshObject,
    headboard: pf.MeshObject,
    footboard: pf.MeshObject | None,
    mattress_child: pf.MeshObject,
    dimensions: pf.Vector,
    frame_height: float,
    base_material: pf.Material,
) -> BedResult:
    pf.ops.object.set_transform(
        headboard, location=(-dimensions.x * 0.5 - 0.04, 0, frame_height - 0.10)
    )
    pf.ops.object.set_material(
        base, surface=base_material.surface, displacement=base_material.displacement
    )
    pf.ops.object.join(platform, base)
    pf.ops.object.join(platform, headboard)
    if footboard is not None:
        pf.ops.object.set_transform(
            footboard,
            location=(dimensions.x * 0.5 + 0.04, 0, frame_height - 0.10),
            rotation_euler=(0, 0, math.pi),
        )
        pf.ops.object.join(platform, footboard)
    pf.ops.uv.cube_project(platform, uv_name="UVMap")
    mattress_child.item().parent = platform.item()
    mattress_child.item().matrix_parent_inverse.identity()
    return BedResult(platform, mattress_child)


@pf.tracer.generator
def bed_board_solid(
    board_dimensions: pf.Vector,
    top_rise: float,
    crease_all: bool,
    crease_long: bool,
    material: pf.Material,
) -> BedPartResult:
    profile = pf.Vector(
        (
            board_dimensions.x,
            board_dimensions.y,
            board_dimensions.z - top_rise * 0.75,
        )
    )
    geo = chair.chair_back_solid(profile, top_rise=top_rise)
    edge = pf.nodes.geo.input_mesh_edge_vertices()
    dx = pf.nodes.math.absolute(edge.position_1.x - edge.position_2.x)
    dy = pf.nodes.math.absolute(edge.position_1.y - edge.position_2.y)
    dz = pf.nodes.math.absolute(edge.position_1.z - edge.position_2.z)
    width_dominant = pf.nodes.func.boolean_and(
        a=(dy > dx).astype(dtype=bool),
        b=(dy > dz).astype(dtype=bool),
    )
    sharp = (
        pf.nodes.geo.input_mesh_edge_angle().unsigned_angle > math.radians(40)
    ).astype(dtype=bool)
    long_crease = pf.nodes.func.boolean_and(a=crease_long, b=width_dominant)
    crease = pf.nodes.func.boolean_or(
        a=crease_all, b=pf.nodes.func.boolean_and(a=long_crease, b=sharp)
    )
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo,
        name="crease_edge",
        value=crease.astype(dtype=float),
        domain="EDGE",
    )
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.object.set_material(
        obj,
        surface=material.surface,
        displacement=material.displacement,
    )
    return BedPartResult(obj)


@pf.tracer.generator
def bed_mattress(
    dimensions: pf.Vector, center_z: float, material: pf.Material
) -> BedPartResult:
    mattress_geo = mesh.box_with_support_loops(
        size=dimensions,
        vertices_x=30,
        vertices_y=30,
        vertices_z=6,
        support_loop_offset=(0.12, 0.12, 0.07),
    )
    mattress = pf.nodes.to_mesh_object(mattress_geo)
    pf.ops.object.set_transform(mattress, location=(0, 0, center_z))
    pf.ops.object.set_material(
        mattress,
        surface=material.surface,
        displacement=material.displacement,
    )
    return BedPartResult(mattress)


def _bed_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    material: pf.Material,
) -> table.TableResult:
    rng_choice, rng_body = rng.spawn(2)

    def storage_base(
        r: pf.RNG, dims: pf.Vector, base_material: pf.Material
    ) -> table.TableResult:
        depth = dims.y * 0.5 - 0.01
        length = dims.x - 0.08
        first = storage.storage_composite_rand(
            r,
            dimensions=pf.Vector((depth, length, dims.z - 0.05)),
            n_spaces_z=1,
            frame_thickness=0.02,
            frame_material=base_material,
            row_divider_width=0.025,
            col_divider_width=0.02,
            back_width=0.02,
        ).mesh
        pf.ops.object.set_transform(first, location=(-depth * 0.5, -length * 0.5, 0))
        pf.ops.mesh.transform_apply(first, location=True, rotation=False, scale=False)
        second = first.clone()
        pf.ops.object.set_transform(
            first,
            location=(0, (depth + 0.02) * 0.5, 0),
            rotation_euler=(0, 0, math.pi * 0.5),
        )
        pf.ops.object.set_transform(
            second,
            location=(0, -(depth + 0.02) * 0.5, 0),
            rotation_euler=(0, 0, -math.pi * 0.5),
        )
        pf.ops.object.join(first, second)
        pf.ops.mesh.transform_apply(first, location=True, rotation=True, scale=False)
        return table.TableResult(first)

    def leg_base(
        r: pf.RNG, dims: pf.Vector, _material: pf.Material
    ) -> table.TableResult:
        r_choice, r_body = r.spawn(2)
        leg_diameter = 0.06
        leg_width = leg_diameter / math.sqrt(2.0)
        footprint_x = dims.x + 0.08
        footprint_y = dims.y + 0.08
        straight_dimensions = pf.Vector(
            (
                footprint_x + leg_diameter - leg_width,
                footprint_y + leg_diameter - leg_width,
                dims.z,
            )
        )
        square_scale = 0.94
        square_dimensions = pf.Vector(
            (
                (footprint_x - leg_width) / square_scale,
                (footprint_y - leg_width) / square_scale,
                dims.z,
            )
        )
        base_fn = pf.control.choice(
            r_choice,
            [
                (
                    partial(
                        furniture_bases.base_straight_rand,
                        dimensions=straight_dimensions,
                    ),
                    1.0,
                ),
                (
                    partial(
                        furniture_bases.base_square_rand,
                        dimensions=square_dimensions,
                    ),
                    1.0,
                ),
            ],
        )
        return base_fn(
            r_body,
            leg_diameter=leg_diameter,
            leg_placement_bottom_scale=1.0,
            close_edges=True,
        )

    base_fn = pf.control.choice(
        rng_choice,
        [
            (leg_base, 2.0),
            (storage_base, 1.0),
        ],
    )
    return base_fn(rng_body, dimensions, material)


def _frame_material_rand(rng: pf.RNG, hard_material: pf.Material) -> pf.Material:
    r_choice, r_material = rng.spawn(2)
    vector = pf.nodes.shader.coord().uv
    fn = pf.control.choice(
        r_choice,
        [
            (lambda _rng, _vector: hard_material, 2.0),
            (fabric_sturdy_rand, 2.0),
        ],
    )
    return fn(r_material, vector)


def _bedding_rand(rng: pf.RNG, vec: t.SocketOrVal[pf.Vector]) -> pf.Material:
    """General cloth with an additional chance of large folds."""
    rng_fabric, rng_choice, rng_wrinkles = rng.spawn(3)
    material = fabric_general_rand(rng_fabric, vec)
    wrinkles_func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v, m: m, 1.0),
            (fabric_wrinkled.wrinkles_big_overlay_rand, 1.0),
        ],
    )
    return wrinkles_func(rng_wrinkles, vec, material)


def bed_mattress_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    center_z: float,
) -> pf.MeshObject:
    material = _bedding_rand(rng, pf.nodes.shader.coord().uv)
    mattress = bed_mattress(dimensions, center_z, material).mesh
    pf.ops.modifier.subdivide_surface(mattress, levels=3, _skip_apply=True)
    return mattress


def bed_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> BedResult:
    (
        r_size,
        r_thick,
        r_frame,
        r_head,
        r_base,
        r_style,
        r_material,
        r_mattress,
    ) = rng.spawn(8)
    if dimensions is None:
        width = pf.random.uniform(r_size, 0.90, 1.80)
        dimensions = pf.Vector((2.0, width, pf.random.uniform(r_thick, 0.20, 0.24)))
    frame_height = pf.random.uniform(r_frame, 0.32, 0.40)
    head_height = pf.random.uniform(r_head, 0.85, 1.00)
    r_hard, r_upholstery = r_material.spawn(2)
    r_back_choice, r_finish, r_head_board, r_foot_choice, r_foot_board = r_style.spawn(
        5
    )
    material = furniture_material_rand(r_hard, pf.nodes.shader.coord().uv)
    frame_material = _frame_material_rand(r_upholstery, material)
    base = _bed_base_rand(
        r_base,
        pf.Vector((dimensions.x, dimensions.y, frame_height - 0.05)),
        material,
    ).mesh

    r_finish_choice, r_top_rise = r_finish.spawn(2)
    rounded_top_rise = pf.random.clip_gaussian(r_top_rise, 0.05, 0.035, 0.0, 0.12)
    finish = pf.control.choice(
        r_finish_choice,
        [
            ((0.0, True, False), 1.0),
            ((rounded_top_rise, False, False), 1.0),
            ((rounded_top_rise, False, True), 1.0),
        ],
    )

    def solid_board(_r: pf.RNG, board_dimensions: pf.Vector) -> pf.MeshObject:
        return bed_board_solid(
            board_dimensions, finish[0], finish[1], finish[2], frame_material
        ).mesh

    def slatted_board(r: pf.RNG, board_dimensions: pf.Vector) -> pf.MeshObject:
        def slat_count_rand(
            r_count: pf.RNG, board_width: float, slat_width: float
        ) -> int:
            max_slats = max(2, math.floor(board_width / (1.2 * slat_width)))
            return pf.random.randint(r_count, 2, max_slats + 1)

        return chair.chair_back_rand(
            r,
            dimensions=board_dimensions,
            material=material,
            slat_count_rand=slat_count_rand,
        ).mesh

    # one style for the pair, so a footboard always matches its headboard
    board_fn = pf.control.choice(
        r_back_choice, [(solid_board, 1.0), (slatted_board, 1.0)]
    )

    def matching_board(r: pf.RNG, board_dimensions: pf.Vector) -> pf.MeshObject | None:
        return board_fn(r, board_dimensions)

    def no_board(_r: pf.RNG, _board_dimensions: pf.Vector) -> pf.MeshObject | None:
        return None

    r_head_thickness, r_head_shape = r_head_board.spawn(2)
    head = board_fn(
        r_head_shape,
        pf.Vector(
            (
                pf.random.uniform(r_head_thickness, 0.035, 0.07),
                dimensions.y + 0.08,
                head_height - frame_height + 0.10,
            )
        ),
    )
    foot_fn = pf.control.choice(r_foot_choice, [(no_board, 2.0), (matching_board, 1.0)])
    foot = foot_fn(
        r_foot_board,
        pf.Vector((0.05, dimensions.y + 0.08, 0.10 + dimensions.z * 0.75)),
    )
    mattress = bed_mattress_rand(r_mattress, dimensions, 0.05 + dimensions.z * 0.5)
    platform = bed_platform(dimensions, frame_height, frame_material).mesh
    # one level here plus the assembled bed's two keeps the soft parts at three
    pf.ops.modifier.subdivide_surface(platform, levels=1)
    assembled = bed(
        platform,
        base,
        head,
        foot,
        mattress,
        dimensions,
        frame_height,
        material,
    )
    frame = assembled.mesh
    pf.ops.modifier.subdivide_surface(frame, levels=2, _skip_apply=True)
    return BedResult(frame, assembled.mattress_child)
