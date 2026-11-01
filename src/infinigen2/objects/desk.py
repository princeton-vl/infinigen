# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import procfunc as pf

from infinigen2.objects import furniture_bases, storage, table
from infinigen2.shaders.functionality_lists import table_top_material_rand

__all__ = [
    "DeskResult",
    "desk_dimensions_rand",
    "desk_integrated_storage_rand",
    "desk_rand",
    "desk_tabletop_rand",
    "desk_with_side_storage_rand",
    "desk_with_top_storage_rand",
]


class DeskResult(NamedTuple):
    mesh: pf.MeshObject


def desk_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Desk footprint and height in meters, ordered as depth, width, height."""
    rng_depth, rng_width, rng_height = rng.spawn(3)
    return pf.Vector(
        (
            pf.random.uniform(rng_depth, 0.55, 0.90),
            pf.random.uniform(rng_width, 0.90, 2.40),
            pf.random.uniform(rng_height, 0.70, 0.78),
        )
    )


def _desktop_support_loop_offset_rand(rng: pf.RNG) -> pf.Vector:
    rng_corner, rng_edge = rng.spawn(2)
    corner_offset = pf.random.uniform(rng_corner, 0.008, 0.025)
    edge_offset = pf.random.uniform(rng_edge, 0.002, 0.006)
    return pf.Vector((corner_offset, corner_offset, edge_offset))


def _side_cell_shelf_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    top_height: float,
) -> pf.MeshObject:
    (
        rng_frame,
        rng_row,
        rng_col,
        rng_back_choice,
        rng_back_width,
        rng_left,
        rng_aspect,
        rng_width,
    ) = rng.spawn(8)
    back_width = pf.control.choice(
        rng_back_choice,
        [
            (0.0, 1.0),
            (pf.random.uniform(rng_back_width, 0.012, 0.018), 1.0),
        ],
    )
    cabinet_width = dimensions.y * pf.random.uniform(rng_width, 0.15, 0.30)
    frame_thickness = pf.random.uniform(rng_frame, 0.015, 0.025)
    row_divider_width = pf.random.uniform(rng_row, 0.012, 0.022)
    col_divider_width = pf.random.uniform(rng_col, 0.012, 0.022)
    cabinet_dimensions = pf.Vector((dimensions.x, cabinet_width, top_height))
    left = storage.storage_composite_rand(
        rng_left,
        dimensions=cabinet_dimensions,
        n_spaces_y=1,
        frame_thickness=frame_thickness,
        row_divider_width=row_divider_width,
        col_divider_width=col_divider_width,
        back_width=back_width,
        desired_slot_aspect=pf.random.uniform(rng_aspect, 2.0, 10.0),
    ).mesh
    right = left.clone()
    pf.ops.object.set_transform(
        left,
        location=(-dimensions.x / 2, -dimensions.y / 2, 0.0),
    )
    pf.ops.object.set_transform(
        right,
        location=(-dimensions.x / 2, dimensions.y / 2 - cabinet_width, 0.0),
    )
    pf.ops.object.join(left, right)
    return left


def _thin_cell_shelf_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    cabinet_height: float,
    bottom_height: float,
    frame_material: pf.Material | None = None,
) -> pf.MeshObject:
    rng_frame, rng_row, rng_col, rng_back, rng_storage, rng_aspect = rng.spawn(6)
    result = storage.storage_composite_rand(
        rng_storage,
        dimensions=pf.Vector((dimensions.x, dimensions.y, cabinet_height)),
        n_spaces_z=1,
        frame_thickness=pf.random.uniform(rng_frame, 0.012, 0.018),
        row_divider_width=pf.random.uniform(rng_row, 0.010, 0.016),
        col_divider_width=pf.random.uniform(rng_col, 0.010, 0.016),
        back_width=pf.random.uniform(rng_back, 0.012, 0.018),
        frame_material=frame_material,
        desired_slot_aspect=pf.random.uniform(rng_aspect, 2.0, 10.0),
    ).mesh
    pf.ops.object.set_transform(
        result,
        location=(-dimensions.x / 2, -dimensions.y / 2, bottom_height),
    )
    return result


def _tabletop_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> pf.MeshObject:
    rng_top, rng_top_shape, rng_base, rng_table = rng.spawn(4)
    top_thickness = pf.random.uniform(rng_top, 0.025, 0.055)
    top_height = dimensions.z - top_thickness
    base = furniture_bases.base_straight_rand(
        rng_base,
        dimensions=pf.Vector((dimensions.x, dimensions.y, top_height)),
        close_edges=True,
    ).mesh
    result = table.table_dining_rand(
        rng_table,
        dimensions=dimensions,
        base=base,
        top_thickness=top_thickness,
        top_support_loop_offset=_desktop_support_loop_offset_rand(rng_top_shape),
    )
    return result.mesh


def _side_storage_tabletop_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> pf.MeshObject:
    rng_top, rng_top_shape, rng_base, rng_table = rng.spawn(4)
    top_thickness = pf.random.uniform(rng_top, 0.025, 0.055)
    top_height = dimensions.z - top_thickness
    base = _side_cell_shelf_base_rand(rng_base, dimensions, top_height)
    result = table.table_dining_rand(
        rng_table,
        dimensions=dimensions,
        base=base,
        top_thickness=top_thickness,
        top_support_loop_offset=_desktop_support_loop_offset_rand(rng_top_shape),
    )
    return result.mesh


def _integrated_storage_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> pf.MeshObject:
    rng_height, rng_base, rng_cabinet, rng_material = rng.spawn(4)
    cabinet_height = pf.random.uniform(rng_height, 0.10, 0.18)
    base_height = dimensions.z - cabinet_height
    base = furniture_bases.base_straight_rand(
        rng_base,
        dimensions=pf.Vector((dimensions.x, dimensions.y, base_height)),
        close_edges=True,
    ).mesh
    material = table_top_material_rand(rng_material, pf.nodes.shader.coord().uv)
    cabinet = _thin_cell_shelf_rand(
        rng_cabinet,
        dimensions,
        cabinet_height=cabinet_height,
        bottom_height=base_height,
        frame_material=material,
    )
    pf.ops.object.join(cabinet, base)
    return cabinet


def _tabletop_storage_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> pf.MeshObject:
    (
        rng_top,
        rng_top_shape,
        rng_height,
        rng_base,
        rng_cabinet,
        rng_table,
    ) = rng.spawn(6)
    top_thickness = pf.random.uniform(rng_top, 0.025, 0.055)
    top_height = dimensions.z - top_thickness
    cabinet_height = pf.random.uniform(rng_height, 0.08, 0.14)
    base_height = top_height - cabinet_height
    base = furniture_bases.base_straight_rand(
        rng_base,
        dimensions=pf.Vector((dimensions.x, dimensions.y, base_height)),
        close_edges=True,
    ).mesh
    cabinet = _thin_cell_shelf_rand(
        rng_cabinet,
        dimensions,
        cabinet_height=cabinet_height,
        bottom_height=base_height,
    )
    pf.ops.object.join(base, cabinet)
    result = table.table_dining_rand(
        rng_table,
        dimensions=dimensions,
        base=base,
        top_thickness=top_thickness,
        top_support_loop_offset=_desktop_support_loop_offset_rand(rng_top_shape),
    )
    return result.mesh


def desk_tabletop_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> DeskResult:
    rng_dimensions, rng_surface = rng.spawn(2)
    if dimensions is None:
        dimensions = desk_dimensions_rand(rng_dimensions)
    mesh = _tabletop_surface_rand(rng_surface, dimensions)
    mesh.item().name = "desk_tabletop"
    return DeskResult(mesh)


def desk_integrated_storage_rand(
    rng: pf.RNG, dimensions: pf.Vector | None = None
) -> DeskResult:
    rng_dimensions, rng_surface = rng.spawn(2)
    if dimensions is None:
        dimensions = desk_dimensions_rand(rng_dimensions)
    mesh = _integrated_storage_surface_rand(rng_surface, dimensions)
    mesh.item().name = "desk_integrated_storage"
    return DeskResult(mesh)


def desk_with_top_storage_rand(
    rng: pf.RNG, dimensions: pf.Vector | None = None
) -> DeskResult:
    rng_dimensions, rng_surface = rng.spawn(2)
    if dimensions is None:
        dimensions = desk_dimensions_rand(rng_dimensions)
    mesh = _tabletop_storage_surface_rand(rng_surface, dimensions)
    mesh.item().name = "desk_with_top_storage"
    return DeskResult(mesh)


def desk_with_side_storage_rand(
    rng: pf.RNG, dimensions: pf.Vector | None = None
) -> DeskResult:
    rng_dimensions, rng_surface = rng.spawn(2)
    if dimensions is None:
        dimensions = desk_dimensions_rand(rng_dimensions)
    mesh = _side_storage_tabletop_surface_rand(rng_surface, dimensions)
    mesh.item().name = "desk_with_side_storage"
    return DeskResult(mesh)


def desk_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> DeskResult:
    """Sample a complete desk from the independently usable desk generators."""
    rng_choice, rng_desk = rng.spawn(2)
    desk_fn = pf.control.choice(
        rng_choice,
        [
            (desk_tabletop_rand, 5.0),
            (desk_integrated_storage_rand, 1.0),
            (desk_with_top_storage_rand, 1.0),
            (desk_with_side_storage_rand, 3.0),
        ],
    )
    return desk_fn(rng_desk, dimensions)
