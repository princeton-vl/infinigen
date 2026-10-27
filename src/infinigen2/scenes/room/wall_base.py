# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.scenes.placement import collision as ccol
from infinigen2.shaders.functionality_lists import wall_material_rand
from infinigen2.util import mesh as mesh_util
from infinigen2.util.scene_cleanup import delete_object

__all__ = [
    "ROOM_SUBSURF_LEVELS",
    "SURFACE_UV_WINDING_SIGN",
    "WallResult",
    "WallSurface",
    "extrude_for_thickness",
    "finish_wall_plane",
    "fit_grid_margins",
    "name_objects",
    "overlap_wall_plane_edges",
    "plain_wall",
    "plane_to_posed_canonical_mesh",
    "resolve_wall_inputs",
    "seat_upright_cabinet",
    "shift_wall_ends",
    "standalone_wall",
    "upright_cabinet_footprint",
    "wall_storage_width_rand",
    "wall_plain_rand",
]

ROOM_SUBSURF_LEVELS = 6
SURFACE_UV_WINDING_SIGN = -1.0


class WallSurface(NamedTuple):
    obj: pf.MeshObject
    width: float
    height: float


class WallResult(NamedTuple):
    all_objects: list[pf.MeshObject]  # every scene object, each exactly once
    wall_planes: list[pf.MeshObject]
    backs: list[pf.MeshObject]  # structural lightblocker meshes (wall backs)
    sills: list[pf.MeshObject]  # cutout reveal/sill meshes
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    lights: list[pf.LightObject]  # window portals or other attached lights
    decorations: dict[
        str, list[pf.MeshObject]
    ]  # window/painting/shelf/door instances by type
    corner_walls: list[pf.MeshObject] = []
    storages: list[pf.MeshObject] = []
    colliders: ccol.CollisionSet | None = None


@pf.tracer.generator
def standalone_wall(width: float, height: float) -> pf.MeshObject:
    grid = pf.nodes.geo.mesh_grid(vertices_x=2, vertices_y=2, size_x=1.0, size_y=1.0)
    position = pf.nodes.geo.input_position()
    along = (position.x + 0.5) * width
    up = (position.y + 0.5) * height
    # mirrored U matches a room's perimeter unwrap, so cutouts recess into the wall
    uv = pf.nodes.math.combine_xyz(x=width - along, y=up)
    geometry = pf.nodes.geo.store_named_attribute(
        geometry=grid.mesh,
        name="UVMap",
        value=uv,
        domain="CORNER",
        data_type=pf.NodeDataType.FLOAT_VECTOR_2D,
    )
    wall_position = pf.nodes.math.combine_xyz(y=along, z=up)
    geometry = pf.nodes.geo.set_position(geometry=geometry, position=wall_position)
    wall = pf.nodes.to_mesh_object(geometry)
    return wall


def _standalone_wall_rand(
    rng: pf.RNG,
    width: float | None = None,
    height: float | None = None,
) -> WallSurface:
    if width is None:
        width = pf.random.clip_gaussian(rng, 4.5, 1.0, 2.5, 8.0)
    if height is None:
        height = pf.random.clip_gaussian(rng, 2.5, 0.5, 2.2, 4.5)
    return WallSurface(obj=standalone_wall(width, height), width=width, height=height)


def resolve_wall_inputs(
    rng: pf.RNG,
    wall: WallSurface | None,
    wall_material: pf.Material | None,
) -> tuple[pf.RNG, WallSurface, pf.Material]:
    if wall is not None and wall_material is not None:
        return rng, wall, wall_material

    rng_wall, rng_material, rng_feature = rng.spawn(3)
    if wall is None:
        wall = _standalone_wall_rand(rng_wall)
    if wall_material is None:
        wall_material = wall_material_rand(rng_material, pf.nodes.shader.coord().uv)
    return rng_feature, wall, wall_material


@pf.tracer.generator
def extrude_for_thickness(obj: pf.MeshObject, thickness: float) -> pf.MeshObject:
    """Solidify a flat surface into a back-thickness slab (the lightblocker body)."""
    geo = pf.nodes.geo.object_info(obj).geometry
    extruded = mesh_util.extrude_mesh_seamless_uvs(
        mesh=geo,
        selection=True,
        offset_scale=-thickness,
        uv_winding_sign=SURFACE_UV_WINDING_SIGN,
    )
    result = pf.nodes.to_mesh_object(extruded.mesh)
    return result


def plane_to_posed_canonical_mesh(
    obj: pf.MeshObject,
    up_axis: str = "Z",
) -> pf.MeshObject:
    """Bake a planar mesh's location/rotation into the object transform, appearance unchanged."""
    # area-weighted normal (sliver faces have garbage normals)
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    normals = pf.ops.attr.polygon_normals(obj)
    areas = pf.ops.attr.polygon_areas(obj)

    pos = pf.Vector((np.array(bbox_min) + np.array(bbox_max)) / 2)
    normal = pf.Vector((normals * areas[:, None]).sum(axis=0)).normalized()

    # normal -> +X, keeping up_axis upright
    rot = normal.to_track_quat("X", up_axis).inverted()

    pf.ops.object.set_transform(obj, location=-pos)
    pf.ops.mesh.transform_apply(obj)
    pf.ops.object.set_transform(obj, rotation_euler=rot.to_euler())
    pf.ops.mesh.transform_apply(obj)

    pf.ops.object.set_transform(obj, location=pos)
    pf.ops.object.set_transform(obj, rotation_euler=rot.inverted().to_euler())

    return obj


@pf.tracer.generator
def finish_wall_plane(wall: pf.MeshObject, wall_material: pf.Material) -> pf.MeshObject:
    refined = mesh_util.refine_lattice_cage(wall, 0.8)
    if refined is not wall:
        pf.ops.object.set_transform(
            refined, wall.item().location, wall.item().rotation_euler
        )
        delete_object(wall.item())
    pf.ops.object.set_material(
        refined,
        surface=wall_material.surface,
        displacement=wall_material.displacement,
    )
    mesh_util.crease_all_edges(refined)
    return refined


def _name_materials(obj: pf.MeshObject, base: str) -> None:
    for j, slot in enumerate(obj.item().material_slots):
        if slot.material is not None:
            slot.material.name = f"{base}_{j}"


def name_objects(objs: list[pf.MeshObject], name: str) -> list[pf.MeshObject]:
    for i, obj in enumerate(objs):
        object_name = f"{name}_{i:02d}"
        obj.item().name = object_name
        _name_materials(obj, object_name)
    return objs


@pf.tracer.primitive(mutates=["wall"])
def shift_wall_ends(
    wall: pf.MeshObject,
    direction: tuple[float, float, float],
    low: float,
    high: float,
) -> None:
    """Move each end of a wall mesh along world `direction`, keeping metric U."""
    if low == 0.0 and high == 0.0:
        return
    item = wall.item()
    matrix = np.array(item.matrix_world)
    rotation = matrix[:3, :3]
    offset = matrix[:3, 3]
    direction_world = np.asarray(direction)
    world = pf.ops.attr.vertex_positions(wall) @ rotation.T + offset
    along = world @ direction_world
    at_low = np.isclose(along, along.min(), rtol=0.0, atol=1e-5)
    at_high = np.isclose(along, along.max(), rtol=0.0, atol=1e-5)

    loop_vertices = np.empty(len(item.data.loops), dtype=int)
    item.data.loops.foreach_get("vertex_index", loop_vertices)
    loops_low = at_low[loop_vertices]
    loops_high = at_high[loop_vertices]
    uvs = pf.ops.attr.uv_coords(wall)
    u_low = np.median(uvs[loops_low, 0])
    u_high = np.median(uvs[loops_high, 0])
    u_per_meter = (u_high - u_low) / (along.max() - along.min())

    world[at_low] -= direction_world * low
    world[at_high] += direction_world * high
    uvs[loops_low, 0] -= u_per_meter * low
    uvs[loops_high, 0] += u_per_meter * high
    local = (world - offset) @ np.linalg.inv(rotation).T
    pf.ops.attr.write_vertex_positions(wall, local)
    pf.ops.attr.write_uv_coords(wall, uvs)
    item.data.update()


def overlap_wall_plane_edges(wall: pf.MeshObject, overlap: float) -> None:
    along_wall = np.array(wall.item().matrix_world)[:3, 1]
    direction = along_wall / np.linalg.norm(along_wall)
    shift_wall_ends(
        wall,
        (float(direction[0]), float(direction[1]), float(direction[2])),
        overlap,
        overlap,
    )


def plain_wall(
    wall: WallSurface, wall_material: pf.Material, wall_thickness: float
) -> tuple[pf.MeshObject, pf.MeshObject]:
    """Materialise, thicken and canonicalise a bare wall (no cutouts)."""
    wall_thick = extrude_for_thickness(wall.obj, wall_thickness)
    wall_thick.item().name = "room_wall_back"
    finished = finish_wall_plane(wall.obj, wall_material)
    pf.ops.modifier.subdivide_surface(
        finished, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )
    finished = plane_to_posed_canonical_mesh(finished)
    return finished, wall_thick


@pf.tracer.grammar
def wall_plain_rand(
    rng: pf.RNG,
    wall: WallSurface | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    finished, wall_thick = plain_wall(wall, wall_material, wall_thickness)
    return WallResult(
        all_objects=[finished, wall_thick],
        wall_planes=[finished],
        backs=[wall_thick],
        sills=[],
        storage_containers=[],
        supports=[],
        lights=[],
        decorations={},
    )


def upright_cabinet_footprint(width: float, height: float) -> pf.MeshObject:
    """Flat wall-frame rect (Y=width=U, Z=height=V) for niche-grid sizing."""
    return pf.nodes.to_mesh_object(
        pf.nodes.geo.mesh_cube(size=(0.0, width, height)).mesh
    )


def seat_upright_cabinet(
    cab: pf.MeshObject, width: float, height: float, back_depth: float
) -> pf.MeshObject:
    """Center an upright cabinet in the wall frame (X=depth out, Y=width, Z=height).

    Y/Z centered on the niche; X shifted so the back sits back_depth behind the surface
    (opening at depth-back_depth). Placement maps local X->normal, Y->along-wall, Z->up.
    """
    bmin, bmax = pf.ops.attr.bbox_min_max(cab)
    pf.ops.object.set_transform(
        cab,
        location=(
            -bmin[0] - back_depth,
            -(bmin[1] + bmax[1]) / 2,
            -(bmin[2] + bmax[2]) / 2,
        ),
    )
    pf.ops.mesh.transform_apply(cab)
    return cab


def fit_grid_margins(
    extent: float,
    item_size: float,
    spacing: float,
    min_margin: float,
    split: float,
    n_cap: int = 1000,
) -> tuple[float, float, int]:
    """Fit item count across extent and push slack into edge margins (low, high, n)."""
    usable = extent - 2 * min_margin
    n = 1
    if usable > item_size:
        n = max(1, int((usable - item_size) / (item_size + spacing)) + 1)
    n = min(n, n_cap)
    used = n * item_size + (n - 1) * spacing
    slack = max(0.02, extent - used - 2 * min_margin) - 0.02
    return min_margin + slack * split, min_margin + slack * (1 - split), n


def wall_storage_width_rand(rng: pf.RNG, wall_width: float, min_margin: float) -> float:
    usable_width = max(0.8, wall_width - 2 * min_margin)
    max_width = 0.9 * usable_width
    return pf.random.uniform(rng, min(1.0, max_width), max_width)
