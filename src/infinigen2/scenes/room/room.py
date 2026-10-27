# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import NamedTuple, cast

import procfunc as pf

from infinigen2.lighting import sky_lighting
from infinigen2.objects import window
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room.bathroom_setup import bathroom_setup_rand
from infinigen2.scenes.room.bed_setup import bed_setup_multi_rand, bed_setup_rand
from infinigen2.scenes.room.ceiling_features import ceiling_feature_rand
from infinigen2.scenes.room.cocktail_table_setup import table_cocktail_setup_wall_rand
from infinigen2.scenes.room.decoration_objects import (
    DecorationObjectsResult,
    decorate_floor_objects_rand,
    decorate_surface_objects_rand,
    decoration_collection_primitives_and_real_rand,
    decoration_collection_primitives_rand,
    scatter_small_objects_on_containers,
    scatter_small_objects_on_support_tops,
)
from infinigen2.scenes.room.desk_setup import desk_setup_rand
from infinigen2.scenes.room.dining_table_setup import (
    DiningTableSetupResult,
    table_dining_setup_rand,
    table_dining_setup_wall_rand,
)
from infinigen2.scenes.room.kitchen_setup import kitchen_setup_rand
from infinigen2.scenes.room.room_shape import (
    RoomShapeResult,
    room_shape_rand,
)
from infinigen2.scenes.room.skirting import skirting_rand
from infinigen2.scenes.room.sofa_setup import (
    sofa_setup_centered_rand,
    sofa_setup_wall_rand,
    tv_setup_wall_rand,
)
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    WallResult,
    WallSurface,
    extrude_for_thickness,
    name_objects,
    overlap_wall_plane_edges,
    resolve_wall_inputs,
    wall_plain_rand,
)
from infinigen2.scenes.room.wall_cutouts import (
    wall_cubby_rand,
    wall_doors_rand,
    wall_full_window_rand,
    wall_painting_grid_rand,
    wall_windows_rand,
)
from infinigen2.scenes.room.wall_mounts import (
    wall_board_shelf_rand,
    wall_storage_flush_rand,
)
from infinigen2.scenes.room.wall_storage_setup import wall_storage_setup_rand
from infinigen2.shaders.functionality_lists import wall_material_rand
from infinigen2.util.scene_cleanup import delete_object

__all__ = [
    "RoomResult",
    "decorate_room_small_objects_rand",
    "room_bathroom_rand",
    "room_bedroom_rand",
    "room_diningroom_rand",
    "room_kitchen_rand",
    "room_livingroom_rand",
    "room_rand",
    "room_unfurnished_rand",
    "room_walls_rand",
    "wall_arrangement_rand",
]

logger = logging.getLogger(__name__)

ROOM_WALL_THICKNESS = 0.1


class RoomResult(NamedTuple):
    """Containers expose internal upward surfaces; supports accept objects on top.
    One object may provide both capabilities."""

    all_objects: list[pf.MeshObject]
    cameras: list[pf.CameraObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet
    floor: pf.MeshObject
    dimensions: pf.Vector
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]
    wall_planes: list[pf.MeshObject]
    plain_wall_planes: list[pf.MeshObject] = []


@pf.tracer.grammar
def wall_arrangement_rand(
    rng: pf.RNG,
    wall: WallSurface | None = None,
    window_obj: pf.MeshObject | None = None,
    window_portal: pf.LightObject | None = None,
    top_profile_height: float = 0.0,
    bottom_profile_height: float = 0.0,
    wall_material: pf.Material | None = None,
    window_spacing: float | None = None,
    window_bottom: float | None = None,
    wall_thickness: float = 0.05,
    colliders: ccol.CollisionSet | None = None,
    window_reveal_depth: float | None = None,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)

    def plain(rng, wall, wall_material):
        return wall_plain_rand(rng, wall, wall_material, wall_thickness=wall_thickness)

    def windows(rng, wall, wall_material):
        return wall_windows_rand(
            rng,
            wall,
            wall_material,
            window_obj=window_obj,
            window_portal=window_portal,
            top_profile_height=top_profile_height,
            bottom_profile_height=bottom_profile_height,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            reveal_depth=window_reveal_depth,
            wall_thickness=wall_thickness,
        )

    def painting_grid(rng, wall, wall_material):
        return wall_painting_grid_rand(
            rng,
            wall,
            wall_material,
            wall_thickness=wall_thickness,
            colliders=colliders,
        )

    def board_shelf(rng, wall, wall_material):
        return wall_board_shelf_rand(
            rng,
            wall,
            wall_material,
            wall_thickness=wall_thickness,
            colliders=colliders,
        )

    def storage_flush(rng, wall, wall_material):
        return wall_storage_flush_rand(
            rng,
            wall,
            wall_material,
            wall_thickness=wall_thickness,
            colliders=colliders,
        )

    def cubby(rng, wall, wall_material):
        return wall_cubby_rand(rng, wall, wall_material, wall_thickness=wall_thickness)

    def full_wall_window(rng, wall, wall_material):
        return wall_full_window_rand(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    rng_choice, rng_feature = rng.spawn(2)
    option = pf.control.choice(
        rng_choice,
        [
            (plain, 0.5),
            (windows, 3.0),
            (painting_grid, 1.0),
            (board_shelf, 1.5),
            (storage_flush, 1.0),
            (cubby, 0.5),
            (full_wall_window, 0.6),
        ],
    )
    return option(rng=rng_feature, wall=wall, wall_material=wall_material)


def _decoration_objects(result: WallResult) -> list[pf.MeshObject]:
    return [obj for objects in result.decorations.values() for obj in objects]


@pf.tracer.grammar
def room_walls_rand(
    rng: pf.RNG,
    shape: RoomShapeResult,
    built_walls: list[WallResult],
    open_walls: list[WallSurface],
    wall_materials: list[pf.Material],
) -> WallResult:
    """Arrange `open_walls` and merge them with the already `built_walls`. Mounted
    features hitting earlier walls' decorations are dropped by each feature."""
    rng_window, rng_walls, rng_window_shape = rng.spawn(3)

    pf.ops.object.set_material(
        shape.walls,
        surface=wall_materials[0].surface,
        displacement=wall_materials[0].displacement,
    )
    pf.ops.modifier.subdivide_surface(
        shape.walls, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )
    corner_back = extrude_for_thickness(shape.walls, ROOM_WALL_THICKNESS)

    edge_gap_pct = 0.10
    edge_gap = edge_gap_pct * shape.dimensions.z
    usable_height = shape.dimensions.z - 2.0 * edge_gap
    window_height_pct = pf.random.clip_gaussian(
        rng_window, 0.75, 0.2, 0.6, 1.0 - 2.0 * edge_gap_pct
    )
    window_height = shape.dimensions.z * window_height_pct
    # window size is independent of walls; too-wide windows fall back to plain wall
    max_window_width = max(2.0, max(shape.dimensions.x, shape.dimensions.y) - 0.5)
    window_width = (
        1.0 + (max_window_width - 1.0) * pf.random.uniform(rng_window, 0.0, 1.0) ** 2
    )
    window_dimensions = window.window_dimensions_rand(
        rng_window, width=window_width, height=window_height
    )
    window_result = window.window_composite_rand(
        rng_window_shape, dimensions=window_dimensions
    )
    window_obj = window_result.mesh
    window_portal = window_result.light
    wall_offset = pf.Vector((0.0, window_dimensions.y * -0.5, 0.0))
    pf.ops.object.set_transform(window_obj, location=wall_offset)
    pf.ops.mesh.transform_apply(window_obj)
    if window_portal is not None:
        pf.ops.object.set_transform(
            window_portal,
            location=window_portal.item().location + wall_offset,
        )

    _depth, _width, _height = window_obj.item().dimensions
    wmin, _wmax = pf.ops.attr.bbox_min_max(window_obj)
    free_height = usable_height - _height
    window_bottom_pct = pf.random.clip_gaussian(rng_window, 0.7, 0.15, 0.35, 0.85)
    window_bottom = edge_gap + free_height * window_bottom_pct - wmin[2]
    window_spacing = pf.random.uniform(rng_window, 0.1, 0.25) * _width

    results = list(built_walls)
    colliders = ccol.collision_set(
        [o for r in built_walls if r.colliders is not None for o in r.colliders.objs]
    )
    for wall, rng_wall in zip(
        open_walls, rng_walls.spawn(len(open_walls)), strict=True
    ):
        rng_material, rng_arrangement = rng_wall.spawn(2)
        material = pf.control.choice(
            rng_material, [(wall_materials[0], 3.0), (wall_materials[1], 1.0)]
        )
        arrangement = wall_arrangement_rand(
            rng_arrangement,
            wall,
            window_obj,
            window_portal,
            top_profile_height=window_result.profile.top_profile_height,
            bottom_profile_height=window_result.profile.bottom_profile_height,
            wall_material=material,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            wall_thickness=ROOM_WALL_THICKNESS,
            colliders=colliders,
        )
        colliders = _with_objects(colliders, _decoration_objects(arrangement))
        results.append(arrangement)

    decorations: dict[str, list[pf.MeshObject]] = {}
    for result in results:
        for kind, objs in result.decorations.items():
            decorations.setdefault(kind, []).extend(objs)
    for kind, objs in sorted(decorations.items()):
        name_objects(objs, kind)
        logger.info(f"Created {len(objs)} wall {kind} objects")

    wall_planes = [obj for result in results for obj in result.wall_planes]
    backs = [corner_back] + [obj for result in results for obj in result.backs]
    sills = [obj for result in results for obj in result.sills]
    name_objects([shape.walls], "room_wall_corners")
    name_objects(wall_planes, "room_wall")
    name_objects(backs, "room_wall_back")
    name_objects(sills, "room_wall_sill")

    return WallResult(
        all_objects=[shape.walls, corner_back]
        + [obj for result in results for obj in result.all_objects],
        wall_planes=wall_planes,
        corner_walls=[shape.walls],
        backs=backs,
        sills=sills,
        storage_containers=[o for result in results for o in result.storage_containers],
        supports=[o for result in results for o in result.supports],
        lights=[light for result in results for light in result.lights],
        decorations=decorations,
        storages=[o for result in results for o in result.storages],
        colliders=colliders,
    )


def _livingroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 22.0, 5.0, 15.0, 30.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.3, 0.25, 1.0, 1.8)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.7, 0.2, 2.5, 3.2)
    return pf.Vector((width, depth, height))


def _diningroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 16.0, 4.0, 11.5, 25.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.2, 0.2, 1.0, 1.35)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.65, 0.2, 2.5, 3.2)
    return pf.Vector((width, depth, height))


def _bedroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 15.5, 3.0, 11.5, 22.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.18, 0.15, 1.0, 1.45)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.6, 0.2, 2.5, 3.2)
    return pf.Vector((width, depth, height))


def _bathroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_width_scale, rng_height = rng.spawn(4)
    area = pf.random.clip_gaussian(rng_area, 5.2, 1.3, 4.1, 9.0)
    maximum_aspect = min(2.1, area / 1.5**2)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.55, 0.25, 1.2, maximum_aspect)
    base_width = math.sqrt(area / aspect)
    depth = area / base_width
    width = base_width * pf.random.uniform(rng_width_scale, 1.0, 1.33)
    height = pf.random.clip_gaussian(rng_height, 2.45, 0.15, 2.2, 2.8)
    return pf.Vector((width, depth, height))


def _kitchen_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Two-sided galleys (2.3 m: two counters plus an aisle) through open plan."""
    rng_width, rng_aspect, rng_height = rng.spawn(3)
    width = pf.random.clip_gaussian(rng_width, 3.4, 0.9, 2.3, 6.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.5, 0.45, 1.0, 2.8)
    depth = width * aspect
    height = pf.random.clip_gaussian(rng_height, 2.6, 0.2, 2.4, 3.0)
    return pf.Vector((width, depth, height))


@pf.tracer.grammar
def room_unfurnished_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
    n_plain_walls: int = 0,
    plain_wall_offset: int = 0,
) -> RoomResult:
    """`n_plain_walls` non-door walls, counted from `plain_wall_offset`, stay
    featureless and are returned as `plain_wall_planes`."""
    del frame_start, frame_end
    (
        rng_shape,
        rng_materials,
        rng_door,
        rng_ceiling_generate,
        rng_ceiling_lights,
        rng_walls,
        rng_skirting,
        rng_sky,
        rng_plain,
    ) = rng.spawn(9)

    shape = room_shape_rand(rng_shape, dimensions=dimensions)
    vec_wall = pf.nodes.shader.coord().uv
    wall_materials = [wall_material_rand(r, vec_wall) for r in rng_materials.spawn(2)]
    door_idx = pf.random.randint(rng_door, 0, len(shape.flat_walls))
    door = wall_doors_rand(
        rng_door,
        shape.flat_walls[door_idx],
        wall_materials[0],
        wall_thickness=ROOM_WALL_THICKNESS,
    )
    open_walls = shape.flat_walls[:door_idx] + shape.flat_walls[door_idx + 1 :]
    ceiling = ceiling_feature_rand(rng_ceiling_generate, shape)
    offset = plain_wall_offset % len(open_walls)
    rotated = open_walls[offset:] + open_walls[:offset]
    plain = rotated[:n_plain_walls]
    plain_walls = []
    for wall, r in zip(plain, rng_plain.spawn(len(plain)), strict=True):
        plain_walls.append(
            wall_plain_rand(
                r, wall, wall_materials[0], wall_thickness=ROOM_WALL_THICKNESS
            )
        )

    walls = room_walls_rand(
        rng_walls, shape, [door] + plain_walls, rotated[n_plain_walls:], wall_materials
    )
    skirting = skirting_rand(rng_skirting, walls=walls.wall_planes + [shape.walls])
    for wall_plane in walls.wall_planes:
        overlap_wall_plane_edges(wall_plane, ROOM_WALL_THICKNESS)
    sky = sky_lighting.sky_hosek_wilkie_with_sun_lamp_rand(rng_sky)

    structure = (
        walls.all_objects + [shape.floor, ceiling.ceiling] + ceiling.light_meshes
    )
    lights = (
        pf.control.choice(
            rng_ceiling_lights,
            [(ceiling.lights, 5.0), ([] if walls.lights else ceiling.lights, 1.0)],
        )
        + walls.lights
        + sky.lights
    )
    return RoomResult(
        all_objects=structure + ceiling.backs + ceiling.sills + skirting,
        cameras=shape.cameras,
        lights=lights,
        colliders=_with_objects(cast(ccol.CollisionSet, walls.colliders), structure),
        floor=shape.floor,
        dimensions=shape.dimensions,
        storage_containers=walls.storage_containers,
        supports=walls.supports,
        storages=walls.storages,
        wall_planes=walls.wall_planes,
        plain_wall_planes=[p for w in plain_walls for p in w.wall_planes],
    )


def _with_objects(
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
) -> ccol.CollisionSet:
    collider_items = {obj.item() for obj in colliders.objs}
    new_objects = [obj for obj in objects if obj.item() not in collider_items]
    return ccol.collision_set(
        colliders.objs + new_objects,
        cache=colliders,
    )


@pf.tracer.grammar
def decorate_room_small_objects_rand(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    containers: list[pf.MeshObject],
    support_tops: list[pf.MeshObject],
    storages: list[pf.MeshObject],
    nonstorage_collection: pf.Collection | None = None,
    storage_collection: pf.Collection | None = None,
) -> DecorationObjectsResult:
    (
        rng_collection_primitives,
        rng_collection_primitives_and_real,
        rng_storage_support_density,
        rng_support_storages,
        rng_storage_containers,
        rng_nonstorage_support_density,
        rng_nonstorage_support_fraction,
        rng_support_nonstorages,
        rng_nonstorage_container_density,
        rng_nonstorage_container_fraction,
        rng_nonstorage_containers,
    ) = rng.spawn(11)
    storage_objects = set(storages)
    support_storages = [obj for obj in support_tops if obj in storage_objects]
    support_nonstorages = [obj for obj in support_tops if obj not in storage_objects]
    container_storages = [obj for obj in containers if obj in storage_objects]
    container_nonstorages = [obj for obj in containers if obj not in storage_objects]

    if nonstorage_collection is None:
        nonstorage_collection = decoration_collection_primitives_rand(
            rng_collection_primitives
        )
    if storage_collection is None:
        storage_collection = decoration_collection_primitives_and_real_rand(
            rng_collection_primitives_and_real
        )
    storage_support_density = pf.random.clip_gaussian(
        rng_storage_support_density, 1.05, 0.35, 0.0, 2.5
    )
    placed_support_storages, colliders = scatter_small_objects_on_support_tops(
        rng_support_storages,
        support_storages,
        colliders,
        collection=storage_collection,
        density=storage_support_density,
    )
    (
        rng_storage_container_density,
        rng_storage_container_spacing,
        rng_storage_container_scatter,
        rng_storage_container_fraction,
    ) = rng_storage_containers.spawn(4)
    storage_container_density = pf.random.clip_gaussian(
        rng_storage_container_density, 2.5, 0.5, 1.5, 3.5
    )
    storage_container_spacing = pf.random.uniform(
        rng_storage_container_spacing, 0.5, 0.8
    )
    storage_container_fraction = pf.random.uniform(
        rng_storage_container_fraction, 0.5, 1.0
    )
    placed_storage_containers, colliders = scatter_small_objects_on_containers(
        rng_storage_container_scatter,
        container_storages,
        colliders,
        collection=storage_collection,
        density=storage_container_density,
        fraction=storage_container_fraction,
        spacing_factor=storage_container_spacing,
    )
    nonstorage_support_density = pf.random.clip_gaussian(
        rng_nonstorage_support_density, 0.525, 0.175, 0.0, 1.25
    )
    nonstorage_support_fraction = pf.random.uniform(
        rng_nonstorage_support_fraction, 0.0, 1.0
    )
    placed_support_nonstorages, colliders = scatter_small_objects_on_support_tops(
        rng_support_nonstorages,
        support_nonstorages,
        colliders,
        collection=nonstorage_collection,
        density=nonstorage_support_density,
        fraction=nonstorage_support_fraction,
    )
    nonstorage_container_density = pf.random.clip_gaussian(
        rng_nonstorage_container_density, 0.75, 0.25, 0.0, 1.25
    )
    nonstorage_container_fraction = pf.random.uniform(
        rng_nonstorage_container_fraction, 0.0, 1.0
    )
    placed_nonstorage_containers, colliders = scatter_small_objects_on_containers(
        rng_nonstorage_containers,
        container_nonstorages,
        colliders,
        collection=nonstorage_collection,
        density=nonstorage_container_density,
        fraction=nonstorage_container_fraction,
    )
    placed = (
        placed_support_storages
        + placed_storage_containers
        + placed_support_nonstorages
        + placed_nonstorage_containers
    )
    return DecorationObjectsResult(objects + placed, [], colliders)


@pf.tracer.grammar
def room_livingroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_sofa,
        rng_tv,
        rng_desk,
        rng_storage,
        rng_decor,
    ) = rng.spawn(7)
    if dimensions is None:
        dimensions = _livingroom_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    rng_sofa_choice, rng_sofa_setup = rng_sofa.spawn(2)
    sofa_func = pf.control.choice(
        rng_sofa_choice,
        [
            (
                lambda sofa_rng: sofa_setup_wall_rand(
                    sofa_rng,
                    wall_planes=room.wall_planes,
                    bbox_min=pf.Vector((0.0, 0.0, 0.0)),
                    colliders=room.colliders,
                ),
                3.0,
            ),
            (
                lambda sofa_rng: sofa_setup_centered_rand(
                    sofa_rng,
                    bbox_min=pf.Vector((0.0, 0.0, 0.0)),
                    bbox_max=dimensions,
                    colliders=room.colliders,
                ),
                7.0,
            ),
        ],
    )
    sofa_setup = sofa_func(rng_sofa_setup)
    name_objects([result.mesh for result in sofa_setup.sofas], "sofa")
    colliders = _with_objects(room.colliders, sofa_setup.all_objects)
    tv_setups = tv_setup_wall_rand(
        rng_tv, wall_planes=room.wall_planes, colliders=colliders
    )
    tv_objects = [obj for setup in tv_setups for obj in setup.all_objects]
    tv_storages = [setup.mesh for setup in tv_setups]
    colliders = _with_objects(colliders, tv_objects)

    desk_objects = []
    desk_containers = []
    desk_supports = []
    desk_storages = []
    rng_desk_active, rng_desk_setup = rng_desk.spawn(2)
    if pf.control.choice(rng_desk_active, [(False, 2.0), (True, 1.0)]):
        desk_setup = desk_setup_rand(
            rng_desk_setup,
            wall_planes=room.wall_planes,
            colliders=colliders,
        )
        if desk_setup is not None:
            name_objects([desk_setup.desk], "desk")
            name_objects([desk_setup.chair], "desk_chair")
            desk_objects = desk_setup.all_objects
            desk_containers = desk_setup.storage_containers
            desk_supports = desk_setup.supports
            desk_storages = desk_setup.storages
            colliders = _with_objects(colliders, desk_objects)

    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        colliders=colliders,
    )
    furniture = (
        sofa_setup.all_objects + tv_objects + desk_objects + storage_setup.all_objects
    )
    storage_containers = (
        room.storage_containers
        + sofa_setup.storage_containers
        + tv_storages
        + desk_containers
        + storage_setup.storage_containers
    )
    supports = (
        room.supports
        + sofa_setup.supports
        + tv_storages
        + desk_supports
        + storage_setup.supports
    )
    storages = (
        room.storages
        + sofa_setup.storages
        + tv_storages
        + desk_storages
        + storage_setup.storages
    )
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=room.all_objects + furniture,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=supports,
        storages=storages,
    )
    small_result = decorate_room_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=supports,
        storages=storages,
    )

    all_objects = small_result.all_objects
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + floor_result.lights + surface_result.lights,
        colliders=small_result.colliders,
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=storage_containers,
        supports=supports,
        storages=storages,
        wall_planes=room.wall_planes,
    )


@pf.tracer.grammar
def room_diningroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_dining,
        rng_storage,
        rng_decor,
    ) = rng.spawn(5)
    if dimensions is None:
        dimensions = _diningroom_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    dining_setup = table_dining_setup_rand(
        rng_dining,
        wall_planes=room.wall_planes,
        bbox_min=pf.Vector((0.0, 0.0, 0.0)),
        bbox_max=dimensions,
        colliders=room.colliders,
    )
    name_objects([r.mesh for r in dining_setup.dining_tables], "dining_table")
    name_objects(dining_setup.dining_chairs, "dining_chair")
    colliders = _with_objects(room.colliders, dining_setup.all_objects)
    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        colliders=colliders,
    )
    furniture = dining_setup.all_objects + storage_setup.all_objects
    storage_containers = (
        room.storage_containers
        + dining_setup.storage_containers
        + storage_setup.storage_containers
    )
    supports = room.supports + dining_setup.supports + storage_setup.supports
    storages = room.storages + dining_setup.storages + storage_setup.storages
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=room.all_objects + furniture,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=supports,
        storages=storages,
    )
    small_result = decorate_room_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=supports,
        storages=storages,
    )

    all_objects = small_result.all_objects
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + floor_result.lights + surface_result.lights,
        colliders=small_result.colliders,
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=storage_containers,
        supports=supports,
        storages=storages,
        wall_planes=room.wall_planes,
    )


@pf.tracer.grammar
def room_bedroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_bed,
        rng_sofa,
        rng_desk,
        rng_storage,
        rng_decor,
        rng_tv,
    ) = rng.spawn(8)
    if dimensions is None:
        dimensions = _bedroom_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    def single_bed():
        return [
            bed_setup_rand(
                rng_bed_setup,
                wall_planes=room.wall_planes,
                bbox_min=pf.Vector((0.0, 0.0, 0.0)),
                bbox_max=dimensions,
                colliders=room.colliders,
            )
        ]

    def multi_bed():
        return bed_setup_multi_rand(
            rng_bed_setup,
            wall_planes=room.wall_planes,
            bbox_min=pf.Vector((0.0, 0.0, 0.0)),
            bbox_max=dimensions,
            colliders=room.colliders,
        )

    rng_bed_choice, rng_bed_setup = rng_bed.spawn(2)
    bed_func = pf.control.choice(rng_bed_choice, [(single_bed, 1.0), (multi_bed, 0.5)])
    bed_setups = bed_func()
    bed_objects = [obj for setup in bed_setups for obj in setup.all_objects]
    bed_containers = [obj for setup in bed_setups for obj in setup.storage_containers]
    bed_supports = [obj for setup in bed_setups for obj in setup.supports]
    bed_storages = [obj for setup in bed_setups for obj in setup.storages]
    bed_lights = [light for setup in bed_setups for light in setup.lights]
    colliders = bed_setups[-1].colliders

    def no_sofa():
        return None

    def wall_sofa():
        return sofa_setup_wall_rand(
            rng_sofa_setup,
            wall_planes=room.wall_planes,
            bbox_min=pf.Vector((0.0, 0.0, 0.0)),
            colliders=colliders,
        )

    rng_sofa_active, rng_sofa_setup = rng_sofa.spawn(2)
    sofa_func = pf.control.choice(
        rng_sofa_active,
        [(no_sofa, 2.0), (wall_sofa, 1.0)],
    )
    sofa_setup = sofa_func()
    sofa_objects = []
    sofa_containers = []
    sofa_supports = []
    sofa_storages = []
    if sofa_setup is not None:
        name_objects([result.mesh for result in sofa_setup.sofas], "sofa")
        sofa_objects = sofa_setup.all_objects
        sofa_containers = sofa_setup.storage_containers
        sofa_supports = sofa_setup.supports
        sofa_storages = sofa_setup.storages
        colliders = _with_objects(colliders, sofa_objects)

    def no_tv():
        return []

    def wall_tv():
        return tv_setup_wall_rand(
            rng_tv_setup, wall_planes=room.wall_planes, colliders=colliders
        )

    rng_tv_active, rng_tv_setup = rng_tv.spawn(2)
    tv_func = pf.control.choice(rng_tv_active, [(no_tv, 1.0), (wall_tv, 1.0)])
    tv_setups = tv_func()
    tv_objects = [obj for setup in tv_setups for obj in setup.all_objects]
    tv_storages = [setup.mesh for setup in tv_setups]
    colliders = _with_objects(colliders, tv_objects)

    desk_objects = []
    desk_containers = []
    desk_supports = []
    desk_storages = []
    rng_desk_active, rng_desk_setup = rng_desk.spawn(2)
    if pf.control.choice(rng_desk_active, [(False, 1.0), (True, 1.0)]):
        desk_setup = desk_setup_rand(
            rng_desk_setup,
            wall_planes=room.wall_planes,
            colliders=colliders,
        )
        if desk_setup is not None:
            name_objects([desk_setup.desk], "desk")
            name_objects([desk_setup.chair], "desk_chair")
            desk_objects = desk_setup.all_objects
            desk_containers = desk_setup.storage_containers
            desk_supports = desk_setup.supports
            desk_storages = desk_setup.storages
            colliders = _with_objects(colliders, desk_objects)

    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        colliders=colliders,
    )
    furniture = (
        bed_objects
        + sofa_objects
        + tv_objects
        + desk_objects
        + storage_setup.all_objects
    )

    storage_containers = (
        room.storage_containers
        + bed_containers
        + sofa_containers
        + tv_storages
        + desk_containers
        + storage_setup.storage_containers
    )
    supports = (
        room.supports
        + bed_supports
        + sofa_supports
        + tv_storages
        + desk_supports
        + storage_setup.supports
    )
    storages = (
        room.storages
        + bed_storages
        + sofa_storages
        + tv_storages
        + desk_storages
        + storage_setup.storages
    )
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=room.all_objects + furniture,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=supports,
        storages=storages,
    )
    small_result = decorate_room_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=supports,
        storages=storages,
    )

    all_objects = small_result.all_objects
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + bed_lights + floor_result.lights + surface_result.lights,
        colliders=small_result.colliders,
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=storage_containers,
        supports=supports,
        storages=storages,
        wall_planes=room.wall_planes,
    )


@pf.tracer.grammar
def room_bathroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_setup,
        rng_storage,
        rng_decor,
    ) = rng.spawn(5)
    if dimensions is None:
        dimensions = _bathroom_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    bathroom_setup = bathroom_setup_rand(
        rng_setup,
        wall_planes=room.wall_planes,
        bbox_min=pf.Vector((0.0, 0.0, 0.0)),
        bbox_max=dimensions,
        colliders=room.colliders,
    )
    colliders = bathroom_setup.colliders
    storage_objects = []
    storage_containers = []
    wall_setup_supports = []
    storage_storages = []
    rng_storage_active, rng_storage_setup = rng_storage.spawn(2)
    if pf.control.choice(rng_storage_active, [(False, 1.0), (True, 1.0)]):
        storage_setup = wall_storage_setup_rand(
            rng_storage_setup,
            wall_planes=room.wall_planes,
            colliders=bathroom_setup.colliders,
        )
        storage_objects = storage_setup.all_objects
        storage_containers = storage_setup.storage_containers
        wall_setup_supports = storage_setup.supports
        storage_storages = storage_setup.storages
        colliders = storage_setup.colliders
    furniture = bathroom_setup.all_objects + storage_objects

    containers = (
        room.storage_containers + bathroom_setup.storage_containers + storage_containers
    )
    supports = room.supports + bathroom_setup.supports + wall_setup_supports
    clearances = bathroom_setup.temporary_objects
    storages = room.storages + bathroom_setup.storages + storage_storages
    rng_surface, rng_collection_primitives, rng_small = rng_decor.spawn(3)
    collection_primitives = decoration_collection_primitives_rand(
        rng_collection_primitives
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=room.all_objects + furniture,
        colliders=ccol.collision_set(
            colliders.objs + clearances,
            cache=colliders,
        ),
        support_tops=supports,
        storages=storages,
    )
    small_result = decorate_room_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=containers,
        support_tops=supports,
        storages=storages,
        nonstorage_collection=collection_primitives,
        storage_collection=collection_primitives,
    )
    clearance_items = {clearance.item() for clearance in clearances}
    colliders = ccol.collision_set(
        [
            obj
            for obj in small_result.colliders.objs
            if obj.item() not in clearance_items
        ],
        cache=small_result.colliders,
    )
    for clearance in clearances:
        delete_object(clearance.item())

    all_objects = small_result.all_objects
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + surface_result.lights,
        colliders=colliders,
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=containers,
        supports=supports,
        storages=storages,
        wall_planes=room.wall_planes,
    )


def _counter_keepout(counter: pf.MeshObject, margin: float) -> pf.MeshObject:
    """A floor-to-counter box `margin` wider than `counter` on every side."""
    lo, hi = pf.ops.attr.bbox_min_max(counter, global_coords=True)
    box = pf.ops.primitives.mesh_cube(size=1.0)
    size = (hi[0] - lo[0] + 2.0 * margin, hi[1] - lo[1] + 2.0 * margin, hi[2])
    pf.ops.mesh.transform(box, scale=size)
    centre = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2] / 2.0)
    pf.ops.object.set_transform(box, location=centre)
    return box


@pf.tracer.grammar
def room_kitchen_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_setup,
        rng_table_choice,
        rng_table,
        rng_plain,
        rng_decor,
        rng_storage,
    ) = rng.spawn(8)
    if dimensions is None:
        dimensions = _kitchen_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
        n_plain_walls=pf.random.randint(rng_plain, 1, 4),
        plain_wall_offset=pf.random.randint(rng_plain, 0, 4),
    )
    plain_walls = set(room.plain_wall_planes)
    kitchen_setup = kitchen_setup_rand(
        rng_setup,
        wall_planes=[p for p in room.wall_planes if p not in plain_walls],
        reserved_wall_planes=room.plain_wall_planes,
        bbox_min=pf.Vector((0.0, 0.0, 0.0)),
        bbox_max=dimensions,
        colliders=room.colliders,
        wall_overlap=ROOM_WALL_THICKNESS,
    )
    table_fn, label = pf.control.choice(
        rng_table_choice,
        [
            ((table_dining_setup_wall_rand, "dining"), 0.6),
            ((table_cocktail_setup_wall_rand, "cocktail"), 0.2),
            (
                (lambda *_: DiningTableSetupResult([], [], [], [], [], []), "dining"),
                0.2,
            ),
        ],
    )
    keepouts = [_counter_keepout(c, 0.91) for c in kitchen_setup.countertops]
    table_colliders = _with_objects(room.colliders, keepouts)
    tables = table_fn(rng_table, room.wall_planes, dimensions, table_colliders)
    for keepout in keepouts:
        delete_object(keepout.item())
    name_objects([r.mesh for r in tables.dining_tables], f"{label}_table")
    name_objects(tables.dining_chairs, f"{label}_chair")
    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        colliders=_with_objects(kitchen_setup.colliders, tables.all_objects),
    )
    furniture = (
        kitchen_setup.all_objects + tables.all_objects + storage_setup.all_objects
    )
    storage_containers = (
        room.storage_containers
        + kitchen_setup.storage_containers
        + tables.storage_containers
        + storage_setup.storage_containers
    )
    supports = (
        room.supports
        + kitchen_setup.supports
        + tables.supports
        + storage_setup.supports
    )
    storages = (
        room.storages
        + kitchen_setup.storages
        + tables.storages
        + storage_setup.storages
    )
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=room.all_objects + furniture,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=supports,
        storages=storages,
    )
    small_result = decorate_room_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=supports,
        storages=storages,
    )
    return RoomResult(
        all_objects=small_result.all_objects,
        cameras=room.cameras,
        lights=room.lights + floor_result.lights + surface_result.lights,
        colliders=small_result.colliders,
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=storage_containers,
        supports=supports,
        storages=storages,
        wall_planes=room.wall_planes,
        plain_wall_planes=room.plain_wall_planes,
    )


@pf.tracer.grammar
def room_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    rng_choice, rng_room = rng.spawn(2)
    room_func = pf.control.choice(
        rng_choice,
        [
            (room_livingroom_rand, 1.0),
            (room_bathroom_rand, 1.0),
            (room_diningroom_rand, 1.0),
            (room_bedroom_rand, 1.0),
            (room_kitchen_rand, 1.0),
        ],
    )
    return room_func(
        rng_room,
        dimensions=dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )
