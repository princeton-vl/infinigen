# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

import logging

import bmesh
import bpy
import gin
import numpy as np
import shapely
import shapely.affinity
import trimesh.convex
from numpy.random import uniform
from shapely import Point
from shapely.ops import nearest_points
from tqdm import trange
from trimesh.transformations import translation_matrix

import infinigen.core.constraints.example_solver.room.constants as constants
from infinigen.assets.materials import plaster, tile
from infinigen.assets.objects.elements import PillarFactory, random_staircase_factory
from infinigen.assets.objects.elements.doors import random_door_factory
from infinigen.assets.objects.windows import WindowFactory
from infinigen.assets.utils.decorate import (
    read_area,
    read_co,
    read_edge_direction,
    read_edge_length,
    read_edges,
    remove_edges,
    remove_faces,
    remove_vertices,
)
from infinigen.assets.utils.object import obj2trimesh
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.room.configs import (
    PILLAR_ROOM_TYPES,
    ROOM_FLOORS,
    ROOM_WALLS,
)
from infinigen.core.constraints.example_solver.room.constants import (
    DOOR_WIDTH,
    WALL_HEIGHT,
    WALL_THICKNESS,
)
from infinigen.core.constraints.example_solver.room.types import (
    RoomType,
    get_room_level,
    get_room_type,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)


def split_rooms(rooms_meshed: list[bpy.types.Object]):
    extract_tags = {
        "wall": {t.Subpart.Wall, t.Subpart.Visible},
        "floor": {t.Subpart.SupportSurface, t.Subpart.Visible},
        "ceiling": {t.Subpart.Ceiling, t.Subpart.Visible},
    }

    meshes = {
        n: [tagging.extract_tagged_faces(r, tags) for r in rooms_meshed]
        for n, tags in extract_tags.items()
    }

    for k, ms in meshes.items():
        m2delete = []
        for m in ms:
            if m.name.startswith("vert"):
                butil.select_none()
                butil.delete(m)
                m2delete.append(m)
        for m in m2delete:
            ms.remove(m)

    meshes["exterior"] = [
        tagging.extract_mask(r, 1 - tagging.tagged_face_mask(r, t.Subpart.Visible))
        for r in rooms_meshed
    ]

    for n, objs in meshes.items():
        for o in objs:
            o.name = o.name.split(".")[0] + f".{n}"
        butil.origin_set(objs, "ORIGIN_GEOMETRY", center="MEDIAN")

    meshes = {
        n: butil.put_in_collection(objs, "unique_assets:room_" + n)
        for n, objs in meshes.items()
    }

    return meshes


def room_walls(wall_objs: list[bpy.types.Object]):
    wall_fns = list(rg(ROOM_WALLS[get_room_type(r.name)]) for r in wall_objs)

    logger.debug(
        f"{room_walls.__name__} adding materials to {len(wall_objs)=}, using {len(wall_fns)=}"
    )

    for wall_fn in set(wall_fns):
        rooms_ = [o for o, w in zip(wall_objs, wall_fns) if w == wall_fn]
        shape = np.random.choice(["square", "rectangle", "hexagon"])
        kwargs = dict(vertical=True, alternating=False, shape=shape)
        if wall_fn in [tile, plaster]:
            indices = np.random.randint(0, 3, len(rooms_))
            for i in range(3):
                rooms__ = [r for r, j in zip(rooms_, indices) if j == i]
                wall_fn.apply(rooms__, **kwargs)
        else:
            wall_fn.apply(rooms_, **kwargs)


def room_ceilings(ceilings: list[bpy.types.Object]):
    logger.debug(f"{room_ceilings.__name__} adding materials to {len(ceilings)=}")
    plaster.apply(ceilings, t.Subpart.Ceiling)


def room_floors(floors: list[bpy.types.Object]):
    floor_fns = list(rg(ROOM_FLOORS[get_room_type(r.name)]) for r in floors)
    logger.debug(
        f"{room_floors.__name__} adding materials to {len(floors)=}, using {len(floor_fns)=}"
    )
    for floor_fn in set(floor_fns):
        rooms_ = [o for o, f in zip(floors, floor_fns) if f == floor_fn]

        if floor_fn in [tile, plaster]:
            indices = np.random.randint(0, 3, len(rooms_))
            for i in range(3):
                rooms__ = [r for r, j in zip(rooms_, indices) if j == i]
                floor_fn.apply(rooms__)
        else:
            floor_fn.apply(rooms_)


@gin.configurable
def populate_doors(
    placeholders: list[bpy.types.Object],
    n_doors=3,
    door_chance=1,
    casing_chance=0.0,
    all_open=False,
):
    factories = [random_door_factory()(np.random.randint(1e7)) for _ in range(3)]

    logger.debug(
        f"{populate_doors.__name__} populating {len(placeholders)=} with {n_doors=} and {len(factories)=}"
    )

    indices = np.random.randint(0, len(factories), len(placeholders))
    col = butil.get_collection("unique_assets:doors")
    casing_col = butil.get_collection("unique_assets:door_casings")

    for i in trange(n_doors, desc="Placing doors"):
        factory = factories[i]
        casing_factory = factory.casing_factory
        doors, casings = [], []
        for j in np.nonzero(indices == i)[0]:
            if uniform() > door_chance:
                continue
            if all_open:
                rot_z = uniform(0.93, 1.93)
            else:
                rot_p = uniform()
                if rot_p < 0.5:
                    rot_z = uniform(0, 0.1)
                elif rot_p < 0.7:
                    rot_z = uniform(0.93, 1.03)
                else:
                    rot_z = uniform(0, 1)
            rot_z *= np.pi / 2

            door = factory(int(j))
            door.parent = placeholders[j]
            door.location = (
                constants.DOOR_WIDTH / 2,
                constants.WALL_THICKNESS / 2,
                -constants.DOOR_SIZE / 2,
            )
            door.rotation_euler[-1] = -rot_z
            doors.append(door)

            if uniform() > casing_chance:
                continue

            casing = casing_factory(int(j))
            casing.parent = placeholders[j]
            casing.location = 0, 0, -constants.DOOR_SIZE / 2
            casings.append(casing)

        factory.finalize_assets(doors)
        butil.put_in_collection(doors, col)

        casing_factory.finalize_assets(casings)
        butil.put_in_collection(casings, casing_col)


def populate_windows(placeholders: list[bpy.types.Object], n_windows=1):
    factories = [WindowFactory(np.random.randint(1e5)) for _ in range(n_windows)]

    logger.debug(
        f"{populate_windows.__name__} populating {len(placeholders)=} with {n_windows=} and {len(factories)=}"
    )

    indices = np.random.randint(0, len(factories), len(placeholders))
    col = butil.get_collection("unique_assets:windows")
    for i in range(n_windows):
        factory = factories[i]
        windows = []
        for j in np.nonzero(indices == i)[0]:
            cutter_dims = placeholders[j].dimensions
            dims = cutter_dims[0], cutter_dims[2], cutter_dims[1] * uniform(0.1, 0.2)
            window = factory(int(j), dimensions=dims)
            window.parent = placeholders[j]
            window.location[1] = -WALL_THICKNESS / 2
            window.rotation_euler[1] = np.pi
            butil.put_in_collection(list(butil.iter_object_tree(window)), col)
        factory.finalize_assets(windows)


def room_stairs(state, rooms_meshed):
    col = butil.get_collection("unique_assets:staircases")
    states = list(
        s for k, s in state.objs.items() if get_room_type(k) == RoomType.Staircase
    )
    contours, doors = [], []
    for s in states:
        doors_ = [
            bpy.data.objects[k]
            for k, o in state.objs.items()
            if any(
                r.relation == cl.CutFrom() and r.target_name == s.obj.name
                for r in o.relations
            )
            and k.startswith("door")
        ]
        contour = shapely.simplify(
            s.contour.buffer(-WALL_THICKNESS / 2, join_style="mitre"), 0.1
        )
        for door in doors_:
            box = shapely.box(
                -DOOR_WIDTH / 2, -DOOR_WIDTH * 2, DOOR_WIDTH / 2, DOOR_WIDTH * 2
            )
            box = shapely.affinity.translate(
                shapely.affinity.rotate(box, door.rotation_euler[-1]), *door.location
            )
            contour = contour.difference(box)
        doors.append(doors_)
        contours.append(contour)
    geoms = []
    for c, c_ in zip(contours[:-1], contours[1:]):
        geom = c.intersection(c_)
        if not geom.geom_type == "Polygon":
            geom = sorted(
                list(g for g in geom.geoms if g.geom_type == "Polygon"),
                key=lambda _: _.area,
            )[-1]
        geoms.append(geom)
    placeholders, offsets, fns = [], [], []
    for _ in trange(100, desc="Generating staircases"):
        butil.delete(placeholders)
        fns = [random_staircase_factory()(np.random.randint(1e7)) for _ in geoms]
        placeholders, mlss, lower, upper = [], [], [], []
        for j, fn in enumerate(fns):
            ph = fn.create_placeholder(i=np.random.randint(1e7))
            placeholders.append(ph)
            polygon = shapely.intersection_all(
                list(
                    shapely.affinity.translate(geoms[j], -x, -y)
                    for x in [ph.bound_box[0][0], ph.bound_box[-1][0]]
                    for y in [ph.bound_box[0][1], ph.bound_box[-1][1]]
                )
            )
            mlss.append(
                polygon.boundary
                if polygon.geom_type == "Polygon"
                else shapely.MultiLineString([p.boundary for p in polygon.geoms])
            )
            x, y, z = read_co(ph).T
            lower.append((x[z < WALL_HEIGHT], y[z < WALL_HEIGHT]))
            upper.append((x[z >= WALL_HEIGHT], y[z >= WALL_HEIGHT]))
        if any(p.is_empty for p in mlss):
            continue
        for _ in range(100):
            offsets = []
            for j, mls in enumerate(mlss):
                p = mls.bounds
                x = uniform(p[0], p[2])
                y = uniform(p[1], p[3])
                p = Point(x, y)
                projected = nearest_points(mls, p)[0]
                if (
                    max(np.abs(p.x - projected.x), np.abs(p.y - projected.y))
                    < constants.STAIRCASE_SNAP
                ):
                    p = projected
                    coords = (
                        mls.coords
                        if mls.geom_type == "LineString"
                        else np.concatenate([ls.coords for ls in mls.geoms])
                    )
                    projected = nearest_points(shapely.MultiPoint(coords), Point(x, y))[
                        0
                    ]
                    if (
                        max(np.abs(p.x - projected.x), np.abs(p.y - projected.y))
                        < constants.STAIRCASE_SNAP
                    ):
                        p = projected
                x, y = p.x, p.y
                placeholders[j].location = x, y, j * WALL_HEIGHT + WALL_THICKNESS / 2
                contains_lower = shapely.contains_xy(
                    contours[j], lower[j][0] + x, lower[j][1] + y
                ).all()
                contains_upper = shapely.contains_xy(
                    contours[j + 1], upper[j][0] + x, upper[j][1] + y
                ).all()
                lower_valid = fns[j].valid_contour((x, y), contours[j], doors[j])
                upper_valid = fns[j].valid_contour(
                    (x, y), contours[j + 1], doors[j + 1], False
                )
                if not (
                    contains_lower and contains_upper and lower_valid and upper_valid
                ):
                    break
                offsets.append((x, y))
            if len(offsets) == len(geoms):
                ts = list(
                    trimesh.convex.convex_hull(
                        obj2trimesh(ph).apply_transform(
                            translation_matrix([*o, WALL_HEIGHT * j])
                        )
                    )
                    for j, (ph, o) in enumerate(zip(placeholders, offsets))
                )
                if all(t.intersection(t_).is_empty for t, t_ in zip(ts[:-1], ts[1:])):
                    break
        if len(offsets) == len(geoms):
            break
    butil.delete(placeholders)
    for j, fn in enumerate(fns):
        s = fn(i=np.random.randint(1e7))
        butil.apply_transform(s, True)
        s.location = *offsets[j], j * WALL_HEIGHT + WALL_THICKNESS / 2
        butil.put_in_collection(s, col)
        cutter = fn.create_cutter(i=np.random.randint(1e7))
        cutter.location = *offsets[j], j * WALL_HEIGHT + WALL_THICKNESS / 2
        for mesh in rooms_meshed:
            if get_room_type(mesh.name) == RoomType.Staircase:
                level = get_room_level(mesh.name)
                if level == j + 1:
                    butil.modify_mesh(
                        mesh,
                        "BOOLEAN",
                        object=cutter,
                        operation="DIFFERENCE",
                        use_self=True,
                        use_hole_tolerant=True,
                    )
                    butil.delete(cutter)
                    m = deep_clone_obj(mesh)
                    m.location = -offsets[j][0], -offsets[j][1], 0
                    butil.apply_transform(m, True)
                    g = fns[j].make_guardrail(m)
                    g.location = s.location
                    g.location[-1] += WALL_HEIGHT
                    butil.put_in_collection(g, col)
    return placeholders


def room_pillars(state: state_def.State, walls: list[bpy.types.Object]):
    col = butil.get_collection("pillars")

    pillar_rooms = [
        s for k, s in state.objs.items() if get_room_type(k) in PILLAR_ROOM_TYPES
    ]

    for s in pillar_rooms:
        factory = PillarFactory(np.random.randint(1e7))
        mesh = next(m for m in walls if m.name.startswith(s.obj.name.split(".")[0]))
        interior = tagging.extract_tagged_faces(mesh, {t.Subpart.Interior})
        remove_faces(interior, read_area(interior) < WALL_THICKNESS / 2 * WALL_HEIGHT)
        selection = (read_edge_length(interior) > WALL_HEIGHT / 2) & (
            np.abs(read_edge_direction(interior))[:, -1] > 0.9
        )
        selection_ = np.bincount(
            read_edges(interior)[selection].reshape(-1),
            minlength=len(interior.data.vertices),
        )
        remove_vertices(interior, selection_ == 0)
        remove_vertices(interior, lambda x, y, z: z > WALL_THICKNESS)
        remove_edges(interior, read_edge_length(interior) < WALL_THICKNESS)
        interiors = butil.split_object(interior)
        for i in interiors:
            with butil.ViewportMode(i, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.dissolve_limited()
                bm = bmesh.from_edit_mesh(i.data)
                geom = [v for v in bm.verts if len(v.link_edges) < 2]
                bmesh.ops.delete(bm, geom=geom)
        interiors_ = [i for i in interiors if len(i.data.vertices) > 0]
        butil.delete([i for i in interiors if len(i.data.vertices) == 0])

        if len(interiors_) == 0:
            return

        with butil.Suppress():
            interior = butil.join_objects(interiors_)

        staircases = list(butil.get_collection("staircases").objects)
        if len(staircases) == 0:
            return

        staircases = np.concatenate(
            [read_co(o) + np.array([o.location]) for o in staircases]
        )
        cos = read_co(interior)
        cos[:, -1] = mesh.location[-1] + WALL_THICKNESS / 2
        cos = cos[
            np.min(
                np.linalg.norm(cos[:, np.newaxis] - staircases[np.newaxis], axis=-1), -1
            )
            > WALL_THICKNESS
        ]
        for co in cos:
            obj = factory(np.random.randint(1e7))
            obj.location = co
            butil.put_in_collection(obj, col)
        butil.delete(interior)
