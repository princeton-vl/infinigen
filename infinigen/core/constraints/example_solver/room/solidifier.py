# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

import logging
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

import bmesh
import bpy
import gin
import numpy as np
from numpy.random import uniform
from shapely import LineString, line_interpolate_point, remove_repeated_points, simplify
from shapely.ops import linemerge

from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.assets.utils.decorate import (
    read_area,
    read_center,
    read_co,
    read_edge_direction,
    read_edge_length,
    remove_faces,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.object import join_objects, new_cube, new_line
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver.geometry import parse_scene
from infinigen.core.constraints.example_solver.room.configs import (
    COMBINED_ROOM_TYPES,
    PANORAMIC_ROOM_TYPES,
    WINDOW_ROOM_TYPES,
)
from infinigen.core.constraints.example_solver.room.constants import (
    DOOR_MARGIN,
    DOOR_SIZE,
    DOOR_WIDTH,
    MAX_WINDOW_LENGTH,
    SEGMENT_MARGIN,
    WALL_HEIGHT,
    WALL_THICKNESS,
    WINDOW_HEIGHT,
    WINDOW_SIZE,
)
from infinigen.core.constraints.example_solver.room.types import (
    RoomGraph,
    RoomType,
    get_room_type,
)
from infinigen.core.constraints.example_solver.room.utils import (
    SIMPLIFY_THRESH,
    WELD_THRESH,
    buffer,
    canonicalize,
    polygon2obj,
)
from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.logging import BadSeedError

logger = logging.getLogger(__name__)

_eps = 0.01


@gin.configurable(denylist=["graph", "level"])
class BlueprintSolidifier:
    def __init__(
        self,
        graph: RoomGraph,
        level,
        has_ceiling=True,
        combined_room_types=COMBINED_ROOM_TYPES,
        panoramic_room_types=PANORAMIC_ROOM_TYPES,
        enable_open=True,
    ):
        self.graph = graph
        self.level = level
        self.has_ceiling = has_ceiling
        self.combined_room_types = combined_room_types
        self.panoramic_room_types = panoramic_room_types
        self.beveler = BevelSharp(mult=10)
        self.enable_open = enable_open

    def get_entrance(self, names):
        return (
            None
            if self.graph.entrance is None
            else {
                k
                for k, n in names.items()
                if n == self.graph.rooms[self.graph.entrance]
            }.pop()
        )

    def get_staircase(self, names):
        return {
            k for k, n in names.items() if get_room_type(n) == RoomType.Staircase
        }.pop()

    @staticmethod
    def unroll(x):
        for k, cs in x.items():
            if isinstance(cs, Mapping):
                for l, c in cs.items():
                    if k < l:
                        yield (k, l), c
            elif isinstance(cs, Iterable):
                for c in cs:
                    yield (k,), c
            else:
                yield (k,), cs

    def solidify(self, assignment, info):
        segments = info["segments"]
        neighbours = info["neighbours"]
        shared_edges = info["shared_edges"]
        exterior_edges = info["exterior_edges"]

        names = {k: self.graph.rooms[assignment.index(k)] for k in segments}
        rooms = {
            k: self.make_room(p, exterior_edges.get(k, None))
            for k, p in segments.items()
        }
        for k, o in rooms.items():
            o.name = f"{names[k]}-{self.level}"
        # if segments[k].area > 2.5 * TYPICAL_AREA_ROOM_TYPES[get_room_type(names[k])] + 5:
        #     raise BadSeedError()
        #

        open_cutters, door_cutters = self.make_interior_cutters(
            neighbours, shared_edges, segments, names
        )
        exterior_cutters = self.make_exterior_cutters(exterior_edges, names)

        for k, r in rooms.items():
            r.location[-1] += WALL_HEIGHT * self.level
        for cutters in [open_cutters, door_cutters, exterior_cutters]:
            for k, c in self.unroll(cutters):
                c.location[-1] += WALL_HEIGHT * self.level

        butil.put_in_collection(rooms.values(), "placeholders:room_shells")

        state = self.convert_solver_state(
            rooms, segments, shared_edges, open_cutters, door_cutters, exterior_cutters
        )

        def clone_as_meshed(o):
            new = butil.copy(o)
            new.name = o.name + ".meshed"
            return new

        rooms = {k: clone_as_meshed(r) for k, r in rooms.items()}

        # Cut windows & doors from final room meshes
        cutter_col = butil.get_collection("placeholders:portal_cutters")
        for cutters in [open_cutters, door_cutters, exterior_cutters]:
            for k, c in self.unroll(cutters):
                for k_ in k:
                    butil.put_in_collection(c, cutter_col)
                    before = len(rooms[k_].data.polygons)
                    butil.modify_mesh(
                        rooms[k_],
                        "BOOLEAN",
                        object=c,
                        operation="DIFFERENCE",
                        use_self=True,
                        use_hole_tolerant=True,
                    )
                    after = len(rooms[k_].data.polygons)
                    logger.debug(
                        f"Cutting {c.name} from {rooms[k_].name}, {before=} {after=}"
                    )

        for r in rooms.values():
            butil.modify_mesh(r, "TRIANGULATE", min_vertices=3)
            remove_faces(r, read_area(r) < 5e-4)
            with butil.ViewportMode(r, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.dissolve_limited(angle_limit=0.001)
            x, y, z = read_co(r).T
            z = np.where(np.abs(z - WALL_THICKNESS / 2) < 0.01, WALL_THICKNESS / 2, z)
            z = np.where(
                np.abs(z - WALL_HEIGHT + WALL_THICKNESS / 2) < 0.01,
                WALL_HEIGHT - WALL_THICKNESS / 2,
                z,
            )
            write_co(r, np.stack([x, y, z], -1))
            butil.modify_mesh(r, "WELD", merge_threshold=WALL_THICKNESS / 10)

            direction = read_edge_direction(r)
            z_edges = np.abs(direction[:, -1])
            orthogonal = (z_edges < 0.1) | (z_edges > 0.9)
            with butil.ViewportMode(r, "EDIT"):
                edge_faces = np.zeros(len(orthogonal))
                bm = bmesh.from_edit_mesh(r.data)
                for f in bm.faces:
                    for e in f.edges:
                        edge_faces[e.index] += 1
            orthogonal = (
                (z_edges < 0.1)
                | (z_edges > 0.9)
                | (edge_faces != 1)
                | (read_edge_length(r) < 0.5)
            )
            if not orthogonal.all():
                raise BadSeedError("No orthogonal edges")

        butil.group_in_collection(rooms.values(), "placeholders:room_meshes")

        return state, rooms

    def convert_solver_state(
        self,
        rooms,
        segments,
        shared_edges,
        open_cutters,
        door_cutters,
        exterior_cutters,
    ):
        obj_states = {}
        for k, o in rooms.items():
            tags = {t.Semantics.Room, t.Semantics(o.name.split("_")[0])}

            tags.add(t.SpecificObject(o.name))
            obj_states[o.name] = ObjectState(obj=o, tags=tags, contour=segments[k])
        for k, r in rooms.items():
            relations = obj_states[r.name].relations
            for other in shared_edges[k]:
                if other in open_cutters[k]:
                    ct = cl.ConnectorType.Open
                elif other in door_cutters[k]:
                    ct = cl.ConnectorType.Door
                else:
                    ct = cl.ConnectorType.Wall
                relations.append(
                    RelationState(cl.RoomNeighbour({ct}), rooms[other].name)
                )

        def cut_state(x):
            return RelationState(cl.CutFrom(), rooms[x].name)

        for cutters in [door_cutters, open_cutters, exterior_cutters]:
            for k, c in self.unroll(cutters):
                tags = set({t.Semantics.Cutter, t.SpecificObject(c.name)})

                # TODO Lingjie - do not store whole-object window/door semantics in per-vertex attributes
                meshtags = tagging.union_object_tags(c)
                for tag in [t.Semantics.Door, t.Semantics.Window, t.Semantics.Entrance]:
                    if tag.value in meshtags:
                        tags.add(tag)

                if t.Semantics.Door in meshtags:
                    # include full possible swing extent of door in state to prevent objects blocking
                    c.scale.x *= (DOOR_WIDTH + WALL_THICKNESS) / DOOR_WIDTH

                obj_states[c.name] = ObjectState(
                    obj=c, tags=tags, relations=list(cut_state(k_) for k_ in k)
                )

        return State(objs=obj_states)

    def make_room(self, obj, exterior_edges=None):
        obj = polygon2obj(canonicalize(obj), True)
        butil.modify_mesh(obj, "WELD", merge_threshold=0.2)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=WALL_HEIGHT, offset=-1)
        self.tag(obj, False)
        if exterior_edges is not None:
            center = read_center(obj)
            exterior_centers = []
            for ls in (
                exterior_edges.geoms
                if exterior_edges.geom_type == "MultiLineString"
                else [exterior_edges]
            ):
                for u, v in zip(ls.coords[:-1], ls.coords[1:]):
                    exterior_centers.append(((u[0] + v[0]) / 2, (u[1] + v[1]) / 2))
            exterior = (
                (
                    np.abs(
                        center[:, np.newaxis, :2]
                        - np.array(exterior_centers)[np.newaxis]
                    ).sum(-1)
                    < WALL_THICKNESS * 4
                )
                .any(-1)
                .astype(int)
            )
        else:
            exterior = np.zeros(len(obj.data.polygons), dtype=int)
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.Exterior.value}", exterior, "INT", "FACE"
        )
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.Interior.value}", 1 - exterior, "INT", "FACE"
        )

        assert len(obj.data.vertices) > 0

        obj.vertex_groups.new(name="visible_")
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=WALL_THICKNESS / 2,
            offset=-1,
            use_even_offset=True,
            shell_vertex_group="visible_",
            use_quality_normals=True,
        )
        write_attribute(
            obj, "visible_", f"{PREFIX}{t.Subpart.Visible.value}", "FACE", "INT"
        )
        obj.vertex_groups.remove(obj.vertex_groups["visible_"])
        tagging.tag_object(obj, t.Semantics.Room)
        return obj

    def make_interior_cutters(self, neighbours, shared_edges, segments, names):
        name_groups = {}
        for k, n in names.items():
            name_groups[k] = set(
                i
                for i, rt in enumerate(self.combined_room_types)
                if get_room_type(n) in rt
            )
        dist2entrance = self.compute_dist2entrance(neighbours, names)
        centroids = {k: np.array(s.centroid.coords[0]) for k, s in segments.items()}
        open_cutters, door_cutters = defaultdict(dict), defaultdict(dict)
        for k, ses in shared_edges.items():
            for l, se in ses.items():
                if l not in neighbours[k] or k >= l:
                    continue
                if (
                    len(name_groups[k].intersection(name_groups[l])) > 0
                    and self.enable_open
                ):
                    open_cutters[k][l] = open_cutters[l][k] = self.make_open_cutter(se)
                else:
                    direction = (centroids[k] - centroids[l]) * (
                        1 if dist2entrance[k] > dist2entrance[l] else -1
                    )
                    door_cutters[k][l] = door_cutters[l][k] = self.make_door_cutter(
                        se, direction
                    )
        return open_cutters, door_cutters

    def compute_dist2entrance(self, neighbours, names):
        root = self.get_entrance(names)
        if root is None:
            root = self.get_staircase(names)
        queue = deque([root])
        dist2living_room = {root: 0}
        while len(queue) > 0:
            node = queue.popleft()
            for n in neighbours[node]:
                if n not in dist2living_room:
                    dist2living_room[n] = dist2living_room[node] + 1
                    queue.append(n)
        return dist2living_room

    def make_exterior_cutters(self, exterior_edges, names):
        cutters = defaultdict(list)
        entrance = self.get_entrance(names)

        for k, mls in exterior_edges.items():
            room_type = get_room_type(names[k])
            pano_chance = self.panoramic_room_types.get(room_type, 0)
            is_panoramic = uniform() < pano_chance

            lss = []
            for ls in mls.geoms:
                coords = ls.coords[:]
                lss.extend(list(zip(coords[:-1], coords[1:])))
            np.random.shuffle(lss)
            if k == entrance:
                ls = lss.pop()
                cutter = self.make_entrance_cutter(ls)
                cutters[k].append(cutter)
            for ls in lss:
                coords = LineString(ls).segmentize(MAX_WINDOW_LENGTH).coords[:]
                for seg in zip(coords[:-1], coords[1:]):
                    length = np.linalg.norm(
                        [seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]]
                    )
                    if (
                        length >= DOOR_WIDTH + WALL_THICKNESS
                        and uniform() < WINDOW_ROOM_TYPES[get_room_type(names[k])]
                    ):
                        cutter = self.make_window_cutter(seg, is_panoramic)
                        cutters[k].append(cutter)
        return cutters

    def make_staircase_cutters(self, staircase, names):
        cutters = defaultdict(list)
        if self.level > 0:
            for k, name in names.items():
                if get_room_type(name) == RoomType.Staircase:
                    with np.errstate(invalid="ignore"):
                        cutter = polygon2obj(buffer(staircase, -WALL_THICKNESS / 2))
                    butil.modify_mesh(
                        cutter, "SOLIDIFY", thickness=WALL_THICKNESS * 1.2, offset=0
                    )
                    self.tag(cutter)
                    cutter.name = "staircase_cutter"
                    cutters[k].append(cutter)
        return cutters

    def make_door_cutter(self, es, direction):
        lengths = [ls.length for ls in es.geoms]
        (x, y), (x_, y_) = es.geoms[np.argmax(lengths)].coords

        cutter = new_cube()
        vertical = np.abs(x - x_) < 0.1
        cutter.scale = DOOR_WIDTH / 2 * (1 - _eps), DOOR_WIDTH, DOOR_SIZE / 2

        butil.apply_transform(cutter, True)
        if vertical:
            y = uniform(min(y, y_) + DOOR_MARGIN, max(y, y_) - DOOR_MARGIN)
            z_rot = -np.pi / 2 if direction[0] > 0 else np.pi / 2
        else:
            x = uniform(min(x, x_) + DOOR_MARGIN, max(x, x_) - DOOR_MARGIN)
            z_rot = 0 if direction[-1] > 0 else np.pi
        cutter.location = x, y, DOOR_SIZE / 2 + WALL_THICKNESS / 2 + _eps
        cutter.rotation_euler[-1] = z_rot
        tagging.tag_object(cutter, t.Semantics.Door)
        self.tag(cutter)
        cutter.name = t.Semantics.Door.value
        return cutter

    def make_entrance_cutter(self, ls):
        (x, y), (x_, y_) = ls
        cutter = new_cube()
        length = np.linalg.norm([y_ - y, x_ - x])
        d = (DOOR_WIDTH + WALL_THICKNESS) / 2 / length
        lam = uniform(d, 1 - d)
        cutter.scale = DOOR_WIDTH / 2, DOOR_WIDTH / 2, DOOR_SIZE / 2
        butil.apply_transform(cutter, True)
        cutter.location = (
            lam * x + (1 - lam) * x_,
            lam * y + (1 - lam) * y_,
            DOOR_SIZE / 2 + WALL_THICKNESS / 2 + _eps,
        )
        cutter.rotation_euler = 0, 0, np.arctan2(y_ - y, x_ - x)
        self.tag(cutter)
        tagging.tag_object(cutter, t.Semantics.Entrance)
        cutter.name = t.Semantics.Entrance.value
        return cutter

    def make_window_cutter(self, ls, is_panoramic):
        (x, y), (x_, y_) = ls
        length = np.linalg.norm([y_ - y, x_ - x])
        if is_panoramic:
            x_scale = length / 2 - WALL_THICKNESS / 2
            lam = 1 / 2
            z_scale = (WALL_HEIGHT - WALL_THICKNESS) / 2
            z_loc = z_scale + WALL_THICKNESS / 2
        else:
            x_scale = uniform(DOOR_WIDTH / 2, length / 2 - WALL_THICKNESS / 2)
            m = (x_scale + WALL_THICKNESS / 2) / length
            lam = uniform(m, 1 - m)
            z_scale = WINDOW_SIZE / 2
            z_loc = z_scale + WINDOW_HEIGHT + WALL_THICKNESS / 2
        cutter = new_cube()
        cutter.scale = x_scale, WALL_THICKNESS, z_scale
        butil.apply_transform(cutter)
        cutter.location = lam * x + (1 - lam) * x_, lam * y + (1 - lam) * y_, z_loc
        cutter.rotation_euler = 0, 0, np.arctan2(y - y_, x - x_)
        self.tag(cutter)
        tagging.tag_object(cutter, t.Semantics.Window)
        cutter.name = t.Semantics.Window.value
        return cutter

    def make_open_cutter(self, es):
        es = remove_repeated_points(
            simplify(es, SIMPLIFY_THRESH).normalize(), WELD_THRESH
        )
        es = linemerge(es) if not isinstance(es, LineString) else es
        es = [es] if isinstance(es, LineString) else es.geoms
        lines = []
        for ls in es:
            coords = np.array(ls.coords[:])
            start, end = 0, -1
            while np.linalg.norm(coords[start] - coords[start + 1]) < SEGMENT_MARGIN:
                start += 1
            while np.linalg.norm(coords[end] - coords[end - 1]) < SEGMENT_MARGIN:
                end -= 1
            coords = coords[start : end + 1] if end < -1 else coords[start:]
            if len(coords) < 2:
                continue
            coords[0] = line_interpolate_point(
                LineString(coords[0:2]), WALL_THICKNESS / 2 + _eps
            ).coords[0]
            coords[-1] = line_interpolate_point(
                LineString(coords[-1:-3:-1]), WALL_THICKNESS / 2 + _eps
            ).coords[0]
            line = new_line(len(coords) - 1)
            write_co(line, np.concatenate([coords, np.zeros((len(coords), 1))], -1))
            lines.append(line)
        cutter = join_objects(lines)
        butil.modify_mesh(cutter, "WELD", merge_threshold=WELD_THRESH)
        butil.select_none()

        with butil.ViewportMode(cutter, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, 0, WALL_HEIGHT - WALL_THICKNESS - 2 * _eps)
                }
            )
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)

        cutter.location[-1] += WALL_THICKNESS / 2 + _eps
        butil.apply_transform(cutter, True)
        butil.modify_mesh(
            cutter,
            "SOLIDIFY",
            thickness=WALL_THICKNESS * 3,
            offset=0,
            use_even_offset=True,
        )
        self.tag(cutter)
        tagging.tag_object(cutter, t.Semantics.Open)
        cutter.name = t.Semantics.Open.value
        return cutter

    @staticmethod
    def tag(obj, visible=True):
        center = read_center(obj) + obj.location
        ceiling = center[:, -1] > WALL_HEIGHT - WALL_THICKNESS / 2 - 0.1
        floor = center[:, -1] < WALL_THICKNESS / 2 + 0.1
        wall = ~(ceiling | floor)
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.Ceiling.value}", ceiling, "INT", "FACE"
        )
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.SupportSurface.value}", floor, "INT", "FACE"
        )
        write_attr_data(obj, f"{PREFIX}{t.Subpart.Wall.value}", wall, "INT", "FACE")
        write_attr_data(obj, "segment_id", np.arange(len(center)), "INT", "FACE")
        write_attr_data(
            obj,
            f"{PREFIX}{t.Subpart.Visible.value}",
            np.ones_like(ceiling) if visible else np.zeros_like(ceiling),
            "INT",
            "FACE",
        )
        write_attr_data(
            obj,
            f"{PREFIX}{t.Subpart.Invisible.value}",
            np.zeros_like(ceiling) if visible else np.ones_like(ceiling),
            "INT",
            "FACE",
        )
        parse_scene.preprocess_obj(obj)
        tagging.tag_canonical_surfaces(obj)
