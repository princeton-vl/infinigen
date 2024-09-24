# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

import logging
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

import bpy
import gin
import numpy as np
import shapely
from numpy.random import uniform
from shapely import LineString, line_interpolate_point
from shapely.ops import linemerge

from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.mesh import canonicalize_mls, prepare_for_boolean
from infinigen.assets.utils.object import new_cube
from infinigen.assets.utils.shapes import buffer, polygon2obj, simplify_polygon
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.tags import Semantics
from infinigen.core.util import blender as butil
from infinigen.core.util.random import random_general as rg

from .base import RoomGraph, room_type, valid_rooms
from .utils import mls_ccw

logger = logging.getLogger(__name__)

_eps = 1e-5
_snap = 0.5

panoramic_rooms = defaultdict(
    float,
    {
        Semantics.Hallway: ("bool", 0.1),
        Semantics.Balcony: ("bool", 0.8),
        Semantics.OpenOffice: ("bool", 0.2),
        Semantics.Office: ("bool", 0.2),
        Semantics.MeetingRoom: ("bool", 0.2),
        Semantics.BreakRoom: ("bool", 0.2),
        Semantics.Garage: ("bool", 1),
    },
)

combined_rooms = [
    ({Semantics.Bedroom}, {"non-adjacent": "none", "adjacent": "none"}),
    (
        {Semantics.Hallway, Semantics.OpenOffice},
        {"non-adjacent": "open", "adjacent": "open"},
    ),
    (
        {
            Semantics.Hallway,
            Semantics.LivingRoom,
            Semantics.DiningRoom,
            Semantics.StaircaseRoom,
        },
        {
            "non-adjacent": (
                "weighted_choice",
                (0.3, "open"),
                (0.3, "panoramic"),
                (0.4, "door"),
            ),
            "adjacent": ("weighted_choice", (0.5, "open"), (0.5, "door")),
        },
    ),
    (
        {Semantics.Garage, Semantics.Warehouse},
        {"non-adjacent": "open", "adjacent": "open"},
    ),
    (
        {Semantics.DiningRoom, Semantics.Kitchen},
        {
            "non-adjacent": (
                "weighted_choice",
                (0.3, "open"),
                (0.3, "panoramic"),
                (0.4, "door"),
            ),
            "adjacent": "open",
        },
    ),
    (
        {Semantics.Balcony, Semantics.LivingRoom, Semantics.Hallway},
        {
            "non-adjacent": (
                "weighted_choice",
                (0.3, "open"),
                (0.3, "panoramic"),
                (0.4, "door"),
            ),
            "adjacent": ("weighted_choice", (0.5, "open"), (0.5, "door")),
        },
    ),
    (
        {Semantics.Balcony, Semantics.Bedroom},
        {
            "non-adjacent": (
                "weighted_choice",
                (0.3, "open"),
                (0.3, "panoramic"),
                (0.4, "door"),
            ),
            "adjacent": "door",
        },
    ),
    (
        {Semantics.OpenOffice, Semantics.Hallway},
        {"non-adjacent": "open", "adjacent": "open"},
    ),
    (
        {
            Semantics.MeetingRoom,
            Semantics.BreakRoom,
            Semantics.Hallway,
            Semantics.OpenOffice,
        },
        {
            "non-adjacent": ("weighted_choice", (0.5, "window"), (0.5, "door")),
            "adjacent": "door",
        },
    ),
]

window_rooms = defaultdict(
    lambda: 1.0,
    {
        Semantics.Utility: ("bool", 0.3),
        Semantics.Closet: 0.0,
        Semantics.Bathroom: ("bool", 0.5),
        Semantics.Garage: 0.0,
        Semantics.Warehouse: 0.0,
    },
)

wall_cut_prob = "bool", 0.5


def split_mls(mls, min_length=-np.inf):
    lss = mls.geoms if mls.geom_type == "MultiLineString" else [mls]
    for ls in lss:
        for (x, y), (x_, y_) in zip(ls.coords[:-1], ls.coords[1:]):
            if np.linalg.norm((x - x_, y - y_)) > min_length:
                yield x, y, x_, y_


def max_mls(mls):
    lss = mls.geoms if mls.geom_type == "MultiLineString" else [mls]
    coords = []
    lengths = []
    for ls in lss:
        for (x, y), (x_, y_) in zip(ls.coords[:-1], ls.coords[1:]):
            lengths.append(np.linalg.norm((x - x_, y - y_)))
            coords.append((x, y, x_, y_))
    return coords[np.argmax(lengths)]


@gin.configurable
class BlueprintSolidifier:
    def __init__(self, consgraph, graph: RoomGraph, level, enable_open=True):
        self.constants = consgraph.constants
        self.consgraph = consgraph
        self.graph = graph
        self.level = level
        self.enable_open = enable_open

    @staticmethod
    def unroll(x):
        for k, cs in x.items():
            if isinstance(cs, Mapping):
                for l, c in cs.items():
                    if k < l:
                        if isinstance(c, Iterable):
                            for cc in c:
                                yield (k, l), cc
                        else:
                            yield (k, l), c
            elif isinstance(cs, Iterable):
                for c in cs:
                    yield (k,), c
            else:
                yield (k,), cs

    def solidify(self, state):
        wt = self.constants.wall_thickness
        segments = {k: obj_st.polygon for k, obj_st in valid_rooms(state)}
        shared_edges = {
            k: {l.target_name: canonicalize_mls(l.value) for l in obj_st.relations}
            for k, obj_st in valid_rooms(state)
        }
        exterior = next(k for k in state.objs if room_type(k) == Semantics.Exterior)
        exterior_edges = {
            r.target_name: mls_ccw(canonicalize_mls(r.value), state, r.target_name)
            for r in state[exterior].relations
        }
        exterior_buffer = shapely.simplify(
            state[exterior].polygon.buffer(-wt / 2 - _eps, join_style="mitre"), 1e-3
        )
        exterior_shape = state[exterior].polygon

        rooms = {k: self.make_room(state, k) for k, _ in valid_rooms(state)}
        open_cutters, door_cutters, interior_cutters = self.make_interior_cutters(
            self.graph.valid_neighbours, shared_edges, segments, exterior_buffer
        )
        window_cutters, entrance_cutters = self.make_exterior_cutters(
            exterior_edges, exterior_shape
        )
        all_cutter_lists = [
            open_cutters,
            door_cutters,
            window_cutters,
            entrance_cutters,
        ]

        w = self.constants.wall_height
        for k, r in rooms.items():
            r.location[-1] += w * self.level
        for cutters in all_cutter_lists:
            for k, c in self.unroll(cutters):
                c.location[-1] += w * self.level

        butil.put_in_collection(rooms.values(), "placeholders:room_shells")
        rooms_ = rooms

        def clone_as_meshed(o):
            new = butil.copy(o)
            new.name = o.name + ".meshed"
            return new

        rooms = {k: clone_as_meshed(r) for k, r in rooms.items()}
        state = self.convert_solver_state(
            rooms_,
            segments,
            shared_edges,
            open_cutters,
            door_cutters,
            window_cutters,
            interior_cutters,
            entrance_cutters,
        )
        for obj in rooms_.values():
            tagging.tag_object(obj)

        # Cut windows & doors from final room meshes
        cutter_col = butil.get_collection("placeholders:portal_cutters")
        for cutters in all_cutter_lists:
            for k, c in self.unroll(cutters):
                butil.put_in_collection(c, cutter_col)
                for k_ in k:
                    obj = rooms[k_]
                    logger.debug(f"Cutting {c.name} from {obj.name}")
                    before = len(obj.data.polygons)
                    prepare_for_boolean(obj)
                    prepare_for_boolean(c)
                    butil.modify_mesh(
                        rooms[k_],
                        "BOOLEAN",
                        object=c,
                        operation="DIFFERENCE",
                        use_self=True,
                        use_hole_tolerant=True,
                    )
                    prepare_for_boolean(obj)
                    prepare_for_boolean(c)
                    after = len(obj.data.polygons)
                    logger.debug(
                        f"Cutting {c.name} from {obj.name}, {before=} {after=}"
                    )

        for obj in rooms.values():
            butil.modify_mesh(obj, "TRIANGULATE", min_vertices=3)
            co = read_co(obj)
            m = wt / 2 + _snap
            low = np.abs(co[:, -1] - m) < _eps
            high = np.abs(co[:, -1] - self.constants.wall_height + m) < _eps
            co[:, -1] = np.where(low, wt / 2, co[:, -1])
            co[:, -1] = np.where(high, self.constants.wall_height - wt / 2, co[:, -1])
            write_co(obj, co)
            tagging.tag_object(obj)

        for obj in cutter_col.objects:
            offset = np.array(obj.location)[np.newaxis, :]
            offset[:, 2] -= w * self.level
            co = read_co(obj) + offset
            m = wt / 2 + _snap
            low = np.abs(co[:, -1] - m) < _eps
            high = np.abs(co[:, -1] - self.constants.wall_height + m) < _eps
            co[:, -1] = np.where(low, wt / 2, co[:, -1])
            co[:, -1] = np.where(high, self.constants.wall_height - wt / 2, co[:, -1])
            write_co(obj, co - offset)
            tagging.tag_object(obj)

        butil.group_in_collection(rooms.values(), "placeholders:room_meshes")
        return state, rooms

    def convert_solver_state(
        self,
        rooms,
        segments,
        shared_edges,
        open_cutters,
        door_cutters,
        window_cutters,
        interior_cutters,
        entrance_cutters,
    ):
        obj_states = {}
        for k, o in rooms.items():
            obj_states[o.name] = ObjectState(
                obj=o,
                tags={t.Semantics.Room, t.SpecificObject(o.name), room_type(o.name)},
                polygon=segments[k],
            )
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

        all_cutters = [
            door_cutters,
            open_cutters,
            window_cutters,
            interior_cutters,
            entrance_cutters,
        ]
        tag_cutters = [
            t.Semantics.Door,
            t.Semantics.Open,
            t.Semantics.Window,
            t.Semantics.Window,
            t.Semantics.Door,
        ]
        for cutters, tag in zip(all_cutters, tag_cutters):
            for k, c in self.unroll(cutters):
                obj_states[c.name] = ObjectState(
                    obj=c,
                    tags={tag, t.Semantics.Cutter},
                    relations=[RelationState(cl.CutFrom(), rooms[k_].name) for k_ in k],
                )

        return State(objs=obj_states)

    def make_room(self, state, name):
        obj_st = state.objs[name]
        obj = polygon2obj(
            shapely.segmentize(
                self.constants.canonicalize(obj_st.polygon), self.constants.door_width
            ),
            True,
            dissolve=False,
        )
        butil.modify_mesh(obj, "WELD", merge_threshold=0.01)
        butil.modify_mesh(
            obj, "SOLIDIFY", thickness=self.constants.wall_height, offset=-1
        )
        obj.name = name
        self.tag(obj, False)
        center = read_center(obj)
        exterior_centers = []

        exterior = next(k for k in state.objs if room_type(k) == Semantics.Exterior)
        exterior_edge = next(
            r.value for r in state.objs[exterior].relations if r.target_name == name
        )
        for ls in exterior_edge.geoms:
            for u, v in zip(ls.coords[:-1], ls.coords[1:]):
                exterior_centers.append(((u[0] + v[0]) / 2, (u[1] + v[1]) / 2))
        if len(exterior_centers) > 0:
            exterior = (
                np.abs(
                    center[:, np.newaxis, :2] - np.array(exterior_centers)[np.newaxis]
                ).sum(-1)
                < self.constants.wall_thickness * 4
            ).any(-1)
        else:
            exterior = np.zeros(len(obj.data.polygons), dtype=bool)
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.Interior.value}", ~exterior, "BOOLEAN", "FACE"
        )
        assert len(obj.data.vertices) > 0

        obj.vertex_groups.new(name="visible_")
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.constants.wall_thickness / 2,
            offset=-1,
            use_even_offset=True,
            shell_vertex_group="visible_",
            use_quality_normals=True,
        )
        write_attribute(
            obj, "visible_", f"{PREFIX}{t.Subpart.Visible.value}", "FACE", "BOOLEAN"
        )
        obj.vertex_groups.remove(obj.vertex_groups["visible_"])
        return obj

    def make_interior_cutters(self, neighbours, shared_edges, segments, exterior):
        name_groups = {}
        for k in shared_edges:
            name_groups[k] = set(
                i for i, (rt, _) in enumerate(combined_rooms) if room_type(k) in rt
            )
        dist2entrance = self.compute_dist2entrance(neighbours)
        centroids = {k: np.array(s.centroid.coords[0]) for k, s in segments.items()}
        open_cutters, door_cutters, interior_cutters = (
            defaultdict(dict),
            defaultdict(dict),
            defaultdict(dict),
        )
        for k, ses in shared_edges.items():
            for l, se in ses.items():
                if k >= l or se.length <= self.constants.segment_margin:
                    continue
                direction = (centroids[k] - centroids[l]) * (
                    1 if dist2entrance[k] > dist2entrance[l] else -1
                )
                i = name_groups[k].intersection(name_groups[l])
                if len(i) > 0 and self.enable_open:
                    group = combined_rooms[next(iter(i))][1]
                    fn = rg(
                        group["adjacent"]
                        if k in neighbours[l]
                        else group["non-adjacent"]
                    )
                else:
                    fn = "door" if k in neighbours[l] else "none"
                match fn:
                    case "open":
                        open_cutters[k][l] = open_cutters[l][k] = self.make_open_cutter(
                            se, exterior
                        )
                    case "door":
                        door_cutters[k][l] = door_cutters[l][k] = self.make_door_cutter(
                            se, direction
                        )
                    case "window":
                        interior_cutters[k][l] = interior_cutters[l][k] = (
                            self.make_window_cutter(se, False)
                        )
                    case "panoramic":
                        interior_cutters[k][l] = interior_cutters[l][k] = (
                            self.make_window_cutter(se, self.level == 0)
                        )
        return open_cutters, door_cutters, interior_cutters

    def compute_dist2entrance(self, neighbours):
        root = self.graph.root
        queue = deque([root])
        dist2living_room = {root: 0}
        while len(queue) > 0:
            node = queue.popleft()
            for n in neighbours[node]:
                if n not in dist2living_room:
                    dist2living_room[n] = dist2living_room[node] + 1
                    queue.append(n)
        return dist2living_room

    def make_exterior_cutters(self, exterior_edges, exterior):
        window_cutters = defaultdict(list)
        entrance_cutters = defaultdict(list)
        entrance = self.graph.entrance

        for k, mls in exterior_edges.items():
            if k == entrance and self.level == 0:
                continue
            for ls in mls.geoms:
                ls = ls.segmentize(self.constants.max_window_length)
                buffered = ls.buffer(0.1, single_sided=True)
                if buffered.intersection(exterior).area < buffered.area / 2:
                    ls = LineString(ls.coords[::-1])
                cutters = self.make_window_cutter(ls, panoramic_rooms[room_type(k)])
                window_cutters[k].extend(cutters)
        for k, mls in exterior_edges.items():
            if k == entrance and self.level == 0:
                x, y, x_, y_ = max_mls(mls)
                ls = LineString([(x, y), (x_, y_)])
                cutter = self.make_entrance_cutter(ls)
                entrance_cutters[k].append(cutter)
                mls = mls.difference(ls)
                if mls.length > 0:
                    cutters = self.make_window_cutter(mls, False)
                    window_cutters[k].extend(cutters)
        return window_cutters, entrance_cutters

    def make_staircase_cutters(self, staircase, names):
        cutters = defaultdict(list)
        if self.level > 0:
            for k, name in names.items():
                if room_type(name) == Semantics.StaircaseRoom:
                    with np.errstate(invalid="ignore"):
                        cutter = polygon2obj(
                            buffer(staircase, -self.constants.wall_thickness / 2)
                        )
                    butil.modify_mesh(
                        cutter,
                        "SOLIDIFY",
                        thickness=self.constants.wall_thickness * 1.2,
                        offset=0,
                    )
                    cutter.name = "staircase_cutter"
                    self.tag(cutter)
                    cutters[k].append(cutter)
        return cutters

    def make_door_cutter(self, mls, direction):
        m = self.constants.door_margin + self.constants.door_width / 2
        x, y, x_, y_ = max_mls(mls)
        cutter = new_cube()
        vertical = np.abs(x - x_) < 0.1
        wt = self.constants.wall_thickness
        cutter.scale = (
            self.constants.door_width / 2,
            self.constants.door_width + wt / 2,
            self.constants.door_size / 2 - _snap / 2,
        )
        cutter.location[-1] += _snap / 2
        butil.apply_transform(cutter, True)
        if vertical:
            y = uniform(min(y, y_) + m, max(y, y_) - m)
            z_rot = -np.pi / 2 if direction[0] > 0 else np.pi / 2
        else:
            x = uniform(min(x, x_) + m, max(x, x_) - m)
            z_rot = 0 if direction[-1] > 0 else np.pi
        cutter.location = x, y, self.constants.door_size / 2 + wt / 2
        cutter.rotation_euler[-1] = z_rot
        cutter.name = t.Semantics.Door.value
        self.tag(cutter)
        return cutter

    def make_entrance_cutter(self, mls):
        x, y, x_, y_ = max_mls(mls)
        cutter = new_cube()
        length = np.linalg.norm([y_ - y, x_ - x])
        m = self.constants.door_margin + self.constants.door_width / 2
        lam = uniform(m / length, 1 - m / length)
        wt = self.constants.wall_thickness
        cutter.scale = (
            self.constants.door_width / 2,
            self.constants.door_width / 2 + wt,
            self.constants.door_size / 2 - _snap / 2,
        )
        cutter.location[-1] += _snap / 2
        butil.apply_transform(cutter, True)
        cutter.location = (
            lam * x + (1 - lam) * x_,
            lam * y + (1 - lam) * y_,
            self.constants.door_size / 2 + wt / 2,
        )
        cutter.rotation_euler = 0, 0, np.arctan2(y_ - y, x_ - x)
        cutter.name = t.Semantics.Entrance.value
        self.tag(cutter)
        return cutter

    def make_window_cutter(self, mls, is_panoramic):
        cutters = []
        for x, y, x_, y_ in split_mls(mls, self.constants.door_width):
            length = np.linalg.norm([y_ - y, x_ - x])
            wt = self.constants.wall_thickness
            wm = self.constants.window_margin

            if rg(is_panoramic) and self.constants.wall_height < 4:
                x_scale = length / 2 - wm
                lam = 1 / 2
                z_scale = (self.constants.wall_height - wt) / 2 - _snap
                z_loc = z_scale + wt / 2 + _snap
            else:
                x_scale = uniform(self.constants.door_width / 2, length / 2 - wm)
                m = (x_scale + wm) / length
                lam = uniform(m, 1 - m)
                z_scale = self.constants.window_size / 2
                z_loc = z_scale + self.constants.window_height + wt / 2

            cutter = new_cube()
            cutter.scale = x_scale, wt, z_scale
            butil.apply_transform(cutter)
            cutter.location = lam * x + (1 - lam) * x_, lam * y + (1 - lam) * y_, z_loc
            cutter.rotation_euler = 0, 0, np.arctan2(y - y_, x - x_)
            cutter.name = t.Semantics.Window.value
            self.tag(cutter)
            cutters.append(cutter)
        return cutters

    def make_open_cutter(self, es, exterior):
        es = simplify_polygon(es)
        es = shapely.remove_repeated_points(
            linemerge(es) if not isinstance(es, LineString) else es, 0.01
        )
        es = [es] if isinstance(es, LineString) else es.geoms
        lines = []
        wt = self.constants.wall_thickness
        for ls in es:
            coords = np.array(ls.coords[:])
            if len(coords) < 2:
                continue
            coords[0] = line_interpolate_point(
                LineString(coords[0:2]), wt / 2 + _eps
            ).coords[0]
            coords[-1] = line_interpolate_point(
                LineString(coords[-1:-3:-1]), wt / 2 + _eps
            ).coords[0]
            lines.append(coords)
        line = shapely.simplify(
            shapely.remove_repeated_points(shapely.MultiLineString(lines), 0.01), 0.01
        )

        p = line.buffer(wt, cap_style="flat", join_style="mitre")
        p = p.intersection(exterior)
        cutters = []
        for p in [p] if p.geom_type == "Polygon" else p.geoms:
            cutter = polygon2obj(p, True)

            with butil.ViewportMode(cutter, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.extrude_region_move(
                    TRANSFORM_OT_translate={
                        "value": (0, 0, self.constants.wall_height - wt - 2 * _snap)
                    }
                )
                bpy.ops.mesh.select_mode(type="FACE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
            cutter.location[-1] += wt / 2 + _snap
            cutter.name = t.Semantics.Open.value
            self.tag(cutter)
            cutters.append(cutter)
        return cutters

    def tag(self, obj, visible=True):
        center = read_center(obj) + obj.location
        high = self.constants.wall_height - self.constants.wall_thickness / 2
        z = center[:, -1]
        low = self.constants.wall_thickness / 2
        ceiling = (z > high - _eps) | (np.abs(z - high + _snap) < _eps)
        floor = (z < low + _eps) | (np.abs(z - low - _snap) < _eps)
        wall = ~(ceiling | floor)
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.Ceiling.value}", ceiling, "BOOLEAN", "FACE"
        )
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.SupportSurface.value}", floor, "BOOLEAN", "FACE"
        )
        write_attr_data(obj, f"{PREFIX}{t.Subpart.Wall.value}", wall, "BOOLEAN", "FACE")
        full = np.ones_like(ceiling)
        if visible:
            write_attr_data(
                obj, f"{PREFIX}{t.Subpart.Visible.value}", full, "BOOLEAN", "FACE"
            )
        else:
            write_attr_data(
                obj, f"{PREFIX}{t.Subpart.Visible.value}", ~full, "BOOLEAN", "FACE"
            )
        write_attr_data(
            obj, f"{PREFIX}{t.Subpart.Interior.value}", full, "BOOLEAN", "FACE"
        )
