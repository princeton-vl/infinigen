# Copyright (c) Princeton University.

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence

import bmesh
import bpy
import gin
import numpy as np
from numpy.random import uniform
from shapely import LineString, line_interpolate_point, remove_repeated_points, simplify
from shapely.ops import linemerge
from numpy.random import uniform
from infinigen.core.constraints.example_solver.room.configs import (
    COMBINED_ROOM_TYPES, PANORAMIC_ROOM_TYPES,
    WINDOW_ROOM_TYPES, TYPICAL_AREA_ROOM_TYPES,
)
from infinigen.core.constraints.example_solver.room.constants import DOOR_MARGIN, DOOR_SIZE, DOOR_WIDTH, \
    MAX_WINDOW_LENGTH, SEGMENT_MARGIN, WALL_HEIGHT, WALL_THICKNESS, WINDOW_HEIGHT, WINDOW_SIZE
from infinigen.core.constraints.example_solver.room.utils import SIMPLIFY_THRESH, WELD_THRESH, buffer, \
    canonicalize, polygon2obj
from infinigen.assets.utils.decorate import (
    read_area, read_center, read_co, remove_edges, remove_faces,
    select_faces, write_attribute, write_co, read_edges, read_edge_direction, read_edge_length,
)
from infinigen.assets.utils.object import data2mesh, join_objects, mesh2obj, new_cube, new_line
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.constraints.example_solver.state_def import ObjectState, RelationState, State
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.util.logging import BadSeedError


@gin.configurable(denylist=['graph', 'level'])
class BlueprintSolidifier:
    def __init__(self, graph: RoomGraph, level, has_ceiling=True, combined_room_types=COMBINED_ROOM_TYPES,
                 panoramic_room_types=PANORAMIC_ROOM_TYPES, enable_open=True):
        self.graph = graph
        self.level = level
        self.has_ceiling = has_ceiling
        self.combined_room_types = combined_room_types
        self.panoramic_room_types = panoramic_room_types
        self.enable_open = enable_open

    def get_entrance(self, names):
        return None if self.graph.entrance is None else {k for k, n in names.items() if
            n == self.graph.rooms[self.graph.entrance]}.pop()

    def get_staircase(self, names):
        return {k for k, n in names.items() if get_room_type(n) == RoomType.Staircase}.pop()

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
        segments = info['segments']
        neighbours = info['neighbours']
        shared_edges = info['shared_edges']
        exterior_edges = info['exterior_edges']
        names = {k: self.graph.rooms[assignment.index(k)] for k in segments}
        rooms = {k: self.make_room(p, exterior_edges.get(k, None)) for k, p in segments.items()}
            o.name = f'{names[k]}-{self.level}'
           # if segments[k].area > 2.5 * TYPICAL_AREA_ROOM_TYPES[get_room_type(names[k])] + 5:
           #     raise BadSeedError()
           # 
        open_cutters, door_cutters = self.make_interior_cutters(neighbours, shared_edges, segments, names)
        exterior_cutters = self.make_exterior_cutters(exterior_edges, names)
        for k, r in rooms.items():
            r.location[-1] += WALL_HEIGHT * self.level
        for cutters in [open_cutters, door_cutters, exterior_cutters]:
            for k, c in self.unroll(cutters):
                for k_ in k:
                    butil.modify_mesh(
                        rooms[k_], 'BOOLEAN', object=c, operation='DIFFERENCE', use_self=True,
                        use_hole_tolerant=True
                    )
            butil.modify_mesh(r, 'TRIANGULATE', min_vertices=3)
            remove_faces(r, read_area(r) < 5e-4)
            with butil.ViewportMode(r, 'EDIT'):
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.dissolve_limited(angle_limit=0.001)
            x, y, z = read_co(r).T
            z = np.where(np.abs(z - WALL_THICKNESS / 2) < .01, WALL_THICKNESS / 2, z)
            z = np.where(np.abs(z - WALL_HEIGHT + WALL_THICKNESS / 2) < .01, WALL_HEIGHT - WALL_THICKNESS / 2,
                         z)
            write_co(r, np.stack([x, y, z], -1))
            butil.modify_mesh(r, 'WELD', merge_threshold=WALL_THICKNESS / 10)
            
            direction = read_edge_direction(r)
            z_edges = np.abs(direction[:, -1])
            orthogonal = (z_edges < .1) | (z_edges > .9)
            with butil.ViewportMode(r, 'EDIT'):
                edge_faces = np.zeros(len(orthogonal))
                bm = bmesh.from_edit_mesh(r.data)
                for f in bm.faces:
                    for e in f.edges:
                        edge_faces[e.index] += 1
            orthogonal = (z_edges < .1) | (z_edges > .9) | (edge_faces != 1) | (read_edge_length(r) < .5)
            if not orthogonal.all():
                raise BadSeedError('No orthogonal edges')


    def convert_solver_state(self, rooms, segments, shared_edges, open_cutters, door_cutters, exterior_cutters):
        for k, o in rooms.items():
        for k, r in rooms.items():
            relations = obj_states[r.name].relations
            for other in shared_edges[k]:
                if other in open_cutters[k]:
                    ct = cl.ConnectorType.Open
                elif other in door_cutters[k]:
                    ct = cl.ConnectorType.Door
                else:
                    ct = cl.ConnectorType.Wall
        cut_state = lambda x: RelationState(cl.CutFrom(), rooms[x].name)
        for cutters in [door_cutters, open_cutters, exterior_cutters]:
            for k, c in self.unroll(cutters):
    def make_room(self, obj, exterior_edges=None):
        obj = polygon2obj(canonicalize(obj), True)
        butil.modify_mesh(obj, "WELD", merge_threshold=.2)
        butil.modify_mesh(obj, 'SOLIDIFY', thickness=WALL_HEIGHT, offset=-1)
        self.tag(obj, False)
        if exterior_edges is not None:
            center = read_center(obj)
            exterior_centers = []
            for ls in exterior_edges.geoms if exterior_edges.geom_type == 'MultiLineString' else [
                exterior_edges]:
                for u, v in zip(ls.coords[:-1], ls.coords[1:]):
                    exterior_centers.append(((u[0] + v[0]) / 2, (u[1] + v[1]) / 2))
            exterior = (np.abs(center[:, np.newaxis, :2] - np.array(exterior_centers)[np.newaxis]).sum(
                -1) < WALL_THICKNESS * 4).any(-1).astype(int)
        else:
            exterior = np.zeros(len(obj.data.polygons), dtype=int)

        obj.vertex_groups.new(name='visible_')
        butil.modify_mesh(obj, 'SOLIDIFY', thickness=WALL_THICKNESS / 2, offset=-1, use_even_offset=True,
                          shell_vertex_group='visible_', use_quality_normals=True)
        obj.vertex_groups.remove(obj.vertex_groups['visible_'])
    def make_interior_cutters(self, neighbours, shared_edges, segments, names):
        name_groups = {}
        for k, n in names.items():
            name_groups[k] = set(i for i, rt in enumerate(self.combined_room_types) if get_room_type(n) in rt)
        dist2entrance = self.compute_dist2entrance(neighbours, names)
        centroids = {k: np.array(s.centroid.coords[0]) for k, s in segments.items()}
        open_cutters, door_cutters = defaultdict(dict), defaultdict(dict)
        for k, ses in shared_edges.items():
            for l, se in ses.items():
                if l not in neighbours[k] or k >= l:
                    continue
                if len(name_groups[k].intersection(name_groups[l])) > 0 and self.enable_open:
                    open_cutters[k][l] = open_cutters[l][k] = self.make_open_cutter(se)
                else:
                    direction = (centroids[k] - centroids[l]) * (
                        1 if dist2entrance[k] > dist2entrance[l] else -1)
                    door_cutters[k][l] = door_cutters[l][k] = self.make_door_cutter(se, direction)
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
                    length = np.linalg.norm([seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]])
                    if length >= DOOR_WIDTH + WALL_THICKNESS and uniform() < WINDOW_ROOM_TYPES[
                        get_room_type(names[k])]:
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
                    butil.modify_mesh(cutter, 'SOLIDIFY', thickness=WALL_THICKNESS * 1.2, offset=0)
                    self.tag(cutter)
                    cutter.name = 'staircase_cutter'
                    cutters[k].append(cutter)
        return cutters

    def make_door_cutter(self, es, direction):
        lengths = [ls.length for ls in es.geoms]
        (x, y), (x_, y_) = es.geoms[np.argmax(lengths)].coords
        cutter = new_cube()
        vertical = np.abs(x - x_) < .1
        butil.apply_transform(cutter, True)
        if vertical:
            y = uniform(min(y, y_) + DOOR_MARGIN, max(y, y_) - DOOR_MARGIN)
            z_rot = -np.pi / 2 if direction[0] > 0 else np.pi / 2
        else:
            x = uniform(min(x, x_) + DOOR_MARGIN, max(x, x_) - DOOR_MARGIN)
            z_rot = 0 if direction[-1] > 0 else np.pi
        cutter.location = x, y, DOOR_SIZE / 2 + WALL_THICKNESS / 2 + _eps
        cutter.rotation_euler[-1] = z_rot
        self.tag(cutter)
        return cutter

    def make_entrance_cutter(self, ls):
        (x, y), (x_, y_) = ls
        cutter = new_cube()
        length = np.linalg.norm([y_ - y, x_ - x])
        d = (DOOR_WIDTH + WALL_THICKNESS) / 2 / length
        lam = uniform(d, 1 - d)
        cutter.scale = DOOR_WIDTH / 2, DOOR_WIDTH / 2, DOOR_SIZE / 2
        butil.apply_transform(cutter, True)
        cutter.location = lam * x + (1 - lam) * x_, lam * y + (
            1 - lam) * y_, DOOR_SIZE / 2 + WALL_THICKNESS / 2 + _eps
        cutter.rotation_euler = 0, 0, np.arctan2(y_ - y, x_ - x)
        self.tag(cutter)
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
        return cutter

    def make_open_cutter(self, es):
        es = remove_repeated_points(simplify(es, SIMPLIFY_THRESH).normalize(), WELD_THRESH)
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
            coords = coords[start:end + 1] if end < -1 else coords[start:]
            if len(coords) < 2:
                continue
            coords[0] = line_interpolate_point(LineString(coords[0: 2]), WALL_THICKNESS / 2 + _eps).coords[0]
            coords[-1] = line_interpolate_point(LineString(coords[-1:-3:-1]), WALL_THICKNESS / 2 + _eps).coords[
                0]
            line = new_line(len(coords) - 1)
            write_co(line, np.concatenate([coords, np.zeros((len(coords), 1))], -1))
            lines.append(line)
        cutter = join_objects(lines)
        butil.modify_mesh(cutter, 'WELD', merge_threshold=WELD_THRESH)
        butil.select_none()

        with butil.ViewportMode(cutter, 'EDIT'):
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={'value': (0, 0, WALL_HEIGHT - WALL_THICKNESS - 2 * _eps)
                })
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)

        cutter.location[-1] += WALL_THICKNESS / 2 + _eps
        butil.apply_transform(cutter, True)
        butil.modify_mesh(cutter, 'SOLIDIFY', thickness=WALL_THICKNESS * 3, offset=0, use_even_offset=True)
        self.tag(cutter)
        return cutter

    @staticmethod
    def tag(obj, visible=True):
        center = read_center(obj) + obj.location
        ceiling = center[:, -1] > WALL_HEIGHT - WALL_THICKNESS / 2 - .1
        floor = center[:, -1] < WALL_THICKNESS / 2 + .1
        wall = ~(ceiling | floor)
        write_attr_data(obj, 'segment_id', np.arange(len(center)), 'INT', 'FACE')
                        np.ones_like(ceiling) if visible else np.zeros_like(ceiling), 'INT', 'FACE')
                        np.zeros_like(ceiling) if visible else np.ones_like(ceiling), 'INT', 'FACE')
