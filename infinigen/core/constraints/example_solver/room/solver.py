# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import gin
import numpy as np
from numpy.random import uniform
from shapely import LineString, Polygon, union
from shapely.ops import shared_paths

import infinigen.core.constraints.example_solver.room.constants as constants
from infinigen.core.constraints.example_solver.room.configs import (
    EXTERIOR_CONNECTED_ROOM_TYPES,
)
from infinigen.core.constraints.example_solver.room.constants import SEGMENT_MARGIN
from infinigen.core.constraints.example_solver.room.types import RoomType, get_room_type
from infinigen.core.constraints.example_solver.room.utils import (
    canonicalize,
    compute_neighbours,
    cut_polygon_by_line,
    is_valid_polygon,
    linear_extend_x,
    linear_extend_y,
    update_exterior_edges,
    update_shared_edges,
    update_staircase_occupancies,
)


@dataclass
class RoomSolverMsg:
    status: str
    index_changed: Optional[List[int]] = None

    @property
    def is_success(self):
        return self.status == "success"


@gin.configurable(denylist=["contour", "graph"])
class BlueprintSolver:
    def __init__(
        self,
        contour,
        graph,
        exterior_connected_room_types=EXTERIOR_CONNECTED_ROOM_TYPES,
        max_stride=1,
        staircase_occupancy_thresh=0.75,
    ):
        self.contour = contour
        x, y = self.contour.boundary.xy
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.y_min, self.y_max = np.min(y), np.max(y)
        self.graph = graph
        self.staircase_occupancy_thresh = staircase_occupancy_thresh
        self.exterior_connected_room_types = exterior_connected_room_types
        self.exterior_connected_rooms = set(
            i
            for i, r in enumerate(self.graph.rooms)
            if get_room_type(r) in self.exterior_connected_room_types
        )
        if self.graph.entrance is not None:
            self.exterior_connected_rooms.add(self.graph.entrance)
        self.staircase_rooms = set(self.graph[RoomType.Staircase])
        self.max_stride = max_stride

    def find_assignment(self, info):
        assignment = [0] * len(self.graph.rooms)
        neighbours_all = info["neighbours_all"]
        exterior_neighbours = info["exterior_neighbours"]
        staircase_occupancies = info["staircase_occupancies"]
        if info["staircase"] is not None:
            staircase_candidates = list(
                (
                    k
                    for k, v in staircase_occupancies.items()
                    if v > self.staircase_occupancy_thresh
                )
            )
            if len(staircase_candidates) == 0:
                return None
        else:
            staircase_candidates = []
        unassigned = set(neighbours_all.keys())

        def assign_(i):
            if i == len(self.graph.rooms):
                return assignment
            if i in self.staircase_rooms:
                candidates = unassigned.intersection(staircase_candidates)
            elif i in self.exterior_connected_rooms:
                candidates = unassigned.intersection(exterior_neighbours)
            else:
                candidates = unassigned.copy()
            n_unassigned = len(list(j for j in self.graph.neighbours[i] if j > i))
            assigned_neighbours = set(
                assignment[j] for j in self.graph.neighbours[i] if j < i
            )
            for n in candidates:
                if assigned_neighbours.issubset(neighbours_all[n]):
                    if len(neighbours_all[n].intersection(unassigned)) >= n_unassigned:
                        assignment[i] = n
                        unassigned.remove(n)
                        r = assign_(i + 1)
                        if r is not None:
                            return r
                        unassigned.add(n)

        return assign_(0)

    def satisfies_constraints(self, assignment, info):
        neighbours_all = info["neighbours_all"]
        exterior_neighbours = info["exterior_neighbours"]
        staircase_occupancies = info["staircase_occupancies"]
        for k, ns in enumerate(self.graph.neighbours):
            for n in ns:
                if assignment[k] not in neighbours_all[assignment[n]]:
                    return RoomSolverMsg("neighbours unsatisfied", [k, n])
            if k in self.exterior_connected_rooms:
                if assignment[k] not in exterior_neighbours:
                    return RoomSolverMsg("exterior neighbours unsatisfied", [k])
            if get_room_type(self.graph.rooms[k]) == RoomType.Staircase:
                if (
                    staircase_occupancies[assignment[k]]
                    < self.staircase_occupancy_thresh
                ):
                    return RoomSolverMsg("staircase occupancy unsatisfied", [k])
        return RoomSolverMsg("success")

    def perturb_solution(self, assignment, info):
        k = np.random.choice(list(info["segments"].keys()))
        while True:
            info_ = deepcopy(info)
            assignment_ = deepcopy(assignment)
            try:
                rn = uniform()
                if rn < 1 / 3:
                    resp = self.extrude_room_out(assignment, info, k)
                elif rn < 2 / 3:
                    resp = self.extrude_room_in(assignment, info, k)
                else:
                    resp = self.swap_room(assignment, info, k)
            except Exception:
                info, assignment = info_, assignment_
            else:
                break
        if not resp.is_success:
            return resp
        for c in resp.index_changed:
            if not is_valid_polygon(info["segments"][c]):
                return RoomSolverMsg("invalid segment", [c])
        try:
            for c in resp.index_changed:
                update_shared_edges(info["segments"], info["shared_edges"], c)
                update_exterior_edges(
                    info["segments"], info["shared_edges"], info["exterior_edges"], c
                )
                update_staircase_occupancies(
                    info["segments"],
                    info["staircase"],
                    info["staircase_occupancies"],
                    c,
                )
        except Exception:
            return RoomSolverMsg("Exception")
        info["neighbours_all"] = {
            k: set(compute_neighbours(se, SEGMENT_MARGIN))
            for k, se in info["shared_edges"].items()
        }
        info["exterior_neighbours"] = set(
            compute_neighbours(info["exterior_edges"], SEGMENT_MARGIN)
        )
        for k, s in info["segments"].items():
            x, y = np.array(s.boundary.coords).T
            if np.any((x < -1.0) | (y < -1.0) | (x > 40.0) | (y > 40.0)):
                return RoomSolverMsg("OOB")
        satisfies = self.satisfies_constraints(assignment, info)
        if not satisfies.is_success:
            return satisfies
        return resp

    def extrude_room(self, i, info, out=True):
        segments = info["segments"]
        coords = canonicalize(segments[i]).boundary.coords[:]
        indices = []
        for k in range(len(coords) - 1):
            (x, y), (x_, y_) = coords[k : k + 2]
            if np.abs(x - x_) < 1e-2 and self.x_min < x < self.x_max:
                indices.append(k)
            elif np.abs(y - y_) < 1e-2 and self.y_min < y < self.y_max:
                indices.append(k)
        k = np.random.choice(indices)
        (x, y), (x_, y_) = coords[k : k + 2]
        is_vertical = np.abs(x - x_) < 1e-2
        line = LineString(coords[k : k + 2])
        mod = len(coords) - 1
        stride = constants.UNIT * (np.random.randint(self.max_stride) + 1)
        if is_vertical:
            new_x = x + stride if (y_ < y) ^ out else x - stride
            new_first = new_x, linear_extend_x(coords[(k - 1) % mod], coords[k], new_x)
            new_second = (
                new_x,
                linear_extend_x(coords[(k + 2) % mod], coords[k + 1], new_x),
            )
        else:
            new_y = y + stride if (x_ > x) ^ out else y - stride
            new_first = linear_extend_y(coords[(k - 1) % mod], coords[k], new_y), new_y
            new_second = (
                linear_extend_y(coords[(k + 2) % mod], coords[k + 1], new_y),
                new_y,
            )
        coords[k % mod] = new_first
        coords[(k + 1) % mod] = new_second
        coords[-1] = coords[0]
        s = canonicalize(Polygon(LineString(coords)))
        return s, line, is_vertical

    def extrude_room_out(self, assignment, info, i):
        segments, shared_edges = map(info.get, ["segments", "info"])
        s, _, _ = self.extrude_room(i, info, True)
        if not is_valid_polygon(s):
            return RoomSolverMsg("extrude_room_out_invalid", [i])
        cutter = s.difference(segments[i])
        if not is_valid_polygon(cutter):
            return RoomSolverMsg("extrude_room_out_invalid", [i])
        cutter = canonicalize(cutter)
        shared = list(
            k
            for k in info["shared_edges"][i].keys()
            if segments[k].intersection(cutter).area > 0.1
        )
        index_changed = [i, *shared]
        total_pre_area = sum([segments[i].area for i in index_changed])
        for l in shared:
            segments[l] = canonicalize(segments[l].difference(cutter))
        segments[i] = s
        total_post_area = sum([segments[i].area for i in index_changed])
        if np.abs(total_pre_area - total_post_area) < 0.1:
            return RoomSolverMsg("success", index_changed)
        else:
            return RoomSolverMsg("extrude_room_out_oob", index_changed)

    def extrude_room_in(self, assignment, info, i):
        segments, shared_edges = map(info.get, ["segments", "shared_edges"])
        s, line, is_vertical = self.extrude_room(i, info, False)
        if not is_valid_polygon(s):
            return RoomSolverMsg("extrude_room_in_invalid", [i])
        cutter = segments[i].difference(s)
        if not is_valid_polygon(cutter):
            return RoomSolverMsg("extrude_room_in_invalid", [i])
        cutter = canonicalize(cutter)
        shared = {}
        for k in shared_edges[i].keys():
            with np.errstate(invalid="ignore"):
                forward, backward = shared_paths(segments[k].boundary, line).geoms
            if forward.length > 0:
                shared[k] = forward.geoms[0]
            elif backward.length > 0:
                shared[k] = backward.geoms[0]
        index_changed = [i, *shared.keys()]
        ranges = []
        for k, ls in shared.items():
            if is_vertical:
                y0, y1 = ls.xy[1]
                ranges.append((min(y0, y1), max(y0, y1)))
            else:
                x0, x1 = ls.xy[0]
                ranges.append((min(x0, x1), max(x0, x1)))
        indices = np.argsort([(m + mm) / 2 for m, mm in ranges])
        affected = [list(shared.keys())[_] for _ in indices]
        cuts = [ranges[_][0] for _ in indices[1:]]
        if is_vertical:
            lss = [LineString([(-1, c), (100, c)]) for c in cuts]
        else:
            lss = [LineString([(c, -1), (c, 100)]) for c in cuts]
        polygons = cut_polygon_by_line(cutter, *lss)
        polygons.sort(key=lambda p: p.centroid.coords[0][1 if is_vertical else 0])
        total_pre_area = sum([segments[i].area for i in index_changed])
        for a, p in zip(affected, polygons):
            segments[a] = canonicalize(union(segments[a], p))
        segments[i] = s
        total_post_area = sum([segments[i].area for i in index_changed])
        if np.abs(total_pre_area - total_post_area) < 0.1:
            return RoomSolverMsg("success", index_changed)
        else:
            return RoomSolverMsg("extrude_room_in_oob", index_changed)

    def swap_room(self, assignment, info, i):
        j = np.random.choice(list(info["neighbours_all"][i]))
        j_ = assignment.index(j)
        i_ = assignment.index(i)
        assignment[i_], assignment[j_] = j, i
        return RoomSolverMsg("success", [i, j])


class BlueprintStaircaseSolver:
    def __init__(self, contours, max_stride=1):
        self.contours = contours
        self.max_stride = max_stride
        self.n_trials = 100

    def perturb_solution(self, assignments, infos):
        resp = self.move_staircase(infos)
        if not resp.is_success:
            return resp
        for info in infos:
            for k in info["segments"]:
                update_staircase_occupancies(
                    info["segments"],
                    info["staircase"],
                    info["staircase_occupancies"],
                    k,
                )
        return resp

    def move_staircase(self, infos):
        staircase = infos[0]["staircase"]
        if staircase is None:
            return RoomSolverMsg("success")
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(self.n_trials):
            stride = constants.UNIT * (np.random.randint(self.max_stride) + 1)
            x, y = directions[np.random.randint(4)]
            coords = list(
                (x_ + x * stride, y_ + y * stride)
                for x_, y_ in staircase.boundary.coords[:]
            )
            p = Polygon(LineString(coords))
            if self.contours[-1].contains(p):
                for info in infos:
                    info["staircase"] = p
                return RoomSolverMsg("success")
        else:
            return RoomSolverMsg("invalid staircase")
