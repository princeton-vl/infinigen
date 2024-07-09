# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

from collections import defaultdict

import gin
import numpy as np
from shapely import LineString, Polygon

import infinigen.core.constraints.example_solver.room.constants as constants
from infinigen.core.constraints.example_solver.room.configs import (
    EXTERIOR_CONNECTED_ROOM_TYPES,
    FUNCTIONAL_ROOM_TYPES,
    SQUARE_ROOM_TYPES,
    TYPICAL_AREA_ROOM_TYPES,
)
from infinigen.core.constraints.example_solver.room.types import RoomType, get_room_type
from infinigen.core.constraints.example_solver.room.utils import (
    abs_distance,
    buffer,
    unit_cast,
)


@gin.configurable(denylist=["graph"])
class BlueprintScorer:
    def __init__(
        self,
        graph,
        shortest_path_weight=2.0,
        typical_area_weight=10.0,
        typical_area_room_types=TYPICAL_AREA_ROOM_TYPES,
        aspect_ratio_weight=10.0,
        aspect_ratio_room_types=SQUARE_ROOM_TYPES,
        convexity_weight=50.0,
        conciseness_weight=2.0,
        exterior_connected_room_types=EXTERIOR_CONNECTED_ROOM_TYPES,
        exterior_length_weight=0.2,
        exterior_corner_weight=0.02,
        collinearity_weight=0.02,
        functional_room_weight=0.2,
        functional_room_types=FUNCTIONAL_ROOM_TYPES,
        narrow_passage_weight=5.0,
        narrow_passage_thresh=1.5,
    ):
        self.graph = graph
        self.shortest_path_weight = shortest_path_weight

        self.typical_area_weight = typical_area_weight
        self.typical_area_room_types = typical_area_room_types

        self.aspect_ratio_weight = aspect_ratio_weight
        self.aspect_ratio_room_types = aspect_ratio_room_types

        self.convexity_weight = convexity_weight

        self.conciseness_weight = conciseness_weight
        self.conciseness_thresh = 4

        self.exterior_length_weight = exterior_length_weight
        self.exterior_connected_room_types = exterior_connected_room_types
        self.exterior_corner_weight = exterior_corner_weight

        self.collinearity_weight = collinearity_weight

        self.functional_room_weight = functional_room_weight
        self.functional_room_types = functional_room_types

        self.narrow_passage_weight = narrow_passage_weight
        self.narrow_passage_thresh = narrow_passage_thresh

    def find_score(self, assignment, info):
        return sum(self.compute_scores(assignment, info).values())

    def compute_scores(self, assignment, info):
        info["neighbours"] = {
            a: set(assignment[_] for _ in self.graph.neighbours[i])
            for i, a in enumerate(assignment)
        }
        scores = {}
        if self.shortest_path_weight > 0:
            score = self.shortest_path_weight * self.shortest_path(assignment, info)
            scores["shortest_path"] = score
        if self.typical_area_weight > 0:
            score = self.typical_area_weight * self.typical_area(assignment, info)
            scores["typical_area"] = score
        if self.aspect_ratio_weight > 0:
            score = self.aspect_ratio_weight * self.aspect_ratio(assignment, info)
            scores["aspect_ratio"] = score
        if self.convexity_weight > 0:
            score = self.convexity_weight * self.convexity(assignment, info)
            scores["convexity"] = score
        if self.conciseness_weight > 0:
            score = self.conciseness_weight * self.conciseness(assignment, info)
            scores["conciseness"] = score
        if self.exterior_length_weight > 0:
            score = self.exterior_length_weight * self.exterior_length(assignment, info)
            scores["exterior_length"] = score
        if self.exterior_corner_weight > 0:
            score = self.exterior_corner_weight * self.exterior_corner(assignment, info)
            scores["exterior_corner"] = score
        if self.collinearity_weight > 0:
            score = self.collinearity_weight * self.collinearity(assignment, info)
            scores["collinearity"] = score
        if self.functional_room_weight > 0:
            score = self.functional_room_weight * self.functional_room(assignment, info)
            scores["functional_room"] = score
        if self.narrow_passage_weight > 0:
            score = self.narrow_passage_weight * self.narrow_passage(assignment, info)
            scores["narrow_passage"] = score
        return scores

    def shortest_path(self, assignment, info):
        shortest_paths = defaultdict(dict)
        centroids = {k: s.centroid.coords[:][0] for k, s in info["segments"].items()}
        for k, ses in info["shared_edges"].items():
            for l, se in ses.items():
                min_distance = np.full(100, 4)
                for ls in se.geoms:
                    for c in ls.coords[:]:
                        dist = abs_distance(centroids[k], c) + abs_distance(
                            c, centroids[l]
                        )
                        if np.sum(dist) <= np.sum(min_distance):
                            min_distance = dist
                shortest_paths[k][l] = min_distance
        roots = self.graph[RoomType.Staircase]
        if self.graph.entrance is not None:
            roots.append(self.graph.entrance)
        scores = {}
        for root in roots:
            root = assignment[root]
            displacement = {a: np.array([1e3] * 4) for a in assignment}
            displacement[root] = np.zeros(4)
            updated = True
            while updated:
                updated = False
                for k, ns in info["neighbours"].items():
                    for n in ns:
                        d = displacement[k] + shortest_paths[k][n]
                        if np.sum(d) < np.sum(displacement[n]):
                            displacement[n] = d
                            updated = True
            displacements = np.stack([d for k, d in displacement.items() if k != root])
            x, xx, y, yy = displacements.T
            score = (
                1.0 / ((np.maximum(x, xx) + np.maximum(y, yy)) / displacements.sum(1))
                - 1
            ) ** 2
            scores[root] = score.sum()
        return sum(s for s in scores.values())

    def typical_area(self, assignment, info):
        total_typical_areas, total_face_areas = [], []
        for i, r in enumerate(self.graph.rooms):
            if get_room_type(r) in self.typical_area_room_types:
                total_typical_areas.append(
                    self.typical_area_room_types[get_room_type(r)]
                )
                total_face_areas.append(info["segments"][assignment[i]].area)
        total_typical_areas = np.array(total_typical_areas)
        total_face_areas = np.array(total_face_areas)
        scores = (
            total_face_areas
            / np.sum(total_face_areas)
            / total_typical_areas
            * np.sum(total_typical_areas)
        )
        scores = np.where(scores > 1, scores, 1 / scores) - 1
        return scores.sum()

    def aspect_ratio(self, assignment, info):
        aspect_ratios = []
        for i, r in enumerate(self.graph.rooms):
            if get_room_type(r) in self.aspect_ratio_room_types:
                x, y, xx, yy = info["segments"][assignment[i]].bounds
                aspect_ratios.append((xx - x) / (yy - y))
        aspect_ratios = np.array(aspect_ratios)
        aspect_ratios = np.where(aspect_ratios > 1, aspect_ratios, 1 / aspect_ratios)
        scores = aspect_ratios - 1
        return scores.sum()

    def convexity(self, assignment, info):
        sharpness = []
        for s in info["segments"].values():
            sharpness.append(s.convex_hull.area / s.area)
        sharpness = np.array(sharpness)
        scores = (sharpness - 1) ** 2
        return scores.sum()

    def conciseness(self, assignment, info):
        conciseness = np.array(
            [len(s.boundary.coords) - 1 for s in info["segments"].values()]
        )
        scores = (conciseness / self.conciseness_thresh - 1) ** 2
        return scores.sum()

    def exterior_length(self, assignment, info):
        exterior_edges = info["exterior_edges"]
        total_length = 0
        for i, r in enumerate(self.graph.rooms):
            if get_room_type(r) in self.exterior_connected_room_types:
                if assignment[i] in exterior_edges:
                    total_length += exterior_edges[assignment[i]].length
        score = total_length / sum(ee.length for ee in exterior_edges.values())
        return (score - 1) ** 2 * len(info["segments"])

    def exterior_corner(self, assignment, info):
        exterior_edges = info["exterior_edges"]
        total_corners, corners = 0, 0
        for i, r in enumerate(self.graph.rooms):
            if assignment[i] in exterior_edges:
                ee = exterior_edges[assignment[i]]
                for e in [ee] if isinstance(ee, LineString) else ee.geoms:
                    n = len(e.coords[:]) - 2
                    corners += n
                    if get_room_type(r) in self.exterior_connected_room_types:
                        total_corners += n
        score = total_corners / corners
        return (score - 1) ** 2 * len(info["segments"])

    def collinearity(self, assignment, info):
        x_skeletons, y_skeletons = set(), set()
        for s in info["segments"].values():
            x, y = s.boundary.xy
            for i in range(len(x) - 1):
                if np.abs(x[i] - x[i + 1]) < 1e-2:
                    x_skeletons.add(unit_cast(x[i]))
                elif np.abs(y[i] - y[i + 1]) < 1e-2:
                    y_skeletons.add(unit_cast(y[i]))
        score = len(x_skeletons) + len(y_skeletons)
        return score * len(info["segments"])

    def functional_room(self, assignment, info):
        total_area = 0
        segments = info["segments"]
        for i, r in enumerate(self.graph.rooms):
            if get_room_type(r) in self.functional_room_types:
                total_area += segments[assignment[i]].area
        score = total_area / sum(s.area for s in segments.values())
        return (1 - score) ** 2 * len(info["segments"])

    def narrow_passage(self, assignment, info):
        scores = []
        for p in info["segments"].values():
            for d in np.arange(1, int(self.narrow_passage_thresh / constants.UNIT)):
                with np.errstate(invalid="ignore"):
                    length = d * constants.UNIT / 2
                    b = buffer(p, -length)
                    c = buffer(b, length)
                scores.append(
                    p.area
                    - c.area
                    + (
                        self.narrow_passage_thresh**2 * 20
                        if not isinstance(b, Polygon)
                        else 0
                    )
                )
        scores = np.array(scores).sum()
        return scores


@gin.configurable(denylist=["graphs"])
class JointBlueprintScorer:
    def __init__(
        self,
        graphs,
        *args,
        staircase_occupancy_weight=1.0,
        staircase_iou_weight=0.5,
        **kwargs,
    ):
        self.scorers = []
        self.graphs = graphs
        for g in self.graphs:
            self.scorers.append(BlueprintScorer(g, *args, **kwargs))
        self.staircase_occupancy_weight = staircase_occupancy_weight
        self.staircase_iou_weight = staircase_iou_weight

    def compute_scores(self, assignments, infos):
        scores = {}
        for i, (assignment, info) in enumerate(zip(assignments, infos)):
            floor_scores = self.scorers[i].compute_scores(assignment, info)
            scores.update({f"{k}_{i:01d}": v for k, v in floor_scores.items()})
        if len(self.graphs) > 1:
            if self.staircase_occupancy_weight > 0:
                score = self.staircase_occupancy_weight * self.staircase_occupancy(
                    assignments, infos
                )
                scores["staircase_occupancy"] = score
            if self.staircase_iou_weight > 0:
                score = self.staircase_iou_weight * self.staircase_iou(
                    assignments, infos
                )
                scores["staircase_iou"] = score
        return scores

    def find_score(self, assignments, infos):
        return sum(self.compute_scores(assignments, infos).values())

    def staircase_occupancy(self, assignments, infos):
        scores = []
        for graph, assignment, info in zip(self.graphs, assignments, infos):
            for _ in graph[RoomType.Staircase]:
                scores.append(info["staircase_occupancies"][assignment[_]])
        scores = np.array(scores)
        return ((scores - 1) ** 2).sum() * sum(len(info["segments"]) for info in infos)

    def staircase_iou(self, assignments, infos):
        scores = []
        for graph, assignment, info in zip(self.graphs, assignments, infos):
            for _ in graph[RoomType.Staircase]:
                segment = info["segments"][assignment[_]]
                staircase = info["staircase"]
                scores.append(
                    segment.intersection(staircase).area / segment.union(staircase).area
                )
        scores = np.array(scores)
        return ((scores - 1) ** 2).sum() * sum(len(info["segments"]) for info in infos)
