# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei, Karhan Kayan: fix constants

from copy import deepcopy

import gin
import numpy as np
from numpy.random import uniform
from shapely import Polygon
from tqdm import tqdm, trange

from infinigen.core.constraints.constraint_language import Semantics
from infinigen.core.constraints.example_solver.room import constants
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_general as rg

from .constants import WALL_HEIGHT
from .contour import ContourFactory
from .graph import GraphMaker
from .scorer import BlueprintScorer, JointBlueprintScorer
from .segment import SegmentMaker
from .solidifier import BlueprintSolidifier
from .solver import BlueprintSolver, BlueprintStaircaseSolver
from .utils import unit_cast


@gin.configurable
class RoomSolver:
    def __init__(
        self,
        factory_seed,
        n_divide_trials=2500,
        iters_mult=150,
    ):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.graph_maker = GraphMaker(factory_seed)
            self.graph = self.graph_maker.make_graph(np.random.randint(1e7))
            self.width, self.height = self.graph_maker.suggest_dimensions(self.graph)
            self.contour_factory = ContourFactory(self.width, self.height)
            self.contour = self.contour_factory.make_contour(np.random.randint(1e7))

            n = len(self.graph.neighbours)

            self.segment_maker = SegmentMaker(self.factory_seed, self.contour, n)
            self.solver = BlueprintSolver(self.contour, self.graph)
            self.scorer = BlueprintScorer(self.graph)
            self.solidifier = BlueprintSolidifier(self.graph, 0)

            self.n_divide_trials = n_divide_trials
            self.iterations = iters_mult * n
            self.score_scale = 5

    def simulated_anneal(self, assignment, info):
        score = self.scorer.find_score(assignment, info)
        with tqdm(total=self.iterations, desc="Sampling solutions") as pbar:
            while pbar.n < self.iterations:
                assignment_, info_ = deepcopy(assignment), deepcopy(info)
                resp = self.solver.perturb_solution(assignment_, info_)
                if not resp.is_success:
                    continue
                pbar.update(1)
                score_ = self.scorer.find_score(assignment_, info_)
                scale = self.score_scale * pbar.n / self.iterations
                if np.log(uniform()) < (score - score_) * scale:
                    assignment, info, score = assignment_, info_, score_
                    pbar.set_description(f"loss={score:.4f}")

        return assignment, info

    def solve(self):
        assignment, info = [], {}
        for i in range(self.n_divide_trials):
            info = self.segment_maker.build_segments()
            assignment = self.solver.find_assignment(info)
            if assignment is not None:
                break

        if assignment is None:
            raise ValueError(
                f"{self.__class__.__name__} got {assignment=} after {self.n_divide_trials=}"
            )

        assignment, info = self.simulated_anneal(assignment, info)

        state, rooms_meshed = self.solidifier.solidify(assignment, info)

        unique_roomtypes = set(Semantics(s.split("_")[0]) for s in self.graph.rooms)
        dimensions = self.width, self.height, constants.WALL_HEIGHT

        return state, unique_roomtypes, dimensions


@gin.configurable
class MultistoryRoomSolver:
    def __init__(
        self,
        factory_seed,
        n_divide_trials=2500,
        iters_mult=150,
        n_stories=("categorical", 0.0, 0.0, 0.5, 0.5),
        fixed_contour=("bool", 0.5),
    ):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.n_stories = rg(n_stories)
            self.fixed_contour = rg(fixed_contour)
            self.n_contour_trials = 100
            self.graph_makers, self.graphs = [], []
            self.widths, self.heights = [], []
            self.build_graphs(factory_seed)
            self.contour_factories, self.contours = [], []
            self.build_contours()

            self.segment_makers = [
                SegmentMaker(self.factory_seed, self.contours[i], len(self.graphs[i]))
                for i in range(self.n_stories)
            ]
            self.solvers = [
                BlueprintSolver(self.contours[i], self.graphs[i])
                for i in range(self.n_stories)
            ]
            self.staircase_solver = BlueprintStaircaseSolver(self.contours)
            self.scorer = JointBlueprintScorer(self.graphs)
            self.solidifiers = [
                BlueprintSolidifier(self.graphs[i], i) for i in range(self.n_stories)
            ]

            self.n_divide_trials = n_divide_trials
            self.iterations = iters_mult * sum(len(g) for g in self.graphs)
            self.score_scale = 5
            self.staircase_solver_prob = 0.1

    def build_graphs(self, factory_seed):
        for i in range(self.n_stories):
            kwargs = {"entrance_type": "none"} if i > 0 else {}
            graph_maker = GraphMaker(factory_seed, i, self.n_stories > 1, **kwargs)
            self.graph_makers.append(graph_maker)
            if self.fixed_contour and i > 0:
                width, height = self.widths[-1], self.heights[-1]
                graph = graph_maker.make_graph(np.random.randint(1e6))
            else:
                for j in range(self.n_contour_trials):
                    graph = graph_maker.make_graph(np.random.randint(1e6))
                    args = (
                        [self.widths[-1], self.heights[-1]]
                        if len(self.graphs) > 0
                        else [None, None]
                    )
                    width, height = graph_maker.suggest_dimensions(graph, *args)
                    if width is not None and height is not None:
                        break
                else:
                    raise Exception("Invalid graph")
            self.widths.append(width)
            self.heights.append(height)
            self.graphs.append(graph)

    def build_contours(self):
        for i in range(self.n_stories):
            while len(self.contours) <= i:
                for j in range(self.n_contour_trials):
                    contour_factory = ContourFactory(self.widths[i], self.heights[i])
                    if self.fixed_contour and i > 0:
                        contour = self.contours[-1]
                    else:
                        contour = contour_factory.make_contour(np.random.randint(1e6))
                        if len(self.contours) > 0:
                            x_offset = unit_cast((self.widths[i] - self.widths[0]) / 2)
                            y_offset = unit_cast(
                                (self.heights[i] - self.heights[0]) / 2
                            )
                            contour = Polygon(
                                [
                                    (x - x_offset, y - y_offset)
                                    for x, y in contour.boundary.coords[:]
                                ]
                            )
                            if not self.contours[-1].contains(contour):
                                continue
                    self.contour_factories.append(contour_factory)
                    self.contours.append(contour)
                    break
                else:
                    self.widths[i] -= constants.UNIT
                    self.heights[i] -= constants.UNIT

    def solve(self):
        assignments, infos = [], []
        while len(assignments) == 0:
            staircase = self.contour_factories[-1].add_staircase(self.contours[-1])
            for j in range(self.n_stories):
                for _ in trange(
                    self.n_divide_trials, desc=f"Dividing segments for {j}"
                ):
                    info = self.segment_makers[j].build_segments(staircase)
                    assignment = self.solvers[j].find_assignment(info)
                    if assignment is not None:
                        assignments.append(assignment)
                        infos.append(info)
                        break
                else:
                    assignments, infos = [], []
                    break

        assignments, infos = self.simulated_anneal(assignments, infos)

        obj_states = {}
        for j in range(self.n_stories):
            state, rooms_meshed = self.solidifiers[j].solidify(assignments[j], infos[j])
            obj_states.update(state.objs)
        unique_roomtypes = set()
        for graph in self.graphs:
            for s in graph.rooms:
                unique_roomtypes.add(Semantics(s.split("_")[0]))
        dimensions = self.widths[0], self.heights[0], WALL_HEIGHT * self.n_stories
        return State(obj_states), unique_roomtypes, dimensions

    def simulated_anneal(self, assignments, infos):
        score = self.scorer.find_score(assignments, infos)
        with tqdm(total=self.iterations, desc="Sampling solutions") as pbar:
            while pbar.n < self.iterations:
                assignments_, infos_ = deepcopy(assignments), deepcopy(infos)
                if uniform() < self.staircase_solver_prob:
                    resp = self.staircase_solver.perturb_solution(assignments, infos)
                else:
                    probs = np.array([len(g) for g in self.graphs])
                    j = np.random.choice(
                        np.arange(self.n_stories), p=probs / probs.sum()
                    )
                    resp = self.solvers[j].perturb_solution(assignments_[j], infos_[j])
                if not resp.is_success:
                    continue
                pbar.update(1)
                score_ = self.scorer.find_score(assignments_, infos_)
                scale = self.score_scale * pbar.n / self.iterations
                if np.log(uniform()) < (score - score_) * scale:
                    assignments, infos, score = assignments_, infos_, score_
        return assignments, infos
