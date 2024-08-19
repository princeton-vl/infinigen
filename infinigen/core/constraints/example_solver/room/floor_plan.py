# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei

import gin
import numpy as np
import shapely
from numpy.random import uniform
from shapely.affinity import translate
from tqdm import tqdm, trange

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.evaluator.evaluate import evaluate_problem
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.tags import Semantics
from infinigen.core.util.math import FixedSeed

from .base import room_level, room_type
from .contour import ContourFactory
from .graph import GraphMaker
from .segment import SegmentMaker
from .solidifier import BlueprintSolidifier
from .solver import FloorPlanMoves


@gin.configurable
class FloorPlanSolver:
    def __init__(self, factory_seed, consgraph, n_divide_trials=100, iters_mult=200):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.constants = consgraph.constants
            self.consgraph = consgraph
            self.n_stories = self.constants.n_stories
            self.fixed_contour = self.constants.fixed_contour
            self.contour_factory = ContourFactory(self.consgraph)
            self.n_trials = 100
            self.graphs = []
            self.widths, self.heights = [], []
            self.contours = []

            self.build_graphs()
            self.segment_makers = [
                SegmentMaker(
                    factory_seed,
                    self.constants,
                    consgraph,
                    self.contours[i],
                    self.graphs[i],
                    i,
                )
                for i in range(self.n_stories)
            ]

            self.solver = FloorPlanMoves(self.constants)
            self.solidifiers = [
                BlueprintSolidifier(consgraph, g, i) for i, g in enumerate(self.graphs)
            ]

            self.n_divide_trials = n_divide_trials
            self.iter_per_room = iters_mult
            self.score_scale = 5
            self.staircase_solver_prob = 0.1

    def build_graphs(self):
        for i in range(self.n_stories):
            graph_maker = GraphMaker(self.factory_seed, self.consgraph, i)
            if self.fixed_contour and i > 0:
                width, height = self.widths[-1], self.heights[-1]
                graph = graph_maker.make_graph(np.random.randint(1e6))
            else:
                for j in range(self.n_trials):
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
            self.graphs.append(graph)
            while len(self.contours) <= i:
                for j in range(self.n_trials):
                    if self.fixed_contour and i > 0:
                        self.contours.append(self.contours[-1])
                        break
                    contour = shapely.box(0, 0, width, height)
                    if len(self.contours) > 0:
                        x_offset = self.constants.unit_cast(
                            (width - self.widths[0]) * uniform(0, 1)
                        )
                        y_offset = self.constants.unit_cast(
                            (height - self.heights[0]) * uniform(0, 1)
                        )
                        contour = translate(contour, -x_offset, -y_offset)
                        if not self.contours[-1].contains(contour):
                            continue
                    self.contours.append(contour)
                    break
                else:
                    if width / height > self.widths[-1] / self.heights[-1]:
                        width -= self.constants.unit
                    else:
                        height -= self.constants.unit

            self.widths.append(width)
            self.heights.append(height)

    def solve(self):
        state = State(graphs=self.graphs)
        states = []
        while len(states) < self.n_stories:
            pholder = self.contour_factory.add_staircase(self.contours[-1])
            state.objs = {}
            states = []
            for j in range(self.n_stories):
                for _ in trange(
                    self.n_divide_trials * (j + 1) ** 2,
                    desc=f"Dividing segments for {j}",
                ):
                    st = self.segment_makers[j].build_segments(pholder)
                    if st is not None:
                        states.append(st)
                        state.objs.update(st.objs)
                        break
                else:
                    break

        state = self.simulated_anneal(state)
        self.contour_factory.decorate(state)

        obj_states = {}
        for j in range(self.n_stories):
            with FixedSeed(self.factory_seed):
                st, rooms_meshed = self.solidifiers[j].solidify(
                    State({k: v for k, v in state.objs.items() if room_level(k) == j})
                )
            obj_states.update(st.objs)
        unique_roomtypes = set()
        for graph in self.graphs:
            for s in graph.names:
                unique_roomtypes.add(Semantics(room_type(s)))
        dimensions = (
            self.widths[0],
            self.heights[0],
            self.constants.wall_height * self.n_stories,
        )
        return State(obj_states), unique_roomtypes, dimensions

    def simulated_anneal(self, state):
        consgraph = self.consgraph.filter("room")
        consgraph.constraints["graph"] = cl.graph_coherent(self.consgraph.constants)
        score, _ = evaluate_problem(consgraph, state, memo={})
        it = self.iter_per_room * sum(len(g) for g in self.graphs)
        with tqdm(total=it, desc="Sampling solutions") as pbar:
            while pbar.n < it:
                state_ = self.solver.perturb_state(state)
                score_, violated_ = evaluate_problem(consgraph, state_, memo={})
                scale = self.score_scale * pbar.n / it
                if np.log(uniform()) < (score - score_) * scale and not violated_:
                    state = state_
                    score = score_
                pbar.update(1)
                pbar.set_postfix(score=score)
        return state
