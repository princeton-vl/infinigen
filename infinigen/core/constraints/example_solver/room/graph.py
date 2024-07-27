# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import operator
from copy import deepcopy

import gin
import numpy as np
import scipy.special
import shapely
from numpy.random import uniform
from tqdm import tqdm

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.constraint_language import Problem
from infinigen.core.constraints.evaluator.evaluate import (
    evaluate_problem,
)
from infinigen.core.constraints.example_solver.room.base import (
    RoomGraph,
    room_name,
    room_type,
)
from infinigen.core.constraints.example_solver.room.utils import update_exterior
from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)
from infinigen.core.tags import Semantics
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


@gin.configurable
class GraphMaker:
    def __init__(self, factory_seed, consgraph, level, fast=False):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.level = level
            self.constants = consgraph.constants
            self.typical_areas = self.get_typical_areas(consgraph)
            consgraph = consgraph.filter("node")
            self.fast = fast
            self.consgraph = Problem(
                {"node": consgraph.constraints["node"]},
                {"node_gen": self.inject(consgraph.constraints["node_gen"])},
                consgraph.constants,
            )
            self.max_samples = 1000
            self.slackness = log_uniform(1.1, 1.3)

    @property
    def semantics_floor(self):
        return Semantics.floors[self.level]

    def inject(self, node, on=False):
        match node:
            case cl.in_range(count, low, high, mean) if mean > 0:
                size = high - low
                if size > 0:
                    p = (mean - low) / size
                    if self.fast:
                        p /= 2
                    return cl.rand(
                        self.inject(count, True),
                        "cat",
                        [0] * low
                        + [
                            p**i * (1 - p) ** (size - i) * scipy.special.comb(size, i)
                            for i in range(size + 1)
                        ],
                    )
                else:
                    assert low == int(low)
                    return cl.rand(
                        self.inject(count, True), "cat", [0] * int(low) + [1]
                    )
            case cl.scene():
                return cl.scene() if on else cl.scene()[-Semantics.New]
            case cl.ForAll(objs, var, pred):
                return cl.SumOver(self.inject(objs), var, self.inject(pred, on))
            case cl.SumOver(objs, var, pred) | cl.MeanOver(objs, var, pred):
                return node.__class__(self.inject(objs), var, self.inject(pred, on))
            case cl.BoolOperatorExpression(operator.and_, operands):
                return cl.ScalarOperatorExpression(operator.add, self.inject(operands))
            case cl.Node():
                first = next(iter(node.__dict__))
                return node.__class__(
                    **{
                        k: self.inject(v, on and k == first)
                        for k, v in node.__dict__.items()
                    }
                )
            case _ if isinstance(node, list):
                return list(self.inject(n, on) for n in node)
            case _ if isinstance(node, dict):
                return {k: self.inject(n, on) for k, n in node.items()}
            case _:
                return node

    def make_graph(self, i):
        with FixedSeed(i):
            while True:
                name = room_name(Semantics.Root, self.level)
                state = State(
                    {
                        name: ObjectState(
                            tags={
                                Semantics.Root,
                                Semantics.RoomContour,
                                self.semantics_floor,
                            }
                        )
                    }
                )
                for _ in tqdm(range(40), desc=f"Generating graphs for {self.level}: "):
                    unvisited = list(
                        sorted(
                            k
                            for k, obj_st in state.objs.items()
                            if Semantics.Visited not in obj_st.tags
                        )
                    )
                    if len(unvisited) == 0:
                        break
                    n = unvisited[np.random.randint(len(unvisited))]
                    score, _ = evaluate_problem(
                        self.consgraph, state, {}, enable_violated=False
                    )
                    scores = [score]
                    states = [state]
                    for t in list(sorted(self.constants.room_types)) + [
                        Semantics.Entrance,
                        Semantics.Exterior,
                    ]:
                        count = len(list(k for k in state.objs if room_type(k) == t))
                        for i in range(1, 3):
                            st = deepcopy(state)
                            for j in range(i):
                                name = room_name(t, self.level, count + j)
                                st[name] = ObjectState(
                                    tags={
                                        t,
                                        Semantics.RoomContour,
                                        Semantics.New,
                                        self.semantics_floor,
                                    },
                                    relations=[RelationState(cl.Traverse(), n)],
                                )
                                st[n].relations.append(
                                    RelationState(cl.Traverse(), name)
                                )
                            score, _ = evaluate_problem(
                                self.consgraph, st, {}, enable_violated=False
                            )
                            states.append(st)
                            scores.append(score)
                        scores_ = np.array(scores) - np.min(scores)
                        if np.all(scores_ < 0.01):
                            i = 0
                        else:
                            i = np.random.choice(
                                np.arange(len(scores)),
                                p=np.exp(-scores_) / np.exp(-scores_).sum(),
                            )
                        state = states[i]
                        scores = [scores[i]]
                        states = [state]
                    for k, obj_st in state.objs.items():
                        if Semantics.New in obj_st.tags:
                            obj_st.tags.remove(Semantics.New)
                    if room_type(n) == Semantics.Root:
                        state.objs.pop(n)
                        first = next(iter(state.objs))
                        for k, obj_st in state.objs.items():
                            if k == first:
                                obj_st.relations = [
                                    RelationState(cl.Traverse(), l)
                                    for l in state.objs
                                    if l != first
                                ]
                            else:
                                obj_st.relations = [RelationState(cl.Traverse(), first)]
                    else:
                        state[n].tags.add(Semantics.Visited)
                _, viol = evaluate_problem(self.consgraph, state)
                if viol == 0:
                    return self.state2graph(state)

    def state2graph(self, state):
        state = self.merge_exterior(state)
        state, entrance = self.merge_entrance(state)
        names = [k for k in state.objs.keys() if room_type(k) != Semantics.Exterior] + [
            room_name(Semantics.Exterior, self.level)
        ]
        return RoomGraph(
            [[names.index(r.target_name) for r in state[n].relations] for n in names],
            names,
            None if entrance is None else names.index(entrance),
        )

    def merge_exterior(self, state):
        exterior_connected = set()
        for k, obj_st in state.objs.items():
            if room_type(k) == Semantics.Exterior:
                for r in obj_st.relations:
                    exterior_connected.add(r.target_name)
        exterior_name = room_name(Semantics.Exterior, self.level)
        state = State(
            {
                k: obj_st
                for k, obj_st in state.objs.items()
                if room_type(k) != Semantics.Exterior
            }
        )
        for k in exterior_connected:
            state[k].relations = [
                r
                for r in state[k].relations
                if room_type(r.target_name) != Semantics.Exterior
            ]
            state[k].relations.append(RelationState(cl.Traverse(), exterior_name))
        state[exterior_name] = ObjectState(
            tags={Semantics.Exterior, Semantics.RoomContour, self.semantics_floor},
            relations=[RelationState(cl.Traverse(), k) for k in exterior_connected],
        )
        return state

    def merge_entrance(self, state):
        entrance_connected = set()
        for k, obj_st in state.objs.items():
            if room_type(k) == Semantics.Entrance:
                for r in obj_st.relations:
                    entrance_connected.add(r.target_name)
        state = State(
            {
                k: obj_st
                for k, obj_st in state.objs.items()
                if room_type(k) != Semantics.Entrance
            }
        )
        for k in entrance_connected:
            state[k].relations = [
                r
                for r in state[k].relations
                if room_type(r.target_name) != Semantics.Entrance
            ]
        if len(entrance_connected) == 0:
            entrance = None
        else:
            entrance = np.random.choice(list(entrance_connected))
            exterior_name = room_name(Semantics.Exterior, self.level)
            state[entrance].relations.append(
                RelationState(cl.Traverse(), exterior_name)
            )
            if exterior_name not in state.objs:
                state[exterior_name] = ObjectState(
                    tags={
                        Semantics.Exterior,
                        Semantics.RoomContour,
                        self.semantics_floor,
                    }
                )
            state[exterior_name].relations.append(
                RelationState(cl.Traverse(), entrance)
            )
        return state, entrance

    __call__ = make_graph

    def suggest_dimensions(self, graph, width=None, height=None):
        area = (
            sum(
                [
                    self.typical_areas[room_type(r)]
                    for r in graph.names
                    if room_type(r) != Semantics.Exterior
                ]
            )
            * self.slackness
        )
        if width is None and height is None:
            aspect_ratio = uniform(*self.constants.aspect_ratio_range)
        else:
            aspect_ratio = width / height
        width = self.constants.unit_cast(np.sqrt(area * aspect_ratio).item())
        height = self.constants.unit_cast(np.sqrt(area / aspect_ratio).item())
        return width, height

    def draw(self, state):
        graph = self.state2graph(state)
        graph.draw()

    def get_typical_areas(self, consgraph):
        consgraph = consgraph.filter("room")
        typical_areas = {}
        undecided = set()
        for t in tqdm(self.constants.room_types, "Computing typical areas: "):
            name = room_name(t, self.level)
            holder = room_name(Semantics.Staircase, self.level)
            exterior = room_name(Semantics.Exterior, self.level)
            state = State(
                {
                    name: ObjectState(
                        tags={Semantics.RoomContour, t, self.semantics_floor}
                    ),
                    holder: ObjectState(
                        tags={
                            Semantics.RoomContour,
                            Semantics.Staircase,
                            self.semantics_floor,
                        }
                    ),
                    exterior: ObjectState(
                        tags={
                            Semantics.RoomContour,
                            Semantics.Exterior,
                            Semantics.Garage,
                        },
                        relations=[RelationState(cl.SharedEdge(), name)],
                    ),
                },
                graphs=[RoomGraph([[]], [name], 0)] * (self.level + 1),
            )
            scores = []
            lengths = np.exp(np.linspace(np.log(1.5), np.log(25), 20))
            for l in lengths:
                state.objs[name].polygon = shapely.box(0, 0, l, l)
                state.objs[holder].polygon = shapely.box(-l, -l, 0, 0)
                state.objs[exterior].polygon = shapely.box(-l, -l, 0, 0)
                update_exterior(state, name)
                score, _ = evaluate_problem(consgraph, state)
                scores.append(score)
            scores = np.array(scores)
            selection = (scores - np.min(scores)) < 1
            if np.sum(selection) > len(selection) / 2:
                undecided.add(t)
            else:
                typical_areas[t] = np.exp(np.log(lengths[selection]).mean()) ** 2
        if len(typical_areas) > 0:
            m = np.mean([v for t, v in typical_areas.items()])
        else:
            m = 10
        for t in undecided:
            typical_areas[t] = m
        return typical_areas
