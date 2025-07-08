from collections import defaultdict
from copy import copy

import shapely

from infinigen.core.constraints.example_solver.room.base import (
    room_level,
    room_name,
    room_type,
)
from infinigen.core.constraints.example_solver.room.solidifier import (
    BlueprintSolidifier,
)
from infinigen.core.constraints.example_solver.state_def import ObjectState, State
from infinigen.core.tags import Semantics
from infinigen.core.util import FixedSeed


class PredefinedBlueprintSolidifier(BlueprintSolidifier):
    def __init__(self, consgraph, config, level):
        super().__init__(consgraph, None, level)
        self.config = config

    def make_interior_cutters(self, neighbours, shared_edges, segments, exterior):
        open_cutters = {}
        door_cutters = defaultdict(list)
        interior_cutters = {}
        doors = {
            d: self.make_door_cutter(door, (0, 0, 0))
            for d, door in self.config["doors"].items()
        }
        for r, room in self.config["rooms"].items():
            for d, door in self.config["doors"].items():
                if (
                    shapely.intersection(
                        room, shapely.buffer(door, self.constants.wall_thickness)
                    ).area
                    > 0
                ):
                    door_cutters[r].append(doors[d])
        return open_cutters, door_cutters, interior_cutters

    def make_exterior_cutters(self, exterior_edges, exterior_shape):
        window_cutters = defaultdict(list)
        entrance_cutters = {}
        for r, room in self.config["rooms"].items():
            for w, window in self.config["windows"].items():
                if (
                    shapely.intersection(
                        room, shapely.buffer(window, self.constants.wall_thickness)
                    ).area
                    > 0
                ):
                    window_cutters[r].extend(self.make_window_cutter(window, False))
        return window_cutters, entrance_cutters


class PredefinedFloorPlanSolver:
    def __init__(self, factory_seed, consgraph, config):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.constants = consgraph.constants
            self.n_stories = self.constants.n_stories
            for k, vs in config.items():
                for n in copy(vs):
                    vs[n] = eval(vs[n]["operation"])(
                        *vs[n].get("args", []), **vs[n].get("kwargs", {})
                    )
                self.config = config
            max_height = max(room_level(n) for n in self.config["rooms"]) + 1
            self.solidifiers = [
                PredefinedBlueprintSolidifier(consgraph, self.config, i)
                for i in range(max_height)
            ]

    def solve(self):
        state = State(
            objs={r: ObjectState(polygon=v) for r, v in self.config["rooms"].items()}
        )
        obj_states = {}
        for j in range(self.n_stories):
            with FixedSeed(self.factory_seed):
                exterior = shapely.union_all(
                    list(v.polygon for k, v in state.objs.items() if room_level(k) == j)
                )
                state.objs[room_name(Semantics.Exterior, j)] = ObjectState(
                    polygon=exterior
                )
                shapely.union_all(list(self.config["rooms"].values()))
                st, rooms_meshed = self.solidifiers[j].solidify(
                    State({k: v for k, v in state.objs.items() if room_level(k) == j})
                )
            obj_states.update(st.objs)
        unique_roomtypes = set()
        for s in self.config["rooms"]:
            unique_roomtypes.add(Semantics(room_type(s)))
        bbox = shapely.union_all(
            [shapely.box(*v.bounds) for v in self.config["rooms"].values()]
        ).bounds
        dimensions = (
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            self.constants.wall_height * self.n_stories,
        )
        return State(obj_states), unique_roomtypes, dimensions
