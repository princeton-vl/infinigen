import json
import pickle
from collections import defaultdict
from collections.abc import Callable

import shapely
import yaml

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
from infinigen.core.util.test_utils import import_item


class PredefinedBlueprintSolidifier(BlueprintSolidifier):
    def __init__(self, consgraph, config, level):
        super().__init__(consgraph, None, level)
        self.config = config

    def make_interior_cutters(self, neighbours, shared_edges, segments, exterior):
        open_cutters = defaultdict(list)
        opens = self.config.get("opens", {})
        for o, open in opens.items():
            open_cutter = self.make_open_cutter(open["shape"])
            for r, room in self.config["rooms"].items():
                if (
                    shapely.intersection(
                        room["shape"],
                        shapely.buffer(open["shape"], self.constants.wall_thickness),
                    ).area
                    > 0
                ):
                    open_cutters[r].extend(open_cutter)
        interior_cutters = defaultdict(list)
        interiors = self.config.get("interiors", {})
        for i, interior in interiors.items():
            interior_cutter = self.make_window_cutter(
                interior["shape"], interior.get("is_panoramic", False)
            )
            for r, room in self.config["rooms"].items():
                if (
                    shapely.intersection(
                        room["shape"],
                        shapely.buffer(
                            interior["shape"], self.constants.wall_thickness
                        ),
                    ).area
                    > 0
                ):
                    interior_cutters[r].extend(interior_cutter)
        door_cutters = defaultdict(list)
        doors = self.config.get("doors", {})
        for d, door in doors.items():
            door_cutter = self.make_door_cutter(door["shape"], (0, 0, 0))
            for r, room in self.config["rooms"].items():
                if (
                    shapely.intersection(
                        room["shape"],
                        shapely.buffer(door["shape"], self.constants.wall_thickness),
                    ).area
                    > 0
                ):
                    door_cutters[r].append(door_cutter)
        return open_cutters, door_cutters, interior_cutters

    def make_exterior_cutters(self, exterior_edges, exterior_shape):
        window_cutters = defaultdict(list)
        windows = self.config.get("windows", {})
        for w, window in windows.items():
            window_cutter = self.make_window_cutter(
                window["shape"], window.get("is_panoramic", False)
            )
            for r, room in self.config["rooms"].items():
                if (
                    shapely.intersection(
                        room["shape"],
                        shapely.buffer(window["shape"], self.constants.wall_thickness),
                    ).area
                    > 0
                ):
                    window_cutters[r].extend(window_cutter)
        entrance_cutters = defaultdict(list)
        entrances = self.config.get("entrance", {})
        for e, entrance in entrances.items():
            entrance_cutter = self.make_door_cutter(entrance["shape"], (0, 0, 0))
            for r, room in self.config["rooms"].items():
                if (
                    shapely.intersection(
                        room["shape"],
                        shapely.buffer(
                            entrance["shape"], self.constants.wall_thickness
                        ),
                    ).area
                    > 0
                ):
                    entrance_cutters[r].append(entrance_cutter)
        return window_cutters, entrance_cutters


class PredefinedFloorPlanSolver:
    def __init__(self, factory_seed, consgraph, floor_plan="", **kwargs):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.constants = consgraph.constants
            self.n_stories = self.constants.n_stories
            if floor_plan == "" or floor_plan is None:
                raise ValueError("No configuration provided")
            elif isinstance(floor_plan, Callable):
                floor_plan = floor_plan(factory_seed)
            elif not isinstance(floor_plan, str):
                pass
            elif floor_plan.endswith(".json"):
                with open(floor_plan) as f:
                    floor_plan = json.load(open(floor_plan))
            elif floor_plan.endswith(".yaml") or floor_plan.endswith(".yml"):
                with open(floor_plan) as f:
                    floor_plan = yaml.safe_load(f)
            elif floor_plan.endswith(".pickle") or floor_plan.endswith(".pkl"):
                with open(floor_plan, "rb") as f:
                    floor_plan = pickle.load(f)
            else:
                floor_plan = import_item(floor_plan)(factory_seed)
            self.floor_plan = floor_plan
            for _, objs in floor_plan.items():
                for _, info in objs.items():
                    if isinstance(info["shape"], str):
                        info["shape"] = eval(info["shape"])
            max_height = max(room_level(n) for n in self.floor_plan["rooms"]) + 1
            self.solidifiers = [
                PredefinedBlueprintSolidifier(consgraph, self.floor_plan, i)
                for i in range(max_height)
            ]

    def solve(self):
        state = State(
            objs={
                r: ObjectState(polygon=v["shape"])
                for r, v in self.floor_plan["rooms"].items()
            }
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
                st, rooms_meshed = self.solidifiers[j].solidify(
                    State({k: v for k, v in state.objs.items() if room_level(k) == j})
                )
            obj_states.update(st.objs)
        unique_roomtypes = set()
        for s in self.floor_plan["rooms"]:
            unique_roomtypes.add(Semantics(room_type(s)))
        bbox = shapely.union_all(
            [shapely.box(*v["shape"].bounds) for v in self.floor_plan["rooms"].values()]
        ).bounds
        dimensions = (
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            self.constants.wall_height * self.n_stories,
        )
        return State(obj_states), unique_roomtypes, dimensions
