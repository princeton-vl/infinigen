# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick: state, print, to_json
# - Karhan Kayan: add dof / trimesh

from __future__ import annotations

import enum
import importlib
import json
import logging
import pickle
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import bpy
import numpy as np
import shapely
import trimesh

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver.geometry.planes import Planes
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import int_hash

from .geometry import parse_scene
from .room.base import RoomGraph

logger = logging.getLogger(__name__)


@dataclass
class RelationState:
    relation: cl.Relation
    target_name: str
    child_plane_idx: int = None
    parent_plane_idx: int = None
    value: typing.Optional[shapely.MultiLineString] = None


@dataclass
class ObjectState:
    obj: bpy.types.Object = None
    polygon: shapely.Polygon = None
    generator: typing.Optional[AssetFactory] = None
    tags: set = field(default_factory=set)
    relations: list[RelationState] = field(default_factory=list)

    dof_matrix_translation: np.array = None
    dof_rotation_axis: np.array = None
    _pose_affects_score = None

    fcl_obj = None
    col_obj = None

    # store whether this object is active for the current greedy stage
    # inactive objects arent returned by scene() and arent accessible through blender (for perf)
    # updated by greedy.update_active_flags()
    active: bool = True

    def __post_init__(self):
        assert not t.contradiction(self.tags)
        assert not any(
            isinstance(r.relation, cl.NegatedRelation) for r in self.relations
        ), self.relations

    def __repr__(self):
        obj = self.obj
        tags = self.tags
        relations = self.relations
        return f"{self.__class__.__name__}(obj.name={obj.name if obj is not None else None}, polygon={self.polygon}, {tags=}, {relations=})"


@dataclass
class State:
    objs: OrderedDict[str, ObjectState] = field(default_factory=dict)

    trimesh_scene: trimesh.Scene = None
    graphs: list[RoomGraph] = field(default_factory=list)
    bvh_cache: dict = field(default_factory=dict)
    planes: Planes = None

    def __getitem__(self, item):
        return self.objs[item]

    def __setitem__(self, key, value):
        self.objs[key] = value

    def __delitem__(self, key):
        del self.objs[key]

    def __len__(self):
        return len(self.objs)

    def print(self):
        print(f"State ({len(self.objs)} objs)")
        order = sorted(self.objs.keys(), key=lambda s: s.split("_")[-1])
        for k in order:
            v = self.objs[k]
            relations = ", ".join(
                f"{r.relation.__class__.__name__}({r.target_name})" for r in v.relations
            )
            semantics = {
                tg
                for tg in t.decompose_tags(v.tags)[0]
                if not isinstance(tg, t.SpecificObject)
            }
            print(f"  {v.obj.name} {semantics} [{relations}]")

    def to_json(self, path: Path):
        JSON_SUPPORTED_TYPES = (int, float, str, bool, list, dict)

        def preprocess_field(x):
            match x:
                case np.ndarray():
                    return x.tolist()
                case np.int64():
                    return x.item()
                case t.Tag():
                    return str(x)
                case bpy.types.Object():
                    return x.name
                case enum.Enum():
                    return x.name
                case type():
                    return x.__module__ + "." + x.__name__
                case set() | frozenset():
                    return list(x)
                case val if isinstance(val, JSON_SUPPORTED_TYPES):
                    return x
                case AssetFactory():
                    return repr(x)
                case ObjectState() | RelationState():
                    return x.__dict__
                case cl.Relation():
                    res = x.__dict__
                    res["relation_type"] = x.__class__.__name__
                    return res
                case _:
                    return "<not-serialized>"

        data = {
            "objs": self.objs,
        }

        with path.open("w") as f:
            json.dump(
                data,
                f,
                default=preprocess_field,
                sort_keys=True,
                indent=4,
                check_circular=True,
            )

    def __post_init__(self):
        bpy_objs = [
            o.obj
            for o in self.objs.values()
            if o.obj is not None and isinstance(o.obj, bpy.types.Object)
        ]
        self.trimesh_scene = parse_scene.parse_scene(bpy_objs)
        self.planes = Planes()

    def save(self, filename: str):
        return
        # serialize objs and python modules
        for os in self.objs.values():
            os.obj = os.obj.name
            if os.generator is not None:
                path = os.generator.__module__ + "." + os.generator.__name__
                os.generator = path

        with open(filename, "wb") as file:
            pickle.dump(self, file)

        for os in self.objs.values():
            os.obj = bpy.data.objects[os.obj]

            if os.generator is not None:
                *mod, name = os.generator.split(".")
                mod = importlib.import_module(".".join(mod))
                os.generator = getattr(mod, name)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as file:
            state = pickle.load(file)

        # all objs were serialized as strings, unpack them
        for o in state.objs:
            if o.obj not in bpy.data.objects:
                raise ValueError(
                    f"While deserializing {filename}, found name {o.obj=} which "
                    "isnt present in current blend scene. Did you load the "
                    "correct blend before loading the state?"
                )
            o.obj = bpy.data.objects[o.obj]

    def __hash__(self):
        return sum(int_hash(k) * int(o.polygon.area) for k, o in self.objs.items())


def state_from_dummy_scene(col: bpy.types.Collection) -> State:
    objs = {}
    for obj in col.all_objects:
        obj.rotation_mode = "AXIS_ANGLE"
        tags = {t.Semantics(c.name) for c in col.children if obj.name in c.objects}
        tags.add(t.SpecificObject(obj.name))
        objs[obj.name] = ObjectState(obj=obj, generator=None, tags=tags)
    return State(objs=objs)
