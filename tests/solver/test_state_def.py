# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import json
import os
import sys
from functools import partial
from itertools import chain

# import pytest
import bpy
import numpy as np
from mathutils import Vector
from test_stable_against import make_scene

from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import example_solver as solver
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.geometry import dof, parse_scene, planes, stability, validity
from infinigen.core.util import blender as butil


def test_state_to_json(tmp_path):
    state = make_scene(Vector((1, 0, 0)))

    path = tmp_path / "state.json"
    state.to_json(path)

    with path.open() as json_file:
        state_json = json.load(json_file)

    assert sorted(list(state_json["objs"].keys())) == ["cup", "table"]
    assert len(state_json["objs"]["cup"]["relations"]) == 1
