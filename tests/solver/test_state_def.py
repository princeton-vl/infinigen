# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from itertools import chain
from functools import partial
import json

# import pytest
import bpy 
import numpy as np
import sys 
import os

from infinigen.core.constraints.example_solver.geometry import dof, parse_scene, planes, stability, validity
from mathutils import Vector

from infinigen.core.constraints import (
    usage_lookup,
    example_solver as solver,
    constraint_language as cl
)
from infinigen.core import tagging, tags as t
from infinigen.core.util import blender as butil
from infinigen.core.constraints.example_solver import (
    state_def
)

from test_stable_against import make_scene

def test_state_to_json(tmp_path):

    state = make_scene(Vector((1, 0, 0)))
    
    path = tmp_path/'state.json'
    state.to_json(path)

    with path.open() as json_file:
        state_json = json.load(json_file)

    assert sorted(list(state_json['objs'].keys())) == ['cup', 'table']
    assert len(state_json['objs']['cup']['relations']) == 1