# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import json

# import pytest
from mathutils import Vector
from test_stable_against import make_scene


def test_state_to_json(tmp_path):
    state = make_scene(Vector((1, 0, 0)))

    path = tmp_path / "state.json"
    state.to_json(path)

    with path.open() as json_file:
        state_json = json.load(json_file)

    assert sorted(list(state_json["objs"].keys())) == ["cup", "table"]
    assert len(state_json["objs"]["cup"]["relations"]) == 1
