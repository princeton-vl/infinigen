# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
from dataclasses import dataclass

from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.geometry import parse_scene
from infinigen.core.constraints.example_solver.moves.moves import Move
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


@dataclass
class Deletion(Move):
    # remove obj from scene
    _backup_state: state_def.ObjectState = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.names})"

    def apply(self, state):
        (target_name,) = self.names
        self._backup_state = state.objs[target_name]

        for obj in butil.iter_object_tree(state.objs[target_name].obj):
            state.trimesh_scene.graph.transforms.remove_node(obj.name)
            state.trimesh_scene.delete_geometry(obj.name + "_mesh")

        del state.objs[target_name]
        return True

    def accept(self, state):
        butil.delete(list(butil.iter_object_tree(self._backup_state.obj)))

    def revert(self, state):
        (target_name,) = self.names
        state.objs[target_name] = self._backup_state
        parse_scene.add_to_scene(
            state.trimesh_scene, self._backup_state.obj, preprocess=True
        )
