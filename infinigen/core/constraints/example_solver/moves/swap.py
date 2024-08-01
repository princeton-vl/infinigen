# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
from dataclasses import dataclass

from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.moves import Move

from .reassignment import pose_backup, restore_pose_backup

logger = logging.getLogger(__name__)


@dataclass
class Swap(Move):
    # swap the poses and relations of two objects

    _obj1_backup = None
    _obj2_backup = None

    def __post_init__(self):
        raise NotImplementedError(f"{self.__class__.__name__} untested")

    def apply(self, state: state_def.State):
        target1, target2 = self.names

        o1 = state[target1].obj
        o2 = state[target2].obj

        self._obj1_backup = pose_backup(o1, dof=False)
        self._obj2_backup = pose_backup(o2, dof=False)

        o1.loc, o2.loc = o2.loc, o1.loc
        o1.rotation_axis_angle, o2.rotation_axis_angle = (
            o2.rotation_axis_angle,
            o1.rotation_axis_angle,
        )
        o1.relation_assignments, o2.relation_assignments = (
            o2.relation_assignments,
            o1.relation_assignments,
        )

    def revert(self, state: state_def.State):
        target1, target2 = self.names
        restore_pose_backup(state, target1, self._obj1_backup)
        restore_pose_backup(state, target2, self._obj2_backup)

        o1 = state[target1].obj
        o2 = state[target2].obj

        o1.relation_assignments, o2.relation_assignments = (
            o2.relation_assignments,
            o1.relation_assignments,
        )
