# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick: primary author
# - Karhan Kayan: sync with trimesh fix

import copy
from dataclasses import dataclass

from infinigen.core.constraints.constraint_language.util import sync_trimesh
from infinigen.core.constraints.example_solver.geometry import dof
from infinigen.core.constraints.example_solver.state_def import ObjectState, State

from . import moves


def pose_backup(os: ObjectState, dof=True):
    bak = dict(
        loc=tuple(os.obj.location),
        rot=tuple(os.obj.rotation_euler),
    )

    if dof:
        bak["dof_trans"] = copy.copy(os.dof_matrix_translation)
        bak["dof_rot"] = copy.copy(os.dof_rotation_axis)

    return bak


def restore_pose_backup(state, name, bak):
    os = state.objs[name]
    os.obj.location = bak["loc"]
    os.obj.rotation_euler = bak["rot"]

    if "dof_trans" in bak:
        os.dof_matrix_translation = bak["dof_trans"]
    if "dof_rot" in bak:
        os.dof_rotation_axis = bak["dof_rot"]

    sync_trimesh(state.trimesh_scene, state.objs[name].obj.name)


@dataclass
class RelationPlaneChange(moves.Move):
    relation_idx: int
    plane_idx: int

    _backup_idx = None
    _backup_poseinfo = None

    def apply(self, state: State):
        (target_name,) = self.names

        os = state.objs[target_name]
        rels = os.relations[self.relation_idx]

        self._backup_idx = rels.parent_plane_idx
        self._backup_poseinfo = pose_backup(os)

        rels.parent_plane_idx = self.plane_idx

        success = dof.try_apply_relation_constraints(state, target_name)
        return success

    def revert(self, state: State):
        (target_name,) = self.names

        os = state.objs[target_name]
        os.relations[self.relation_idx].parent_plane_idx = self._backup_idx
        restore_pose_backup(state, target_name, self._backup_poseinfo)


@dataclass
class RelationTargetChange(moves.Move):
    # reassign obj to new parent
    name: str
    relation_idx: int
    new_target: str

    _backup_target = None
    _backup_poseinfo = None

    def apply(self, state: State):
        os = state.objs[self.name]
        rels = os.relations[self.relation_idx]

        self._backup_target = rels.target_name
        self._backup_poseinfo = pose_backup(os)
        rels.target_name = self.new_target

        return dof.try_apply_relation_constraints(state, self._new_name)

    def revert(self, state: State):
        os = state.objs[self.name]
        rels = os.relations[self.relation_idx]

        rels.target_name = self._backup_target
        restore_pose_backup(state, self.name, self._backup_poseinfo)
