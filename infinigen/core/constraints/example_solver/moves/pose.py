# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick, Karhan Kayan

import logging
from dataclasses import dataclass

import numpy as np

from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.example_solver.geometry import dof, validity
from infinigen.core.constraints.example_solver.state_def import State

from . import moves
from .reassignment import pose_backup, restore_pose_backup

logger = logging.getLogger(__name__)


@dataclass
class TranslateMove(moves.Move):
    # translate obj by vector

    translation: np.array

    _backup_pose: dict = None

    def __repr__(self):
        norm = np.linalg.norm(self.translation)
        return f"{self.__class__.__name__}({self.names}, {norm:.2e})"

    def apply(self, state: State):
        (target_name,) = self.names

        os = state.objs[target_name]
        self._backup_pose = pose_backup(os, dof=False)

        iu.translate(state.trimesh_scene, os.obj.name, self.translation)

        if not validity.check_post_move_validity(state, target_name):
            return False

        return True

    def revert(self, state: State):
        (target_name,) = self.names
        restore_pose_backup(state, target_name, self._backup_pose)


@dataclass
class RotateMove(moves.Move):
    axis: np.array
    angle: float

    _backup_pose = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.names}, {self.angle:.2e})"

    def apply(self, state: State):
        (target_name,) = self.names

        os = state.objs[target_name]
        self._backup_pose = pose_backup(os, dof=False)

        iu.rotate(state.trimesh_scene, os.obj.name, self.axis, self.angle)

        if not validity.check_post_move_validity(state, target_name):
            return False

        return True

    def revert(self, state: State):
        (target_name,) = self.names
        restore_pose_backup(state, target_name, self._backup_pose)


@dataclass
class ReinitPoseMove(moves.Move):
    _backup_pose: dict = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.names})"

    def apply(self, state: State):
        (target_name,) = self.names
        ostate = state.objs[target_name]
        self._backup_pose = pose_backup(ostate)
        return dof.try_apply_relation_constraints(state, target_name)

    def revert(self, state: State):
        (target_name,) = self.names
        restore_pose_backup(state, target_name, self._backup_pose)


"""
@dataclass
class ScaleMove(Move):
    name: str
    scale: np.array

    def apply(self, state: State):
        blender_obj = self.obj.bpy_obj
        trimesh_obj = state.get_trimesh_object(self.obj.name)
        blender_obj.scale *= Vector(self.scale)
        trimesh_obj.apply_transform(trimesh.transformations.compose_matrix(scale=list(self.scale)))
        self.obj.update()

    def revert(self, state: State):
        blender_obj = self.obj.bpy_obj
        trimesh_obj = state.get_trimesh_object(self.obj.name)
        blender_obj.scale /= Vector(self.scale)
        trimesh_obj.apply_transform(trimesh.transformations.compose_matrix(scale=list(1/self.scale)))
        self.obj.update()
"""
