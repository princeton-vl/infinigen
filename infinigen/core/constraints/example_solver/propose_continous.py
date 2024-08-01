# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
import typing

import numpy as np

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator.domain_contains import domain_contains

from . import moves, state_def

logger = logging.getLogger(__name__)

TRANS_MULT = 8
TRANS_MIN = 0.01
ROT_MULT = np.pi
ROT_MIN = 0  # 2 * np.pi / 200

ANGLE_STEP_SIZE = (2 * np.pi) / 8


def get_pose_candidates(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    require_rot_free: bool = False,
):
    return [
        k
        for k, o in state.objs.items()
        if o.active
        and domain_contains(filter_domain, state, o)
        and not (require_rot_free and o.dof_rotation_axis is None)
    ]


def propose_translate(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
) -> typing.Iterator[moves.TranslateMove]:
    candidates = get_pose_candidates(consgraph, state, filter_domain)
    candidates = [
        c for c in candidates if state.objs[c].dof_matrix_translation is not None
    ]
    if not len(candidates):
        return

    while True:
        obj_state_name = np.random.choice(candidates)
        obj_state = state.objs[obj_state_name]

        var = max(TRANS_MIN, TRANS_MULT * temperature)
        random_vector = np.random.normal(0, var, size=3)
        projected_vector = obj_state.dof_matrix_translation @ random_vector

        yield moves.TranslateMove(
            names=[obj_state_name],
            translation=projected_vector,
        )


def propose_rotate(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
) -> typing.Iterator[moves.RotateMove]:
    candidates = get_pose_candidates(consgraph, state, filter_domain)
    candidates = [
        c
        for c in candidates
        if (
            t.Semantics.NoRotation not in state.objs[c].tags
            and state.objs[c].dof_rotation_axis is not None
            and state.objs[c].dof_rotation_axis.dot(np.array((0, 0, 1))) > 0.95
        )
    ]
    if not len(candidates):
        return

    while True:
        obj_state_name = np.random.choice(candidates)
        obj_state = state.objs[obj_state_name]

        var = max(ROT_MIN, ROT_MULT * temperature)
        random_angle = np.random.normal(0, var)

        ang = random_angle / ANGLE_STEP_SIZE
        ang = np.ceil(ang) if ang > 0 else np.floor(ang)
        random_angle = ang * ANGLE_STEP_SIZE

        axis = obj_state.dof_rotation_axis
        yield moves.RotateMove(names=[obj_state_name], axis=axis, angle=random_angle)


def propose_reinit_pose(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
) -> typing.Iterator[moves.ReinitPoseMove]:
    candidates = get_pose_candidates(consgraph, state, filter_domain)
    candidates = [
        c for c in candidates if state.objs[c].dof_matrix_translation is not None
    ]

    if len(candidates) == 0:
        return

    while True:
        obj_state_name = np.random.choice(candidates)
        state.objs[obj_state_name]

        yield moves.ReinitPoseMove(
            names=[obj_state_name],
        )


def propose_scale(consgraph, state, temperature):
    raise NotImplementedError
    obj_state = np.random.choice(state.objs)
    random_scale = np.random.normal(0, temperature, size=3)
    return moves.ScaleMove(name=obj_state.name, scale=random_scale)
