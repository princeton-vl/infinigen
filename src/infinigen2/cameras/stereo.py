# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import numpy as np
import procfunc as pf

import infinigen2.scenes.placement.collision as ccol

from .util import (
    AcceptPred,
    attach_stereo_right,
    camera_collision_check,
    camera_transform_collision_check,
)

__all__ = [
    "DEFAULT_BASELINE_RANGE",
    "attach_stereo_right",
    "sample_baseline",
    "stereo_accept_pred",
]

DEFAULT_BASELINE_RANGE = (0.03, 0.4)


def sample_baseline(rng: pf.RNG, baseline: float | None = None) -> float:
    if baseline is not None:
        return baseline
    return float(pf.random.uniform(rng, *DEFAULT_BASELINE_RANGE))


def stereo_accept_pred(
    baseline: float,
    accept_pred: AcceptPred | None = None,
) -> AcceptPred:
    """Wrap a left-camera accept predicate so the baseline-offset right camera
    is collision-validated too. Pass to any monocular camera placement (e.g.
    ``random_walk_camera``) so both eyes are checked at every pose, then realize
    the right camera with :func:`attach_stereo_right`."""
    left_pred = accept_pred or camera_collision_check

    def pred(camera_left: pf.CameraObject, colliders: ccol.CollisionSet) -> bool:
        if not left_pred(camera_left, colliders):
            return False
        left_transform = np.array(camera_left.item().matrix_world, dtype=np.float64)
        right_transform = left_transform.copy()
        right_origin = camera_left.item().matrix_world @ pf.Vector((baseline, 0, 0))
        right_transform[:3, 3] = np.array(right_origin, dtype=np.float64)
        return camera_transform_collision_check(right_transform, colliders)

    return pred
