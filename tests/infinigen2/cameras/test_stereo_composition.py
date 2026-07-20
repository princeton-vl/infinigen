# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf

import infinigen2.scenes.placement.collision as ccol
from infinigen2.cameras.stereo import stereo_accept_pred
from infinigen2.cameras.util import attach_stereo_right, camera_collision_check


def _left_clear_right_blocked():
    """A left camera at the origin clears the colliders, but a collider box sits
    where the baseline-offset right eye would land."""
    baseline = 3.0
    cube = pf.ops.primitives.mesh_cube(size=2.0)
    pf.ops.object.set_transform(cube, location=(baseline, 0.0, 0.0))
    colliders = ccol.collision_set([cube])

    left = pf.ops.primitives.perspective_camera(focal_length_mm=15)
    pf.ops.object.set_transform(
        left, location=(0.0, 0.0, 0.0), rotation_euler=(np.pi / 2, 0, 0)
    )
    return left, colliders, baseline


def test_stereo_accept_pred_rejects_blocked_right_eye():
    left, colliders, baseline = _left_clear_right_blocked()
    # the monocular check passes - the left camera itself is clear
    assert camera_collision_check(left, colliders) is True
    # the stereo predicate rejects it because the right eye would collide
    assert stereo_accept_pred(baseline)(left, colliders) is False


def test_stereo_accept_pred_accepts_clear_rig():
    left = pf.ops.primitives.perspective_camera(focal_length_mm=15)
    pf.ops.object.set_transform(
        left, location=(0.0, 0.0, 0.0), rotation_euler=(np.pi / 2, 0, 0)
    )
    far_cube = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.object.set_transform(far_cube, location=(50.0, 0.0, 0.0))
    colliders = ccol.collision_set([far_cube])

    assert stereo_accept_pred(0.1)(left, colliders) is True
    # the realized right camera is parented to the left with the baseline offset
    rig = attach_stereo_right(left, baseline=0.1)
    assert len(rig) == 2
    assert np.allclose(rig[1].item().location, (0.1, 0.0, 0.0))
