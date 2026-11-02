# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

from functools import partial

import bpy
import numpy as np

import infinigen
from infinigen.core.placement import animation_policy, camera
from infinigen.core.util import blender as butil
from infinigen.perf.core.placement import animation_policy as opt_policy


def _animated_empty():
    butil.clear_scene()
    # other scene contents, which frame_set evaluates and set_frame does not
    for i in range(3):
        bpy.ops.mesh.primitive_uv_sphere_add(location=(i, 0, 0))
    obj = butil.spawn_empty("animated")
    for frame, loc, rot in ((1, (0, 0, 0), (0, 0, 0)), (24, (3, -2, 1), (0.3, 0, 1.2))):
        obj.location = loc
        obj.rotation_euler = rot
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_euler", frame=frame)
    return obj


def _poses(obj, set_frame):
    out = []
    for frame in list(range(1, 25)) + [7, 3, 19]:
        set_frame(frame)
        out.append(np.array(obj.matrix_world, dtype=np.float32))
    return np.stack(out)


def test_set_frame_in_a_scope_matches_frame_set():
    scene = bpy.context.scene
    obj = _animated_empty()
    expected = _poses(obj, scene.frame_set)
    try:
        infinigen.use_optimized(["placement"])
        with opt_policy.AnimationScope(obj, None, None, tag="test") as scope:
            assert scope.fast, scope.reason
            got = _poses(obj, opt_policy.set_frame)
    finally:
        infinigen.use_optimized(False)
    assert np.array_equal(expected, got)


def test_only_self_reading_policies_and_validators_take_the_fast_path():
    assert not opt_policy.reads_depsgraph(None)
    assert not opt_policy.reads_depsgraph(
        animation_policy.AnimPolicyRandomWalkLookaround()
    )
    assert not opt_policy.reads_depsgraph(
        partial(camera.keep_cam_pose_proposal, terrain=None)
    )
    assert opt_policy.reads_depsgraph(lambda obj: True)
