# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: primary author of the original Infinigen animation policy (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/animation_policy.py)
# - Zeyu Ma: path-finding animation in the original Infinigen animation policy (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/animation_policy.py)
# - David Yan, Jia Deng: optimized implementation

"""Trajectory search that evaluates only the animated object when setting a frame.

Inside a fast `AnimationScope`, `set_frame` updates just the scratch view layer holding
the object; the main one catches up at its next evaluation. Keyframes are the default's.
"""

import logging
import time
from copy import deepcopy
from functools import partial

import bpy
from mathutils import Vector

from infinigen.assets.utils.geometry.curve import Curve
from infinigen.core.placement import animation_policy as _default
from infinigen.core.util import blender as butil
from infinigen.perf import options
from infinigen.perf.core.util import isolation as _isolation

# these run in place of the default functions, so they log as the default module
logger = logging.getLogger(_default.__name__)


def _default_body(gated):
    """The default implementation behind a gate, under any wrappers (gin's)."""
    f = gated
    for _ in range(8):
        body = getattr(f, "default", None)
        if body is not None:
            return body
        f = getattr(f, "__wrapped__", None)
        if f is None:
            break
    raise TypeError(f"{gated!r} is not a gated definition")


def _chain(fn):
    """`fn` and what it wraps, through `functools.partial` and `__wrapped__`."""
    out = []
    for _ in range(16):
        if fn is None:
            break
        out.append(fn)
        if isinstance(fn, partial):
            fn = fn.func
        else:
            fn = getattr(fn, "__wrapped__", None)
    return out


def _partial_keyword(fn, name):
    """Keyword `name` bound by `fn` (a possibly nested `functools.partial`), or None."""
    while isinstance(fn, partial):
        if name in fn.keywords:
            return fn.keywords[name]
        fn = fn.func
    return None


def _self_reading():
    """Policies and pose validators that read nothing but the animated object's own pose
    and structures independent of the dependency graph."""
    from infinigen.core.placement import camera as _camera

    classes = (
        _default.AnimPolicyBrownian,
        _default.AnimPolicyPan,
        _default.AnimPolicyRandomForwardWalk,
        _default.AnimPolicyRandomWalkLookaround,
        _camera.AnimPolicyGoToProposals,
    )
    functions = (_camera.keep_cam_pose_proposal, _camera.anim_valid_camrig_pose_func)
    return classes, functions


def reads_depsgraph(fn):
    """Whether policy or pose validator `fn` may read evaluated data of objects other
    than the animated one; True unless it is None or known to read only that object."""
    if fn is None:
        return False
    classes, functions = _self_reading()
    for f in _chain(fn):
        if isinstance(f, classes) or any(f is g for g in functions):
            return False
    return True


def _prefetch(validator):
    """Function (obj, frames) evaluating `validator`'s terrain SDF for every frame of a
    step at once, or None when the validator has none."""
    from infinigen.core.placement import camera as _camera
    from infinigen.perf.core.placement import camera as _perf_camera

    for f in _chain(validator):
        if f is _camera.keep_cam_pose_proposal:
            terrain = _partial_keyword(validator, "terrain")
            return lambda obj, frames: _perf_camera.prefetch_terrain_sdf(
                terrain, [obj], frames
            )
        if f is _camera.anim_valid_camrig_pose_func:
            pre = _partial_keyword(validator, "scene_preprocessed")
            if pre is None:
                return None
            return lambda rig, frames: _perf_camera.prefetch_terrain_sdf(
                pre.get("terrain"), list(rig.children), frames
            )
    return None


def _self_contained(obj):
    """True if evaluating `obj` and its descendants at a frame needs no other object: no
    parent, constraints or drivers, and none hidden in the viewport."""
    for o in (obj, *obj.children_recursive):
        if o.hide_viewport or len(o.constraints):
            return False
        ad = o.animation_data
        if ad is not None and len(ad.drivers):
            return False
    return obj.parent is None


class AnimationScope:
    """Context evaluating only the animated object while its trajectory is built, in the
    view layer of an enclosing scope or isolated asset build if there is one."""

    active = []

    def __init__(self, obj, policy_func=None, validate_pose_func=None, tag=None):
        self.obj = obj
        self.policy_func = policy_func
        self.validate_pose_func = validate_pose_func
        self.tag = tag
        self.fast = False
        self.iso = None
        self.reason = None

    def _eligible(self):
        """True if the fast path applies; otherwise `self.reason` says why."""
        if not options.isolation_enabled():
            self.reason = "scene isolation is turned off"
        elif reads_depsgraph(self.policy_func):
            self.reason = f"policy {self.policy_func!r} reads the depsgraph"
        elif reads_depsgraph(self.validate_pose_func):
            self.reason = f"validator {self.validate_pose_func!r} reads the depsgraph"
        elif not _self_contained(self.obj):
            self.reason = f"{self.obj.name!r} has a parent, constraints or drivers"
        else:
            return True
        return False

    @staticmethod
    def _evaluated_in(vl, objs):
        for o in objs:
            if o.name not in vl.objects or o.hide_get(view_layer=vl):
                return False
        return True

    def __enter__(self):
        self.t_enter = time.perf_counter()
        self.n_frames = 0
        self.t_frames = 0.0
        if AnimationScope.active:
            # nested (policy_create_bezier_path around animate_trajectory): reuse
            # the outer scope's view layer if this scope qualifies too
            self.fast = (
                AnimationScope.active[-1].fast
                and self._eligible()
                and self._evaluated_in(
                    bpy.context.view_layer, [self.obj, *self.obj.children_recursive]
                )
            )
        elif self._eligible():
            objs = [self.obj, *self.obj.children_recursive]
            if _isolation.IsolatedScene.active:
                # inside an isolated asset build: use its scratch view layer,
                # exposing objects not in it (policy_create_bezier_path's
                # temporary empty)
                for o in objs:
                    _isolation.IsolatedScene.ensure_selectable(o)
                self.fast = self._evaluated_in(bpy.context.view_layer, objs)
                if not self.fast:
                    self.reason = (
                        "not evaluated in the isolated build's view layer "
                        f"{bpy.context.view_layer.name!r}"
                    )
            else:
                self.iso = _isolation.IsolatedScene(link=objs, tag=self.tag)
                self.iso.__enter__()
                self.fast = self._evaluated_in(bpy.context.view_layer, objs)
                if not self.fast:
                    self.reason = "not evaluated in the scratch view layer"
        if not self.fast and not AnimationScope.active:
            logger.info(f"{self.tag} {self.obj.name!r}: frame_set path ({self.reason})")
        AnimationScope.active.append(self)
        return self

    def __exit__(self, *exc):
        AnimationScope.active.remove(self)
        if self.iso is not None:
            self.iso.__exit__(*exc)
            self.iso = None
        if AnimationScope.active:
            outer = AnimationScope.active[-1]
            outer.n_frames += self.n_frames
            outer.t_frames += self.t_frames
        else:
            logger.debug(
                f"{self.tag} {self.obj.name!r}: {self.n_frames} frame evaluations in "
                f"{self.t_frames:.2f} s (fast={self.fast}), scope total "
                f"{time.perf_counter() - self.t_enter:.2f} s"
            )
        return False


def set_frame(frame):
    """`frame_set(frame)` for just the animated object in a fast `AnimationScope`."""
    scene = bpy.context.scene
    scope = AnimationScope.active[-1] if AnimationScope.active else None
    t0 = time.perf_counter()
    if scope is not None and scope.fast:
        scene.frame_current = frame
        # frame_set re-evaluates every animated datablock even when the frame
        # does not change; tag the animation so the update does too
        for o in (scope.obj, *scope.obj.children_recursive):
            o.update_tag(refresh={"TIME"})
        bpy.context.view_layer.update()
    else:
        scene.frame_set(frame)
    if scope is not None:
        scope.n_frames += 1
        scope.t_frames += time.perf_counter() - t0


def validate_keyframe_range(
    obj,
    start_frame,
    end_frame,
    bvhtree,
    validate_pose_func=None,
    stride=1,  # runs faster but imperfect precision
    check_straight_line=True,  # rules out proposals faster, but has imperfect precision
    check_keyframe_straight_line=True,
):
    last_pos = deepcopy(obj.location)

    def freespace_ray_check(a, b):
        location, *_ = bvhtree.ray_cast(a, b - a, (a - b).length)
        return location is None

    scope = AnimationScope.active[-1] if AnimationScope.active else None
    prefetch = _prefetch(validate_pose_func)
    if scope is not None and scope.fast and prefetch is not None:
        # evaluate the validator's terrain SDF for every frame of the step at
        # once; the frames are still set one by one below
        frames = range(start_frame, end_frame + 1, stride)
        if check_straight_line:
            frames = [end_frame, *frames]
        prefetch(obj, frames)

    if check_straight_line:
        set_frame(end_frame)
        if not freespace_ray_check(last_pos, obj.location):
            logger.debug("straight line check failed")
            return False

    for frame_idx in range(start_frame, end_frame + 1, stride):
        set_frame(frame_idx)

        if check_keyframe_straight_line:
            if not freespace_ray_check(last_pos, obj.location):
                logger.debug(f"{frame_idx=} freespace_ray_check failed")
                return False

        if validate_pose_func is not None and not validate_pose_func(obj):
            # technically we should validate against all cameras, but this would be expensive
            logger.debug(f"{frame_idx} validate_pose_func failed")
            return False

        last_pos = deepcopy(obj.location)

    return True


def animate_trajectory(
    obj,
    bvh,
    policy_func,
    validate_pose_func=None,
    max_step_tries=25,
    max_full_retries=40,
    retry_rotation=False,
    verbose=True,
    fatal=False,
    reverse_time=False,
    bounding_box=None,
    path_finding_enabled=False,
):
    with AnimationScope(obj, policy_func, validate_pose_func, tag="animate_trajectory"):
        return _default_body(_default.animate_trajectory)(
            obj,
            bvh,
            policy_func,
            validate_pose_func=validate_pose_func,
            max_step_tries=max_step_tries,
            max_full_retries=max_full_retries,
            retry_rotation=retry_rotation,
            verbose=verbose,
            fatal=fatal,
            reverse_time=reverse_time,
            bounding_box=bounding_box,
            path_finding_enabled=path_finding_enabled,
        )


def policy_create_bezier_path(
    start_pose_obj, bvh, policy_func, to_mesh=False, eval_offset=(0, 0, 0), **kwargs
):
    eval_offset = Vector(eval_offset)

    # animate a dummy using the policy
    temp = butil.spawn_empty("policy_create_bezier_path.temp")
    temp.location = start_pose_obj.location + eval_offset
    temp.rotation_euler = start_pose_obj.rotation_euler
    with AnimationScope(
        temp,
        policy_func,
        kwargs.get("validate_pose_func"),
        tag="policy_create_bezier_path",
    ):
        _default.animate_trajectory(temp, bvh, policy_func, **kwargs)

        # read off the keyframe locations
        positions = []
        if temp.animation_data is not None:
            fc = next(
                fc
                for fc in temp.animation_data.action.fcurves
                if fc.data_path == "location"
            )
            for p in fc.keyframe_points:
                f = int(p.co[0])
                set_frame(f)
                positions.append(deepcopy(temp.location - eval_offset))

    logger.debug(f"Created policy path with {len(positions)} keypoints")

    res = Curve(points=positions).to_curve_obj(name="policy_path", to_mesh=to_mesh)
    butil.delete(temp)
    return res
