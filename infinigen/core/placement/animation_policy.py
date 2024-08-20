# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: Primary author
# - Zeyu Ma: Animation with path finding


import logging
from copy import copy, deepcopy

import bpy
import gin
import mathutils
import numpy as np
from mathutils import Euler, Vector
from mathutils.bvhtree import BVHTree
from numpy.random import normal as N
from numpy.random import uniform as U
from tqdm import tqdm

from infinigen.assets.utils.geometry.curve import Curve
from infinigen.core.placement.path_finding import path_finding
from infinigen.core.util import blender as butil
from infinigen.core.util.math import lerp
from infinigen.core.util.random import random_general

logger = logging.getLogger(__name__)


class PolicyError(ValueError):
    pass


def get_altitude(loc, scene_bvh, dir=Vector((0.0, 0.0, -1.0))):
    *_, straight_down_dist = scene_bvh.ray_cast(loc, dir)
    return straight_down_dist


@gin.configurable
def walk_same_altitude(
    start_loc,
    sampler,
    bvh,
    filter_func=None,
    fall_ratio=1.5,
    retries=30,
    step_up_height=2,
    ignore_missed_rays=False,
    z_move_up=1,
):
    """
    fall_ratio: what is the slope at which the camera is willing to go down / glide
    """

    # retry until we find something that doesnt walk off the map
    for retry in range(retries):
        pos = start_loc + Vector(sampler())

        pos.z += z_move_up  # move it up a ways, so that it can raycast back down onto something
        curr_alt = get_altitude(start_loc, bvh)
        new_alt = get_altitude(pos, bvh)

        if ignore_missed_rays:
            if curr_alt is None:
                curr_alt = start_loc.z
            if new_alt is None:
                new_alt = pos.z

        if curr_alt is None or new_alt is None:
            if curr_alt is None:
                raise PolicyError()
            logger.debug(
                f"walk_same_altitude failed {retry=} with {curr_alt=}, {new_alt=}"
            )
            continue

        fall_dist = new_alt - curr_alt
        max_fall_dist = step_up_height + fall_ratio * (pos - start_loc).length
        fall_dist = np.clip(fall_dist, -100, max_fall_dist)

        pos.z = pos.z - fall_dist

        if filter_func is not None and not filter_func(pos):
            continue

        break
    else:
        raise PolicyError()

    return pos


@gin.configurable
class AnimPolicyBrownian:
    def __init__(self, speed=3, pos_var=15.0):
        self.speed = speed
        self.pos_var = pos_var

    def __call__(self, obj, frame_curr, bvh, retry_pct):
        speed = random_general(self.speed)

        def sampler():
            return N(0, [self.pos_var, self.pos_var, 0.5])

        pos = walk_same_altitude(obj.location, sampler, bvh)
        time = np.linalg.norm(pos - obj.location) / speed

        rot = np.array(obj.rotation_euler) + np.deg2rad(N(0, [5, 0, 5], 3))

        return Vector(pos), Vector(rot), time, "BEZIER"


@gin.configurable
class AnimPolicyPan:
    def __init__(self, speed=3, dist=("uniform", 5, 20), rot_var=[10, 0, 20]):
        self.speed = speed
        self.dist = dist
        self.rot_var = rot_var

    def __call__(self, obj, frame_curr, bvh, retry_pct):
        speed = random_general(self.speed)

        def sampler():
            theta = U(0, 2 * np.pi)
            zoff = np.sin(np.deg2rad(N(-30, 30)))
            off = random_general(self.dist) * np.array(
                [np.sin(theta), np.cos(theta), zoff]
            )
            off = off * lerp(1, 0.2, 1 - retry_pct)
            return off

        pos = walk_same_altitude(obj.location, sampler, bvh=bvh)
        time = np.linalg.norm(pos - obj.location) / speed

        rot = np.array(obj.rotation_euler) + np.deg2rad(N(0, self.rot_var, 3))
        return Vector(pos), Vector(rot), time, "LINEAR"


@gin.configurable
class AnimPolicyRandomForwardWalk:
    def __init__(
        self,
        forward_vec,
        speed=2,
        yaw_dist=("uniform", -30, 30),
        altitude_var=0,
        step_range=(1, 10),
        rot_vars=[5, 0, 5],
    ):
        self.speed = speed
        self.yaw_dist = yaw_dist
        self.step_range = step_range
        self.altitude_var = altitude_var
        self.rot_vars = rot_vars
        self.forward_vec = forward_vec

    def __call__(self, obj, frame_curr, bvh, retry_pct):
        orig_rot = np.array(obj.rotation_euler)

        def sampler():
            obj.rotation_euler = tuple(orig_rot)
            obj.rotation_euler[2] += np.deg2rad(random_general(self.yaw_dist))
            step = U(*self.step_range)
            off = obj.rotation_euler.to_matrix() @ (step * Vector(self.forward_vec))
            off.z = 0
            return off

        pos = walk_same_altitude(obj.location, sampler, bvh)
        pos.z += N(0, self.altitude_var)

        time = np.linalg.norm(pos - obj.location) / self.speed
        rot = np.array(obj.rotation_euler) + np.deg2rad(N(0, self.rot_vars, 3))

        return Vector(pos), Vector(rot), time, "BEZIER"


@gin.configurable
class AnimPolicyRandomWalkLookaround:
    def __init__(
        self,
        speed=("uniform", 1, 2.5),
        step_speed_mult=("uniform", 0.5, 2),
        yaw_sampler=("uniform", -20, 20),
        step_range=("clip_gaussian", 3, 5, 0.5, 10),
        rot_vars=(5, 0, 5),
        motion_dir_zoff=("uniform", 0, 360),
        force_single_keyframe=False,
    ):
        self.speed = random_general(speed)

        self.step_speed_mult = step_speed_mult
        self.yaw_sampler = yaw_sampler
        self.step_range = step_range
        self.rot_vars = rot_vars

        self.motion_dir_euler = None
        self.motion_dir_zoff = motion_dir_zoff

        self.force_single_keyframe = force_single_keyframe

    def __call__(self, obj, frame_curr, bvh, retry_pct):
        if self.motion_dir_euler is None:
            self.motion_dir_euler = copy(obj.rotation_euler)

            self.motion_dir_euler[2] += np.deg2rad(random_general(self.motion_dir_zoff))

        orig_motion_dir_euler = copy(self.motion_dir_euler)

        def sampler():
            self.motion_dir_euler = copy(orig_motion_dir_euler)
            self.motion_dir_euler[2] += np.deg2rad(random_general(self.yaw_sampler))
            step = random_general(self.step_range)
            off = Euler(self.motion_dir_euler, "XYZ").to_matrix() @ Vector(
                (0, 0, -step)
            )
            off.z = 0
            return off

        pos = walk_same_altitude(obj.location, sampler, bvh)

        step_speed = self.speed * random_general(self.step_speed_mult)
        time = np.linalg.norm(pos - obj.location) / step_speed
        rot = np.array(obj.rotation_euler) + np.deg2rad(N(0, self.rot_vars, 3))

        if self.force_single_keyframe:
            time = bpy.context.scene.frame_end - frame_curr

        return Vector(pos), Vector(rot), time, "BEZIER"


@gin.configurable
class AnimPolicyFollowObject:
    def __init__(
        self,
        target_obj,
        pois,
        bvh,
        zrot_vel_var=20,
        follow_zrot=0,
        follow_rad_mult=("uniform", 1, 6),
        alt_mult=("uniform", 0.25, 1),
    ):
        self.pois = pois
        self.target_obj = target_obj
        self.follow_zrot = follow_zrot
        self.zrot_vel = np.deg2rad(N(0, zrot_vel_var))
        self.rad_vel = N(0, 0.03)
        self.bvh = bvh

        self.follow_rad_mult = follow_rad_mult
        self.alt_mult = alt_mult

        self.follow_obj = None

        self.reset()

    def reset(self):
        """
        Called at __init__ and whenever the animation aborts and retries
        """

        self.follow_obj = np.random.choice(self.pois)
        if self.follow_obj.type == "MESH":
            self.follow_size = max(self.follow_obj.dimensions)
        else:
            self.follow_size = 2
            logger.warning(
                f"{self.follow_obj.name} had {self.follow_obj.type=}, using {self.follow_size=} instead of .dimensions"
            )

        follow_loc = self.follow_obj.matrix_world.translation
        off = follow_loc - self.target_obj.location
        s = self.follow_size * random_general(self.follow_rad_mult)
        self.target_obj.location = follow_loc + off.normalized() * s
        self.target_obj.location.z = follow_loc.z + self.follow_size * random_general(
            self.alt_mult
        )

        alt = get_altitude(self.target_obj.location, self.bvh)
        if alt is None:
            logger.warning(f"In AnimPolicyFollowObject.reset(), got {alt=}")
        if alt is not None and alt < 2:
            self.target_obj.location *= self.target_obj.location.z / 2

        for c in self.target_obj.constraints:
            self.target_obj.constraints.remove(c)

        butil.constrain_object(self.target_obj, "TRACK_TO", target=self.follow_obj)

    def __call__(self, obj, frame_curr, bvh, retry_pct):
        try:
            ts = []
            for fc in self.follow_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    ts.append(int(kp.co[0]))
            frame_next = min(t for t in ts if t > frame_curr)
        except (
            ValueError,
            AttributeError,
        ):  # no next frame, or no animation_data.action
            frame_next = frame_curr + bpy.context.scene.render.fps

        time = (frame_next - frame_curr) / bpy.context.scene.render.fps

        bpy.context.scene.frame_set(frame_curr)

        prev_off = obj.location - self.follow_obj.matrix_world.translation
        prev_zrot = copy(self.follow_obj.rotation_euler.z)
        prev_dist = prev_off.length / self.follow_size

        bpy.context.scene.frame_set(frame_next)

        zrot_diff = obj.rotation_euler.z - prev_zrot
        zrot = self.follow_zrot * zrot_diff + time * self.zrot_vel

        new_dist = np.clip(prev_dist + self.rad_vel, 0.7, 5)
        new_off = prev_off.normalized() * self.follow_size * new_dist
        new_off = mathutils.Matrix.Rotation(np.deg2rad(zrot), 4, "Z") @ new_off
        pos = self.follow_obj.matrix_world.translation + new_off

        return Vector(pos), None, time, "BEZIER"


def validate_keyframe_range(
    obj,
    start_frame,
    end_frame,
    bvhtree,
    validate_pose_func=None,
    stride=5,  # runs faster but imperfect precision
    check_straight_line=True,  # rules out proposals faster, but has imperfect precision
):
    last_pos = deepcopy(obj.location)

    def freespace_ray_check(a, b):
        location, *_ = bvhtree.ray_cast(a, b - a, (a - b).length)
        return location is None

    if check_straight_line:
        bpy.context.scene.frame_set(end_frame)
        if not freespace_ray_check(last_pos, obj.location):
            logger.debug("straight line check failed")
            return False

    for frame_idx in range(start_frame, end_frame + 1, stride):
        bpy.context.scene.frame_set(frame_idx)

        if not freespace_ray_check(last_pos, obj.location):
            logger.debug(f"{frame_idx=} freespace_ray_check failed")
            return False

        if validate_pose_func is not None and not validate_pose_func(obj):
            # technically we should validate against all cameras, but this would be expensive
            logger.debug(f"{frame_idx} validate_pose_func failed")
            return False

        last_pos = deepcopy(obj.location)

    return True


def try_animate_trajectory(
    obj: bpy.types.Object,
    bvh: BVHTree,
    policy_func,
    keyframe,
    duration_frames,
    validate_pose_func=None,
    max_step_tries=50,
    verbose=True,
):
    frame_curr = bpy.context.scene.frame_start
    pbar = tqdm(total=duration_frames) if verbose else None
    while frame_curr < bpy.context.scene.frame_start + duration_frames:
        orig_loc = copy(obj.location)
        orig_rot = copy(obj.rotation_euler)
        for retry in range(max_step_tries):
            obj.location, obj.rotation_euler = orig_loc, orig_rot
            bpy.context.view_layer.update()
            try:
                loc, rot, duration, interp = policy_func(
                    obj,
                    frame_curr=frame_curr,
                    retry_pct=retry / max_step_tries,
                    bvh=bvh,
                )
            except PolicyError as e:
                logger.debug(f"PolicyError on {retry=} {e=}")
                continue

            step_frames = int(duration * bpy.context.scene.render.fps) + 1
            step_end_frame = frame_curr + step_frames

            keyframe(obj, loc, rot, step_end_frame, interp="BEZIER")

            if not validate_keyframe_range(
                obj, frame_curr, step_end_frame, bvh, validate_pose_func
            ):
                logger.debug(
                    f"validate_keyframe_range failed on moving {obj.location} to {loc}"
                )
                # clear out the candidate keyframes we just inserted, they were no good
                for fc in obj.animation_data.action.fcurves:
                    if fc.data_path == "":
                        continue
                    obj.keyframe_delete(data_path=fc.data_path, frame=step_end_frame)
                continue

            if verbose:
                pbar.update(
                    min(step_frames, duration_frames - frame_curr)
                )  # dont overshoot the pbar, it makes the formatting not nice

            break  # we found a good pose

        else:  # for-else block triggers when for loop terminates w/o a break statement
            return False

        frame_curr = step_end_frame
        bpy.context.scene.frame_current = frame_curr

    return True


@gin.configurable
def try_animate_with_pathfinding(
    obj,
    bvh,
    policy_func,
    keyframe,
    duration_frames,
    validate_pose_func,
    max_step_tries,
    verbose,
    bounding_box,
    turning_limit_degree=10,
):
    frame_curr = bpy.context.scene.frame_start
    pbar = tqdm(total=duration_frames) if verbose else None
    while frame_curr < bpy.context.scene.frame_start + duration_frames:
        orig_loc = copy(obj.location)
        orig_rot = copy(obj.rotation_euler)
        for retry in range(max_step_tries):
            obj.location, obj.rotation_euler = orig_loc, orig_rot
            bpy.context.view_layer.update()
            try:
                loc, rot, duration, interp = policy_func(
                    obj,
                    frame_curr=frame_curr,
                    retry_pct=retry / max_step_tries,
                    bvh=bvh,
                )
            except PolicyError as e:
                logger.debug(f"PolicyError on {retry=} {e=}")
                continue

            bpy.context.scene.frame_set(frame_curr)
            last_pose = (deepcopy(obj.location), deepcopy(obj.rotation_euler))
            keyframe(obj, loc, rot, frame_curr + 1, interp="BEZIER")
            bpy.context.scene.frame_set(frame_curr + 1)

            valid_target = True
            if validate_pose_func is not None and not validate_pose_func(obj):
                valid_target = False

            # if valid_target:
            #     bpy.ops.wm.save_mainfile(filepath="/u/zeyum/p/debug.blend")
            #     assert(0)

            for fc in obj.animation_data.action.fcurves:
                if fc.data_path == "":
                    continue
                obj.keyframe_delete(data_path=fc.data_path, frame=frame_curr + 1)

            if not valid_target:
                logger.debug(
                    "validate_pose_func at target pose failed, aborting path finding"
                )
                continue

            bounded = True
            current_pose = (loc, rot)
            for i in range(3):
                if (
                    current_pose[0][i] < bounding_box[0][i]
                    or current_pose[0][i] >= bounding_box[1][i]
                ):
                    bounded = False
            if not bounded:
                logger.debug("target pose out of bound, aborting path finding")
                continue

            poses = path_finding(bvh, bounding_box, last_pose, current_pose)

            if poses is None:
                logger.debug("path not found, aborting")
                continue

            base_length = (last_pose[0] - current_pose[0]).length
            scaling = poses[-1][0] / base_length

            step_frames = int(duration * bpy.context.scene.render.fps * scaling) + 1
            step_end_frame = frame_curr + step_frames

            turning_too_fast = False
            for i in range(len(poses) - 1):
                rotation_euler0 = poses[i][2]
                rotation_euler1 = poses[i + 1][2]
                if (
                    abs(rotation_euler0.z - rotation_euler1.z)
                    > turning_limit_degree
                    / 180
                    * np.pi
                    * step_frames
                    * (poses[i + 1][0] - poses[i][0])
                    / poses[-1][0]
                ):
                    turning_too_fast = True
                    break
            if turning_too_fast:
                logger.debug("path turns too fast, aborting")
                continue

            for l, location, rotation_euler in poses:
                t = l / poses[-1][0]
                keyframe(
                    obj,
                    location,
                    rotation_euler,
                    round(frame_curr + (step_end_frame - frame_curr) * t),
                    interp="LINEAR",
                )

            if verbose:
                pbar.update(
                    min(step_frames, duration_frames - frame_curr)
                )  # dont overshoot the pbar, it makes the formatting not nice

            break  # we found a good pose

        else:  # for-else block triggers when for loop terminates w/o a break statement
            return False

        frame_curr = step_end_frame
        bpy.context.scene.frame_current = frame_curr

    return True


def keyframe(obj, loc, rot, t, interp="BEZIER"):
    if obj.animation_data is not None and obj.animation_data.action is not None:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                if kp.co > t:
                    raise ValueError(
                        f"Unexpected out-of-order keyframing {kp.co=}, {t=}"
                    )

    if loc is not None:
        obj.location = loc
        (obj.keyframe_insert(data_path="location", frame=t),)

    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert(data_path="rotation_euler", frame=t)

    for fc in obj.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            if k.co[0] == t:
                k.interpolation = interp


@gin.configurable
def animate_trajectory(
    obj,
    bvh,
    policy_func,
    validate_pose_func=None,
    max_step_tries=25,
    max_full_retries=10,
    retry_rotation=False,
    verbose=True,
    fatal=False,
    reverse_time=False,
    bounding_box=None,
    path_finding_enabled=False,
):
    duration_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start
    duration_sec = duration_frames / bpy.context.scene.render.fps
    if duration_sec < 1e-3:
        return

    obj_orig_loc = copy(obj.location)
    obj_orig_rot = copy(obj.rotation_euler)

    for attempt in range(max_full_retries):
        obj.animation_data_clear()
        obj.location = obj_orig_loc
        obj.rotation_euler = obj_orig_rot
        if attempt > 0 and retry_rotation:
            obj.rotation_euler.z = U(0, 2 * np.pi)

        if hasattr(policy_func, "reset"):
            policy_func.reset()

        keyframe(obj, obj.location, obj.rotation_euler, 0, interp="LINEAR")
        try_animate_trajectory_func = (
            try_animate_trajectory
            if not path_finding_enabled
            else try_animate_with_pathfinding
        )
        args = [
            obj,
            bvh,
            policy_func,
            keyframe,
            duration_frames,
            validate_pose_func,
            max_step_tries,
            verbose,
        ]
        if path_finding_enabled:
            args.append(bounding_box)
        if try_animate_trajectory_func(*args):
            if reverse_time:
                kf_locs = []
                kf_rots = []
                kf_ts = []
                for j in range(
                    len(obj.animation_data.action.fcurves[0].keyframe_points)
                ):
                    kf_ts.append(
                        obj.animation_data.action.fcurves[0].keyframe_points[j].co.x
                    )
                    kf_locs.append(
                        (
                            obj.animation_data.action.fcurves[0]
                            .keyframe_points[j]
                            .co.y,
                            obj.animation_data.action.fcurves[1]
                            .keyframe_points[j]
                            .co.y,
                            obj.animation_data.action.fcurves[2]
                            .keyframe_points[j]
                            .co.y,
                        )
                    )
                    kf_rots.append(
                        (
                            obj.animation_data.action.fcurves[3]
                            .keyframe_points[j]
                            .co.y,
                            obj.animation_data.action.fcurves[4]
                            .keyframe_points[j]
                            .co.y,
                            obj.animation_data.action.fcurves[5]
                            .keyframe_points[j]
                            .co.y,
                        )
                    )
                obj.animation_data_clear()
                for i, t in enumerate(kf_ts):
                    keyframe(
                        obj,
                        kf_locs[i],
                        kf_rots[i],
                        bpy.context.scene.frame_end + bpy.context.scene.frame_start - t,
                        interp="LINEAR",
                    )
                # bpy.context.scene.frame_set(bpy.context.scene.frame_end)
                # obj.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_end)
                # obj.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_end)
                # assert(0)
            break
        logger.info(f"Failed {attempt=} out of {max_full_retries=} for {obj.name=}")
    else:
        err = f"Animation for {obj.name=} failed with {max_full_retries=} and {max_step_tries=}, quitting"
        if fatal:
            raise ValueError(err)
        else:
            logger.warning(err)
            return


def policy_create_bezier_path(
    start_pose_obj, bvh, policy_func, to_mesh=False, eval_offset=(0, 0, 0), **kwargs
):
    eval_offset = Vector(eval_offset)

    # animate a dummy using the policy
    temp = butil.spawn_empty("policy_create_bezier_path.temp")
    temp.location = start_pose_obj.location + eval_offset
    temp.rotation_euler = start_pose_obj.rotation_euler
    animate_trajectory(temp, bvh, policy_func, **kwargs)

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
            bpy.context.scene.frame_set(f)
            positions.append(deepcopy(temp.location - eval_offset))

    logger.debug(f"Created policy path with {len(positions)} keypoints")

    res = Curve(points=positions).to_curve_obj(name="policy_path", to_mesh=to_mesh)
    butil.delete(temp)
    return res
