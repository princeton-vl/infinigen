# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: camera pose proposal and random walk animations in the original Infinigen camera trajectories (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/camera_trajectories.py)
# - Dylan Li: refactor into camera_trajectories.py with the RRT option in the original Infinigen camera trajectories (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/camera_trajectories.py)
# - Sumanth Maddirala: base view selection in the original Infinigen camera trajectories (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/camera_trajectories.py)
# - David Yan, Jia Deng: optimized implementation

"""`compute_trajectories` with its whole search in one `AnimationScope`, so each frame
change evaluates the probe camera alone. Trajectories, scores and keyframes are the
default's."""

import logging
from copy import deepcopy
from functools import partial

import bpy
import gin
import numpy as np
from tqdm import tqdm

from infinigen.core.placement import animation_policy as _default_policy
from infinigen.core.placement import camera as cam_util
from infinigen.core.placement import camera_trajectories as _default
from infinigen.core.util import blender as butil
from infinigen.core.util.rrt import AnimPolicyRRT
from infinigen.perf.core.placement import animation_policy as _policy

# these run in place of the default functions, so they log as the default module
logger = logging.getLogger(_default.__name__)


def compute_trajectories(
    cam_rigs,
    base_views,
    scene_preprocessed: dict,
    obj_groups=None,
    camera_selection_answers={},
    camera_selection_ratio=None,
    min_candidates_ratio=5,
    pois=None,
    animation_mode: str = "random_walk",
    validate_pose_func=None,
):
    n_cams = len(cam_rigs)
    cam = cam_util.spawn_camera()
    trajectory_proposals = []

    # num trajectories to fully compute and score
    n_min_candidates = int(min_candidates_ratio * n_cams)

    start = bpy.context.scene.frame_start
    end = bpy.context.scene.frame_end

    if end <= start:
        butil.delete(cam)
        return []

    if validate_pose_func is None:
        anim_valid_pose_func = partial(
            cam_util.keep_cam_pose_proposal,
            placeholders_kd=scene_preprocessed["placeholders_kd"],
            scene_bvh=scene_preprocessed["scene_bvh"],
            terrain=scene_preprocessed["terrain"],
            camera_selection_answers=camera_selection_answers,
            vertexwise_min_dist=scene_preprocessed["vertexwise_min_dist"],
            camera_selection_ratio=camera_selection_ratio,
        )
    else:
        anim_valid_pose_func = validate_pose_func

    # the scope also covers the keyframe read-back and scoring loops below
    with (
        _policy.AnimationScope(
            cam, None, anim_valid_pose_func, tag="compute_trajectories"
        ),
        tqdm(
            total=n_min_candidates, desc="Searching for potential trajectories"
        ) as pbar,
    ):
        i = 0
        for view in base_views:
            i += 1
            # set initial base view, and then propose a trajectory afterwards
            _, proposal, focus_dist = view
            cam.location = proposal.loc
            cam.rotation_euler = proposal.rot

            if not anim_valid_pose_func(cam):
                continue

            match animation_mode:
                case "random_walk":
                    with gin.config_scope("cam"):
                        policy = _default_policy.AnimPolicyRandomWalkLookaround()
                case "random_walk_forward":
                    with gin.config_scope("cam"):
                        policy = _default_policy.AnimPolicyRandomForwardWalk()
                case "follow_poi":
                    policy = _default_policy.AnimPolicyFollowObject(
                        target_obj=cam, pois=pois, bvh=scene_preprocessed["scene_bvh"]
                    )
                case "rrt":
                    with gin.config_scope("rrt"):
                        policy = AnimPolicyRRT(obj_groups=obj_groups)
                case _:
                    raise ValueError(f"Invalid animation mode: {animation_mode}")

            logger.info(f"Computing trajectory using {policy=}")

            # keyframe the base trajectory
            try:
                _default_policy.animate_trajectory(
                    obj=cam,
                    bvh=scene_preprocessed["scene_bvh"],
                    policy_func=policy,
                    validate_pose_func=anim_valid_pose_func,
                    fatal=True,
                    verbose=False,
                )
            except ValueError as error:
                logger.info(
                    f"Compute trajectory for base view {i}/{len(base_views)} failed with {error=}"
                )
                continue

            # save the trajectory with deepcopies of rotation_euler and locations on keyframes
            f_curve = cam.animation_data.action.fcurves.find("location")
            keyframe_list = [
                int(keyframe.co[0]) for keyframe in f_curve.keyframe_points
            ]
            original_locations = list()
            original_rotations = list()
            for frame in keyframe_list:
                _policy.set_frame(frame)
                original_locations.append(deepcopy(cam.location))
                original_rotations.append(deepcopy(cam.rotation_euler))

            # compute score for each frame on the trajectory
            current_traj_scores = []
            for frame in range(start, end + 1, 2):
                _policy.set_frame(frame)
                dists, _, _ = cam_util.terrain_camera_query(
                    cam,
                    scene_preprocessed["scene_bvh"],
                    camera_selection_answers,
                    scene_preprocessed["vertexwise_min_dist"],
                )
                current_traj_scores.append(np.std(dists))

            # remove camera keyframes from path to allow for keyframing of subsequent proposals
            _default.delete_prior_keyframes(cam)

            # store average score and copies of current trajectory proposal
            if len(current_traj_scores) == 0:
                logger.info("Failed to validate trajectory")
                continue

            avg_score = np.mean(current_traj_scores)
            trajectory_proposals.append(
                (
                    avg_score,
                    deepcopy(keyframe_list),
                    deepcopy(original_locations),
                    deepcopy(original_rotations),
                    focus_dist,
                )
            )
            pbar.update()
            if len(trajectory_proposals) >= n_min_candidates:
                break

    butil.delete(cam)
    if len(trajectory_proposals) < n_cams:
        raise ValueError(f"Could not find {n_cams} trajectory proposals")
    return sorted(trajectory_proposals, reverse=True)[:n_cams]
