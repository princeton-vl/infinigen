# Copyright (c) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original placement/camera.py: camera pose proposal, camera random walk animations
# - Dylan Li: refactored to camera_trajectories.py to support RRT option
# - Sumanth Maddirala: base view selection

import logging
from copy import deepcopy
from functools import partial

import bpy
import gin
import numpy as np
from tqdm import tqdm

from infinigen.core.placement.camera import configure_cameras, terrain_camera_query
from infinigen.core.util import blender as butil
from infinigen.core.util.rrt import AnimPolicyRRT

from . import animation_policy
from . import camera as cam_util

logger = logging.getLogger(__name__)


@gin.configurable
def compute_poses(
    cam_rigs,
    scene_preprocessed: dict,
    init_bounding_box: tuple[np.array, np.array] = None,
    init_surfaces: list[bpy.types.Object] = None,
    terrain_mesh=None,
    min_candidates_ratio=5,
    min_base_views_ratio=10,
):
    n_cams = len(cam_rigs)
    cam = cam_util.spawn_camera()

    # num trajectories to fully compute and score
    n_min_candidates = int(min_candidates_ratio * n_cams)

    start = bpy.context.scene.frame_start
    end = bpy.context.scene.frame_end

    if end <= start:
        configure_cameras(
            cam_rigs=cam_rigs,
            scene_preprocessed=scene_preprocessed,
            init_bounding_box=init_bounding_box,
            init_surfaces=init_surfaces,
            terrain_mesh=terrain_mesh,
        )
        butil.delete(cam)
        return []

    base_views = configure_cameras(
        cam_rigs=cam_rigs,
        scene_preprocessed=scene_preprocessed,
        init_bounding_box=init_bounding_box,
        init_surfaces=init_surfaces,
        terrain_mesh=terrain_mesh,
        n_views=n_min_candidates * min_base_views_ratio,
    )

    butil.delete(cam)
    return base_views


@gin.configurable
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

    with tqdm(
        total=n_min_candidates, desc="Searching for potential trajectories"
    ) as pbar:
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
                        policy = animation_policy.AnimPolicyRandomWalkLookaround()
                case "random_walk_forward":
                    with gin.config_scope("cam"):
                        policy = animation_policy.AnimPolicyRandomForwardWalk()
                case "follow_poi":
                    policy = animation_policy.AnimPolicyFollowObject(
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
                animation_policy.animate_trajectory(
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
                bpy.context.scene.frame_set(frame)
                original_locations.append(deepcopy(cam.location))
                original_rotations.append(deepcopy(cam.rotation_euler))

            # compute score for each frame on the trajectory
            current_traj_scores = []
            for frame in range(start, end + 1, 2):
                bpy.context.scene.frame_set(frame)
                dists, _, _ = terrain_camera_query(
                    cam,
                    scene_preprocessed["scene_bvh"],
                    camera_selection_answers,
                    scene_preprocessed["vertexwise_min_dist"],
                )
                current_traj_scores.append(np.std(dists))

            # remove camera keyframes from path to allow for keyframing of subsequent proposals
            delete_prior_keyframes(cam)

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


@gin.configurable
def animate_trajectories(
    cam_rigs,
    base_views,
    scene_preprocessed: dict,
    obj_groups=None,
    follow_poi_chance=0.0,
    pois=None,
    animation_mode="random_walk",
    validate_pose_func=None,
    fatal=True,
):
    bpy.context.view_layer.update()

    try:
        trajectories = compute_trajectories(
            cam_rigs=cam_rigs,
            base_views=base_views,
            scene_preprocessed=scene_preprocessed,
            obj_groups=obj_groups,
            follow_poi_chance=follow_poi_chance,
            pois=pois,
            animation_mode=animation_mode,
            validate_pose_func=validate_pose_func,
        )
    except ValueError as err:
        if fatal:
            raise ValueError(err)
        else:
            logger.warning(err)
            return

    # keyframe in trajectory for each camera rig
    for cam_rig, trajectory in zip(cam_rigs, trajectories):
        for keyframe, location, rotation in zip(
            trajectory[1], trajectory[2], trajectory[3]
        ):
            cam_rig.location = location
            cam_rig.rotation_euler = rotation
            cam_rig.keyframe_insert(data_path="location", frame=keyframe)
            cam_rig.keyframe_insert(data_path="rotation_euler", frame=keyframe)

        # set focus distance on children when available
        if trajectory[4] is not None:
            for cam in cam_rig.children:
                if not cam.type == "CAMERA":
                    continue
                cam.data.dof.focus_distance = trajectory[4]


# deletes keyframes on camera from prior animation phase
def delete_prior_keyframes(cam):
    f_curve = cam.animation_data.action.fcurves.find("location")
    keyframe_list = [int(keyframe.co[0]) for keyframe in f_curve.keyframe_points]
    for frame in keyframe_list:
        cam.keyframe_delete(data_path="location", frame=frame)
        cam.keyframe_delete(data_path="rotation_euler", frame=frame)
