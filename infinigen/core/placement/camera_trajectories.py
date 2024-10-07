# Copyright (c) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Dylan Li: primary author
# - Sumanth Maddirala: base view selection

import logging
from copy import deepcopy
from functools import partial

import bpy
import gin
import numpy as np
from numpy.random import uniform as U
from tqdm import tqdm

from infinigen.core.placement.camera import terrain_camera_query
from infinigen.core.util import blender as butil

from . import animation_policy
from . import camera as cam_util

logger = logging.getLogger(__name__)


# replaces configure cameras
@gin.configurable
def compute_trajectories(
    cam,
    n_trajectories,
    terrain,
    bbox,
    scene_bvh,
    placeholders_kd=None,
    camera_selection_answers={},
    vertexwise_min_dist=None,
    camera_selection_ratio=None,
    min_candidates_ratio=5,
    min_base_views_ratio=20,
    pois=None,
    follow_poi_chance=0.0,
    policy_registry=None,
    validate_pose_func=None,
):
    trajectory_proposals = []
    n_min_candidates = int(min_candidates_ratio * n_trajectories)

    bpy.context.scene.frame_end = 200
    start = bpy.context.scene.frame_start
    end = bpy.context.scene.frame_end
    if end <= start:
        return []
    if validate_pose_func is None:
        anim_valid_pose_func = partial(
            cam_util.keep_cam_pose_proposal,
            placeholders_kd=placeholders_kd,
            scene_bvh=scene_bvh,
            terrain=terrain,
            camera_selection_answers=camera_selection_answers,
            vertexwise_min_dist=vertexwise_min_dist,
            camera_selection_ratio=camera_selection_ratio,
        )
    else:
        anim_valid_pose_func = validate_pose_func

    def location_sample():
        return np.random.uniform(*bbox)

    # generate 2 * n_trajectories base_views from which we should generate potential trajectories
    # pregenerating base views helps avoid bad trajectory suggestions
    base_view_selections = cam_util.compute_base_views(
        cam,
        n_views=n_min_candidates * min_base_views_ratio,
        terrain=terrain,
        scene_bvh=scene_bvh,
        location_sample=location_sample,
        placeholders_kd=placeholders_kd,
        vertexwise_min_dist=vertexwise_min_dist,
        min_candidates_ratio=1,
        camera_selection_ratio=camera_selection_ratio,
    )
    with tqdm(
        total=n_min_candidates, desc="Searching for potential trajectories"
    ) as pbar:
        for view in base_view_selections:
            # set initial base view, and then propose a trajectory afterwards
            score, proposal, focus_dist = view
            cam.location = proposal.loc
            cam.rotation_euler = proposal.rot

            if not anim_valid_pose_func(cam):
                continue

            if policy_registry is None:
                if U() < follow_poi_chance and pois is not None and len(pois):
                    policy = animation_policy.AnimPolicyFollowObject(
                        target_obj=cam, pois=pois, bvh=scene_bvh
                    )
                else:
                    with gin.config_scope("cam"):
                        policy = animation_policy.AnimPolicyRandomWalkLookaround()
            else:
                policy = policy_registry()

            logger.info(f"Animating {cam=} using {policy=}")

            # keyframe the base trajectory
            try:
                animation_policy.animate_trajectory(
                    obj=cam,
                    bvh=scene_bvh,
                    policy_func=policy,
                    validate_pose_func=anim_valid_pose_func,
                    fatal=True,
                    verbose=False,
                )
            except ValueError as err:
                logger.info(err)
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
                    cam, scene_bvh, camera_selection_answers, vertexwise_min_dist
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

    if len(trajectory_proposals) < n_trajectories:
        raise ValueError(f"Could not find {n_trajectories} trajectory proposals")
    return sorted(trajectory_proposals, reverse=True)[:n_trajectories]


@gin.configurable
def animate_trajectories(
    cam_rigs,
    bounding_box,
    scene_preprocessed,
    follow_poi_chance=0.0,
    pois=None,
    policy_registry=None,
    validate_pose_func=None,
):
    bpy.context.view_layer.update()
    dummy_camera = cam_util.spawn_camera()

    # generate potential trajectories
    trajectories = compute_trajectories(
        dummy_camera,
        n_trajectories=len(cam_rigs),
        terrain=scene_preprocessed["terrain"],
        scene_bvh=scene_preprocessed["scene_bvh"],
        bbox=bounding_box,
        placeholders_kd=scene_preprocessed["placeholders_kd"],
        vertexwise_min_dist=scene_preprocessed["vertexwise_min_dist"],
        follow_poi_chance=follow_poi_chance,
        pois=pois,
        policy_registry=policy_registry,
        validate_pose_func=validate_pose_func,
    )

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
    butil.delete(dummy_camera)


# deletes keyframes on camera from prior animation phase
def delete_prior_keyframes(cam):
    f_curve = cam.animation_data.action.fcurves.find("location")
    keyframe_list = [int(keyframe.co[0]) for keyframe in f_curve.keyframe_points]
    for frame in keyframe_list:
        cam.keyframe_delete(data_path="location", frame=frame)
        cam.keyframe_delete(data_path="rotation_euler", frame=frame)
