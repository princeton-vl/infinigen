# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Zeyu Ma, Lahav Lipson: Stationary camera selection
# - Alex Raistrick: Refactor into proposal/validate, camera animation
# - Lingjie Mei: get_camera_trajectory
# Date Signed: May 2 2023

from random import sample
import sys
import warnings
from copy import deepcopy
from functools import partial
import logging
from pathlib import Path

from numpy.random import uniform as U

import bpy
import bpy_extras
import gin
import imageio
import numpy as np
from mathutils import Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree

from assets.utils.decorate import toggle_hide
from rendering.post_render import depth_to_jet
from tqdm import tqdm, trange
from placement import placement

from nodes import node_utils
from nodes.node_wrangler import NodeWrangler, Nodes

from . import animation_policy

from util import blender as butil
from util.logging import Timer
from util.math import clip_gaussian, lerp
from util import camera
from util.random import random_general

logger = logging.getLogger(__name__)

@gin.configurable
def get_sensor_coords(cam, H, W, sparse=False):

    camd = cam.data
    f_in_m = camd.lens / 1000
    scene = bpy.context.scene
    resolution_x_in_px = W
    resolution_y_in_px = H

    scale = scene.render.resolution_percentage / 100
    sensor_width_in_m = camd.sensor_width / 1000
    sensor_height_in_m = camd.sensor_height / 1000
    assert abs(sensor_width_in_m/sensor_height_in_m - W/H) < 1e-4, (sensor_width_in_m, sensor_height_in_m, W, H)

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):

        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_m / pixel_aspect_ratio  # pixels per milimeter
        s_v = resolution_y_in_px * scale / sensor_height_in_m
        
    else: # 'HORIZONTAL' and 'AUTO'

        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_m
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_m

    u_0 = resolution_x_in_px * scale / 2 # cx (in pixels) Usually is just W/2
    v_0 = resolution_y_in_px * scale / 2 # cx (in pixels) Usually is just H/2
    xx, yy = np.meshgrid(np.arange(W).astype(float), np.arange(H).astype(float))
    coords_x = (xx - u_0) / s_u # relative, in mm
    coords_y = (yy - v_0 + 1) / s_v # relative, in mm

    coords_z = np.full(coords_x.shape, -f_in_m)
    relative_cam_coords = np.stack((coords_x, coords_y, coords_z), axis=-1)

    cam_coords_vectors = np.empty((H,W), dtype=Vector)
    pixel_locs = np.stack((np.meshgrid(np.arange(W), np.arange(H))), axis=-1).reshape((W*H, 2))#np.array(list(product(range(H), range(W))))
    if sparse:
        ii = np.random.choice(H*W, size=1000)
        pixel_locs = pixel_locs[ii]

    for x,y in tqdm(pixel_locs, desc="Building Camera Vectors", disable=True):
        pixelVector = Vector(relative_cam_coords[y,x])
        cam_coords_vectors[y,x] = cam.matrix_world @ pixelVector

    return cam_coords_vectors, pixel_locs

def adjust_camera_sensor(cam):
    scene = bpy.context.scene
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    sensor_width = 18 * (W/H)
    assert sensor_width.is_integer(), (18, W, H)
    cam.data.sensor_height = 18
    cam.data.sensor_width = int(sensor_width)

def spawn_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.clip_end = 1e4
    adjust_camera_sensor(cam)
    return cam

def camera_name(rig_id, cam_id):
    return f'CameraRigs/{rig_id}/{cam_id}'

@gin.configurable
def spawn_camera_rigs(
    camera_rig_config,
    n_camera_rigs,
):

    def spawn_rig(i):
        rig_parent = butil.spawn_empty(f'CameraRigs/{i}')
        for j, config in enumerate(camera_rig_config):
            cam = spawn_camera()
            cam.name = camera_name(i, j)
            cam.parent = rig_parent

            cam.location = config['loc']
            cam.rotation_euler = config['rot_euler']

        return rig_parent

    camera_rigs = [spawn_rig(i) for i in range(n_camera_rigs)]
    butil.group_in_collection(camera_rigs, 'CameraRigs')
        
    return camera_rigs

def get_camera(rig_id, subcam_id, checkonly=False):
    col = bpy.data.collections['CameraRigs']
    name = camera_name(rig_id, subcam_id)
    if name in col.objects.keys():
        return col.objects[name]
    if checkonly: return None
    raise ValueError(f'Could not get_camera({rig_id=}, {subcam_id=}). {list(col.objects.keys())=}')

@node_utils.to_nodegroup('nodegroup_camera_info', singleton=True, type='GeometryNodeTree')
def nodegroup_active_cam_info(nw: NodeWrangler):
    info = nw.new_node(Nodes.ObjectInfo, [bpy.context.scene.camera])
    nw.new_node(Nodes.GroupOutput, input_kwargs={
        k: info.outputs[k] for k in info.outputs.keys()
    })

def set_active_camera(rig_id, subcam_id):
    camera = get_camera(rig_id, subcam_id)
    bpy.context.scene.camera = camera

    ng = nodegroup_active_cam_info() # does not create a new node group, retrieves singleton
    ng.nodes['Object Info'].inputs['Object'].default_value = camera

    return bpy.context.scene.camera

def positive_gaussian(mean, std):
    while True:
        val = np.random.normal(mean, std)
        if val > 0:
            return val

def set_camera(
    camera,
    location,
    rotation,
    focus_dist,
    frame,
):
    camera.location = location
    camera.rotation_euler = rotation
    if focus_dist is not None:
        camera.data.dof.focus_distance = focus_dist # this should come before view_layer.update()
    bpy.context.view_layer.update()

    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    if focus_dist is not None:
        camera.data.dof.keyframe_insert(data_path="focus_distance", frame=frame)

def terrain_camera_query(cam, terrain_bvh, terrain_tags_queries, vertexwise_min_dist, min_dist=0):

    dists = []
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=True)
    terrain_tags_queries_counts = {q: 0 for q in terrain_tags_queries}

    for x,y in pix_it:
        direction = (sensor_coords[y,x] - cam.matrix_world.translation).normalized()
        _, _, index, dist = terrain_bvh.ray_cast(cam.matrix_world.translation, direction)
        if dist is None:
            continue
        dists.append(dist)
        if dist < min_dist or dist < vertexwise_min_dist[index]:
            dists = None # means dist < min
            break
        for q in terrain_tags_queries:
            terrain_tags_queries_counts[q] += terrain_tags_queries[q][index]

    n_pix = pix_it.shape[0]

    return dists, terrain_tags_queries_counts, n_pix

@gin.configurable
def camera_pose_proposal(terrain_bvh, terrain_bbox, altitude=2, pitch=90, roll=0, headspace_retries=30):

    loc = np.random.uniform(*terrain_bbox)

    alt = animation_policy.get_altitude(loc, terrain_bvh)
    if alt is None: return None

    headspace = animation_policy.get_altitude(loc, terrain_bvh, dir=Vector((0, 0, 1)))
    for headspace_retry in range(headspace_retries):
        desired_alt = random_general(altitude)
        if desired_alt is None:
            zoff = 0
            break
        zoff = desired_alt - alt
        if headspace is None:
            break
        if desired_alt < headspace:
            break
        logger.debug(f'camera_pose_proposal failed {headspace_retry=} due to {headspace=} {desired_alt=} {alt=}')
    else: # for-else triggers if no break, IE no acceptable voffset was found
        logger.warning(f'camera_pose_proposal found no acceptable zoff after {headspace_retries=}')
        return None

    loc[2] = loc[2] + zoff
    if loc[2] > terrain_bbox[1][2] or loc[2] < terrain_bbox[0][2]: return None

    yaw = np.random.uniform(-180, 180)
    rot =  np.deg2rad([random_general(pitch), random_general(roll), yaw])

    return loc, rot

@gin.configurable
def keep_cam_pose_proposal(
    cam,
    terrain,
    terrain_bvh, 
    placeholders_kd, 
    terrain_tags_answers,
    vertexwise_min_dist,
    terrain_tags_ratio,
    min_placeholder_dist=1,
    min_terrain_distance=0,
    terrain_coverage_range=(0.5, 1),
):
    # Reject cameras inside of terrain volumes
    terrain_sdf = terrain.compute_camera_space_sdf(np.array(cam.location).reshape((1, 3)))

    if not cam.type == 'CAMERA':
        cam = cam.children[0]
    if not cam.type == 'CAMERA':
        raise ValueError(f'{cam.name=} had {cam.type=}')


    if terrain_sdf <= 0:
        logger.debug(f'keep_cam_pose_proposal rejects {terrain_sdf=}')
        return None

    bpy.context.view_layer.update()
    
    # Reject cameras too close to any placeholder vertex
    v, i, dist_to_placeholder = placeholders_kd.find(cam.matrix_world.translation)
    if dist_to_placeholder is not None and dist_to_placeholder < min_placeholder_dist:
        logger.debug(f'keep_cam_pose_proposal rejects {dist_to_placeholder=}, {v, i}')
        return None

    dists, terrain_tags_answers_counts, n_pix = terrain_camera_query(
        cam, terrain_bvh, terrain_tags_answers, vertexwise_min_dist, min_dist=min_terrain_distance)
    
    if dists is None:
        logger.debug(f'keep_cam_pose_proposal rejects terrain dists')
        return None
    
    coverage = len(dists)/n_pix
    if coverage < terrain_coverage_range[0] or coverage > terrain_coverage_range[1]:
        return None

    if rparams := terrain_tags_ratio:
        for q in rparams:
            if type(q) is tuple and q[0] == "closeup":
                closeup = len([d for d in dists if d < q[1]])/n_pix
                if closeup < rparams[q][0] or closeup > rparams[q][1]:
                    return None
            else:
                minv, maxv = rparams[q][0], rparams[q][1]
                if q in terrain_tags_answers_counts:
                    ratio = terrain_tags_answers_counts[q] / n_pix
                    if ratio < minv or ratio > maxv:
                        return None

    return np.std(dists)
    
@gin.configurable
class AnimPolicyGoToProposals:

    def __init__(self, speed=("uniform", 1.5, 2.5), min_dist=4, max_dist=10, retries=30):
        self.speed=speed
        self.min_dist=min_dist
        self.max_dist=max_dist
        self.retries=retries

    def __call__(self, camera_rig, frame_curr, retry_pct, bvh):
        margin = Vector((self.max_dist, self.max_dist, self.max_dist))
        bbox = (camera_rig.location - margin, camera_rig.location + margin)

        for _ in range(self.retries):
            res = camera_pose_proposal(bvh, bbox)
            if res is None:
                continue
            pos, rot = res
            pos = np.array(pos)
            if np.linalg.norm(pos - np.array(camera_rig.location)) < self.min_dist:
                continue
            break
        else:
            raise animation_policy.PolicyError(f'{__name__} found no keyframe after {self.retries=}')

        time = np.linalg.norm(pos - camera_rig.location) / random_general(self.speed)
        return Vector(pos), Vector(rot), time, 'BEZIER'
   
@gin.configurable
def compute_base_views(
    cam, n_views,
    terrain, terrain_bvh, terrain_bbox, terrain_tags_answers, vertexwise_min_dist,
    terrain_tags_ratio,
    placeholders_kd,
    min_candidates_ratio=20,
    max_tries=10000,
):
    potential_views = []
    n_min_candidates = int(min_candidates_ratio * n_views)
    with tqdm(total=n_min_candidates, desc='Searching for camera viewpoints') as pbar:
        for it in range(1, max_tries):

            props = camera_pose_proposal(terrain_bvh=terrain_bvh, terrain_bbox=terrain_bbox)
            if props is None: continue
            loc, rot = props

            cam.location = loc
            cam.rotation_euler = rot

            criterion = keep_cam_pose_proposal(
                cam, terrain, terrain_bvh, placeholders_kd,
                terrain_tags_answers=terrain_tags_answers,
                vertexwise_min_dist=vertexwise_min_dist,
                terrain_tags_ratio=terrain_tags_ratio,
            )
            if criterion is None:
                continue

            # Compute focus distance
            destination = cam.matrix_world @ Vector((0.,0.,-1.))
            forward_dir = (destination - cam.location).normalized()
            *_, straight_ahead_dist = terrain_bvh.ray_cast(cam.location, forward_dir)

            potential_views.append((criterion, deepcopy(cam.location), deepcopy(cam.rotation_euler), straight_ahead_dist))
            pbar.update(1)

            if len(potential_views) >= n_min_candidates:
                break
    assert potential_views != [], "no camera view found"
    return sorted(potential_views, reverse=True)[:n_views]

@gin.configurable
def camera_selection_preprocessing(
    terrain, terrain_tags_ratio={},
):

    with Timer(f'Building terrain BVHTree'):
        terrain_bvh, terrain_tags_answers, vertexwise_min_dist = terrain.build_terrain_bvh_and_attrs(terrain_tags_ratio.keys())

    with Timer(f'Building placeholders KDTree'):
        placeholders_kd = placement.placeholder_kd()

    return dict(
        terrain_bvh=terrain_bvh,
        terrain=terrain,
        terrain_tags_answers=terrain_tags_answers,
        vertexwise_min_dist=vertexwise_min_dist,
        placeholders_kd=placeholders_kd,
        terrain_tags_ratio=terrain_tags_ratio,
    )

@gin.configurable
def configure_cameras(
    cam_rigs,
    scene_preprocessed,
    terrain,
):
    bpy.context.view_layer.update()
    dummy_camera = spawn_camera()

    terrain_bbox = terrain.get_bounding_box()

    base_views = compute_base_views(dummy_camera, n_views=len(cam_rigs), 
        terrain_bbox=terrain_bbox, **scene_preprocessed)

    for view, cam_rig in zip(base_views, cam_rigs):
        
        score, loc, rot, focus_dist = view
        cam_rig.location = loc
        cam_rig.rotation_euler = rot

        if focus_dist is not None:
            for cam in cam_rig.children:
                if not cam.type =='CAMERA': continue
                cam.data.dof.focus_distance = focus_dist

    butil.delete(dummy_camera)

@gin.configurable
def animate_cameras(
    cam_rigs, scene_preprocessed, pois=None,
    follow_poi_chance=0.0,
    strict_selection=False,
):
    
    anim_valid_pose_func = partial(
        keep_cam_pose_proposal,
        terrain_bvh=scene_preprocessed['terrain_bvh'],
        terrain=scene_preprocessed['terrain'],
        placeholders_kd=scene_preprocessed['placeholders_kd'],
        terrain_tags_answers=scene_preprocessed['terrain_tags_answers'] if strict_selection else {},
        vertexwise_min_dist=scene_preprocessed['vertexwise_min_dist'],
        terrain_tags_ratio=scene_preprocessed['terrain_tags_ratio'] if strict_selection else {},
    )


    for cam_rig in cam_rigs:       
        if U() < follow_poi_chance and pois is not None and len(pois):
            policy = animation_policy.AnimPolicyFollowObject(
                target_obj=cam_rig, pois=pois, bvh=scene_preprocessed['terrain_bvh'])
        else:
            policy = animation_policy.AnimPolicyRandomWalkLookaround()
        logger.info(f'Animating {cam_rig=} using {policy=}')
        animation_policy.animate_trajectory(cam_rig, scene_preprocessed['terrain_bvh'],
            policy_func=policy,
            validate_pose_func=anim_valid_pose_func, verbose=True, 
            fatal=True)

@gin.configurable
def save_camera_parameters(camera_pair_id, camera_ids, output_folder, frame, use_dof=False):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    for camera_id in camera_ids:
        camera_obj = get_camera(camera_pair_id, camera_id)
        if use_dof is not None:
            camera_obj.data.dof.use_dof = use_dof
        # Saving camera parameters
        K = camera.get_calibration_matrix_K_from_blender(camera_obj.data)
        np.save(
            output_folder / f"K{frame:04d}_{camera_pair_id:02d}_{camera_id:02d}.npy",
            np.asarray(K, dtype=np.float64),
        )
        np.save(
            output_folder / f"T{frame:04d}_{camera_pair_id:02d}_{camera_id:02d}.npy",
            np.asarray(camera_obj.matrix_world, dtype=np.float64) @ np.diag((1.,-1.,-1.,1.)),
    )

@node_utils.to_nodegroup('ng_dist2camera', singleton=True, type='GeometryNodeTree')
def ng_dist2camera(nw: NodeWrangler):
    traj = get_camera_trajectory()
    trajectory = nw.new_node(Nodes.ObjectInfo, [traj]).outputs['Geometry']
    distance = nw.new_node(Nodes.Proximity, [trajectory], attrs={'target_element': 'POINTS'}).outputs['Distance']
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Distance": distance})


def get_camera_trajectory():
    current_frame = bpy.context.scene.frame_current
    if 'Cameras' in bpy.data.collections:
        locations = []
        for camera_pair in bpy.data.collections['Cameras'].children:
            for camera in camera_pair.objects:
                for f in camera.animation_data.action.fcurves:
                    for k in f.keyframe_points:
                        fr = int(k.co[0])
                        bpy.context.scene.frame_set(fr)
                        locations.append(camera.location)
        bpy.context.scene.frame_set(current_frame)
    else:
        locations = [[100] * 3]
    mesh = bpy.data.meshes.new('temp')
    mesh.from_pydata(np.vstack(locations), [], [])
    mesh.update()
    obj = bpy.data.objects.new('temp', mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    butil.modify_mesh(obj, 'WELD', merge_threshold=.1)
    toggle_hide(obj)
    return obj

if __name__ == "__main__":
    """
    This interactive section generates a depth map by raycasting through each pixel. 
    It is very useful for debugging camera.py.
    """
    cam = bpy.context.scene.camera

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    adjust_camera_sensor(cam)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvhtree = BVHTree.FromObject(bpy.context.active_object, depsgraph)

    target_obj = bpy.context.active_object
    to_obj_coords = target_obj.matrix_world.inverted()
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=False)

    H,W = sensor_coords.shape
    depth_output = np.zeros((H,W), dtype=np.float64)

    for x,y in tqdm(pix_it):
        destination = sensor_coords[y,x]
        direction = (destination - cam.location).normalized()
        location, normal, index, dist = bvhtree.ray_cast(cam.location, direction)
        if dist is not None:
            dist_diff = (destination - cam.location).length
            assert dist > (location - destination).length, (dist, (location - destination).length)
            assert dist > dist_diff
            depth_output[H-y-1,x] = dist - dist_diff

    color_depth = depth_to_jet(depth_output)
    imageio.imwrite(f"color_depth.png", color_depth)
