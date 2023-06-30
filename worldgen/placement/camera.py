from random import sample
from copy import deepcopy
from functools import partial
import logging

from numpy.random import uniform as U

import bpy
import gin
import numpy as np
from mathutils import Matrix, Vector, Euler
from tqdm import tqdm, trange
from placement import placement
from nodes import node_utils
from nodes.node_wrangler import NodeWrangler, Nodes

from . import animation_policy

from util import blender as butil
from util.logging import Timer
from util.math import clip_gaussian, lerp
from util.random import random_general
logger = logging.getLogger(__name__)




        


    scene = bpy.context.scene
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    sensor_width = 18 * (W/H)
    assert sensor_width.is_integer(), (18, W, H)
    cam.data.sensor_height = 18
    cam.data.sensor_width = int(sensor_width)
    return cam
def camera_name(rig_id, cam_id):
    return f'CameraRigs/{rig_id}/{cam_id}'

@gin.configurable
def spawn_camera_rigs(
    camera_rig_config,
    n_camera_rigs,

    def spawn_rig(i):
        rig_parent = butil.spawn_empty(f'CameraRigs/{i}')
        for j, config in enumerate(camera_rig_config):
            cam.name = camera_name(i, j)
            cam.parent = rig_parent
            cam.location = config['loc']
            cam.rotation_euler = config['rot_euler']
        return rig_parent

    camera_rigs = [spawn_rig(i) for i in range(n_camera_rigs)]
    butil.group_in_collection(camera_rigs, 'CameraRigs')
        
    return camera_rigs
    col = bpy.data.collections['CameraRigs']
    name = camera_name(rig_id, subcam_id)
    if name in col.objects.keys():
        return col.objects[name]
    raise ValueError(f'Could not get_camera({rig_id=}, {subcam_id=}). {list(col.objects.keys())=}')
@node_utils.to_nodegroup('nodegroup_camera_info', singleton=True, type='GeometryNodeTree')
def nodegroup_active_cam_info(nw: NodeWrangler):
    info = nw.new_node(Nodes.ObjectInfo, [bpy.context.scene.camera])
    nw.new_node(Nodes.GroupOutput, input_kwargs={
        k: info.outputs[k] for k in info.outputs.keys()
    })

def set_active_camera(rig_id, subcam_id):
    camera = get_camera(rig_id, subcam_id)
    ng = nodegroup_active_cam_info() # does not create a new node group, retrieves singleton
    ng.nodes['Object Info'].inputs['Object'].default_value = camera
    return bpy.context.scene.camera
    camera.location = location
    if focus_dist is not None:
        camera.data.dof.keyframe_insert(data_path="focus_distance", frame=frame)


    sensor_coords, pix_it = get_sensor_coords(cam, sparse=True)

    for x,y in pix_it:
        direction = (sensor_coords[y,x] - cam.matrix_world.translation).normalized()
        if dist is None:
            continue
        dists.append(dist)
            break

    n_pix = pix_it.shape[0]

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
    terrain_bvh, 
    placeholders_kd, 
    min_placeholder_dist=1,
    min_terrain_distance=0,
):
    if not cam.type == 'CAMERA':
        cam = cam.children[0]
    if not cam.type == 'CAMERA':
        raise ValueError(f'{cam.name=} had {cam.type=}')

        logger.debug(f'keep_cam_pose_proposal rejects {terrain_sdf=}')
        return None

    bpy.context.view_layer.update()
    
    # Reject cameras too close to any placeholder vertex
    v, i, dist_to_placeholder = placeholders_kd.find(cam.matrix_world.translation)
    if dist_to_placeholder is not None and dist_to_placeholder < min_placeholder_dist:
        logger.debug(f'keep_cam_pose_proposal rejects {dist_to_placeholder=}, {v, i}')
        return None

    
        logger.debug(f'keep_cam_pose_proposal rejects terrain dists')
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
    placeholders_kd,
):
    n_min_candidates = int(min_candidates_ratio * n_views)
    with tqdm(total=n_min_candidates, desc='Searching for camera viewpoints') as pbar:

            props = camera_pose_proposal(terrain_bvh=terrain_bvh, terrain_bbox=terrain_bbox)

                break
    return sorted(potential_views, reverse=True)[:n_views]

@gin.configurable
def camera_selection_preprocessing(
):

    with Timer(f'Building terrain BVHTree'):

    with Timer(f'Building placeholders KDTree'):
        placeholders_kd = placement.placeholder_kd()

    return dict(
        terrain_bvh=terrain_bvh,
        vertexwise_min_dist=vertexwise_min_dist,
        placeholders_kd=placeholders_kd,
    )

@gin.configurable
def configure_cameras(
    cam_rigs,
    scene_preprocessed,
    terrain,
):
    bpy.context.view_layer.update()
    dummy_camera = spawn_camera()


    base_views = compute_base_views(dummy_camera, n_views=len(cam_rigs), 

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
):
    
        placeholders_kd=scene_preprocessed['placeholders_kd'],
        vertexwise_min_dist=scene_preprocessed['vertexwise_min_dist'],


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
