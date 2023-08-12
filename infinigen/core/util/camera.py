# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import numpy as np

from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import bpy
import bpy_extras
from tqdm import trange

from infinigen.core.util.math import homogenize, dehomogenize
from infinigen.core.util import blender as butil

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    W = resolution_x_in_px = scene.render.resolution_x
    H = resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height

    if sensor_width_in_mm/sensor_height_in_mm != W/H:
        vals = f'{(sensor_width_in_mm, sensor_height_in_mm, W, H)=}'
        raise ValueError(f'Camera sensor has not been properly configured, you probably need to call camera.adjust_camera_sensor on it. {vals}')
    
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio  # pixels per milimeter
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

def compute_vis_dists(points, cam):
    projmat, K, RT = map(np.array, get_3x4_P_matrix_from_blender(cam))
    proj = points @ projmat.T
    uv, d = dehomogenize(proj), proj[:, -1]

    clamped_uv = np.clip(uv, [0, 0], butil.get_camera_res())
    clamped_d = np.maximum(d, 0)
    
    RT_4x4_inv = np.array(Matrix(RT).to_4x4().inverted())
    clipped_pos = homogenize((homogenize(clamped_uv) * clamped_d[:, None]) @ np.linalg.inv(K).T) @ RT_4x4_inv.T
    
    vis_dist = np.linalg.norm(points[:, :-1] - clipped_pos[:, :-1], axis=-1)

    return d, vis_dist

def min_dists_from_cam_trajectory(points, cam, start=None, end=None, verbose=False):
    
    assert len(points.shape) == 2 and points.shape[-1] == 3
    assert cam.type == 'CAMERA'

    if start is None:
        start = bpy.context.scene.frame_start
    if end is None:
        end = bpy.context.scene.frame_end

    points = homogenize(points)
    min_dists = np.full(len(points), 1e7)
    min_vis_dists = np.full(len(points), 1e7)

    rangeiter = trange if verbose else range
    for i in rangeiter(start, end+1):
        bpy.context.scene.frame_set(i)
        dists, vis_dists = compute_vis_dists(points, cam)
        min_dists = np.minimum(dists, min_dists)
        min_vis_dists = np.minimum(vis_dists, min_vis_dists)
    
    return min_dists, min_vis_dists
    



