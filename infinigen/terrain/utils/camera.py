# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import bpy
import gin
import numpy as np
from infinigen.core.placement.camera import get_camera
from scipy.spatial.transform import Rotation as R


def getK(fov, H, W):
    fx = W / 2 / np.tan(fov[1] / 2)
    fy = fx
    return np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])

def pose_average(poses):
    translation = poses[:, :3, 3].mean(axis=0)
    quats = []
    for pose in poses:
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        if len(quats) > 1 and np.dot(quat, quats[-1]) < 0:
            quat *= -1
        quats.append(quat)
    quat = np.mean(np.asarray(quats), axis=0)
    res = np.eye(4)
    res[:3, :3] = R.from_quat(quat).as_matrix()
    res[:3, 3] = translation
    return res

def get_expanded_fov(cam_pose0, cam_poses, fov):
    rot0 = cam_pose0[:3, :3]
    bounds = np.array([1e9, -1e9, 1e9, -1e9])
    for cam_pose in cam_poses:
        rot = cam_pose[:3, :3]
        for i in [-1, 1]:
            for j in [-1, 1]:
                p = [np.tan(fov[1] / 2) * i, np.tan(fov[0] / 2) * j, 1]
                p = np.dot(np.linalg.inv(rot0), np.dot(rot, p))
                bounds[0] = min(bounds[0], p[0] / p[2])
                bounds[1] = max(bounds[1], p[0] / p[2])
                bounds[2] = min(bounds[2], p[1] / p[2])
                bounds[3] = max(bounds[3], p[1] / p[2])
    return (max(-bounds[2], bounds[3]) * 2, max(-bounds[0], bounds[1]) * 2)


@gin.configurable
def get_caminfo(cameras, relax=1.05):
    cam_poses = []
    coords_trans_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    fs, fe = bpy.context.scene.frame_start, bpy.context.scene.frame_end
    fc = bpy.context.scene.frame_current
    for f in range(fs, fe + 1):
        bpy.context.scene.frame_set(f)
        for cam in cameras:
            cam_pose = np.array(cam.matrix_world)
            cam_pose = np.dot(np.array(cam_pose), coords_trans_matrix)
            cam_poses.append(cam_pose)
            fov_rad  = cam.data.angle
            _, cid0, cid1 = cam.name.split("/")
            cam = get_camera(cid0, 1 - int(cid1), 1)
            if cam is None: continue
            cam_pose = np.array(cam.matrix_world)
            cam_pose = np.dot(np.array(cam_pose), coords_trans_matrix)
            cam_poses.append(cam_pose)
    bpy.context.scene.frame_set(fc)
    cam_poses = np.stack(cam_poses)
    cam_pose = pose_average(cam_poses)
    fov_rad *= relax
    H, W = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x
    fov0 = np.arctan(H / 2 / (W / 2 / np.tan(fov_rad / 2))) * 2
    fov = (fov0, fov_rad)
    fov = get_expanded_fov(cam_pose, cam_poses, fov)
    K = getK(fov, H, W)
    return cam_pose, cam_poses, fov, H, W, K
