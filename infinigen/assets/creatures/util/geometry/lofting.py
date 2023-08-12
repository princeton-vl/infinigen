# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from dataclasses import dataclass
import pdb

import bpy
import bmesh

import numpy as np

from infinigen.core.util import blender as butil
from infinigen.core.util.math import rotate_match_directions, lerp_sample, inverse_interpolate
from .nurbs import nurbs

def factorize_nurbs_handles(handles):
    
    '''
    Factorize (n,m,3) handles into a skeleton, radii and relative normalized profiles.

    IE, profiles output will all face x axis, and have mean radius ~= 1
    '''

    skeleton_polyline = handles.mean(axis=1)
    tangents = skeleton_to_tangents(skeleton_polyline)

    forward = np.zeros_like(tangents)
    forward[:, 0] = 1
    rot_mats = rotate_match_directions(tangents, forward)

    profiles = handles - skeleton_polyline[:, None]
    profiles = np.einsum('bij,bvj->bvi', rot_mats, profiles)
    
    ts = np.linspace(0, 1, handles.shape[0])
    
    return skeleton_polyline, ts, profiles

@dataclass
class Skin:

    '''
    Defines all the data for a loft mesh besides its skeleton, ie how far
    and what shape should the mesh extend beyond the structure of the skeleton

    N = number of defined profiles along the skeleton
    M = number of points per profile
    '''

    ts: np.array # shape (N) float
    profiles: np.array # shape (N x M) float as polar distances; or NxMx3 as points with x as forward axis
    
    profile_as_points: bool = False # whether to interpret profiles as points

    angles: np.array = None # shape (M) float
    surface_params: np.array = None # shape (N x M x K) float, K is num params per vert

def dist_pcts_to_ts(skeleton, ds):
    lengths = np.linalg.norm(skeleton[1:] - skeleton[:-1], axis=-1)
    dists = np.concatenate([np.array([0]), np.cumsum(lengths)])
    ts = inverse_interpolate(dists, ds * dists[-1]) 
    return ts / (len(skeleton) - 1)

def skeleton_to_tangents(skeleton):
    axes = np.empty_like(skeleton, dtype=np.float32)
    axes[-1] = skeleton[-1] - skeleton[-2]
    axes[:-1] = skeleton[1:] - skeleton[:-1]
    axes[1:-1] = (axes[1:-1] + axes[:-2]) / 2 # use average of neighboring edge directions where available
    
    norm = np.linalg.norm(axes, axis=-1)
    axes[norm > 0] /= norm[norm > 0, None]

    return axes

def default_profile_angles(m):
    return np.linspace(-np.pi/2, 1.5 * np.pi, m, endpoint=False)

def compute_profile_verts(skeleton, ts, profiles, angles=None, profile_as_points=False):

    n, m = profiles.shape[0:2]
    k = len(skeleton)

    # default angles point index 0 to -z by convention
    if angles is None and not profile_as_points:
        angles = default_profile_angles(m)

    # decide the axes of rotation for each integer distance along the skeleton
    axes = skeleton_to_tangents(skeleton)
    
    # user gives t in [0, 1] representing percent of distance along skeleton
    #ts = dist_pcts_to_ts(skeleton, ts)
    axes = lerp_sample(axes, ts * (k - 1))
    pos = lerp_sample(skeleton, ts * (k - 1))
    
    # compute profile shapes
    if profile_as_points:
        assert(profiles.shape[2]==3)
        profile_verts = profiles; 
    else:
        unit_circle = np.stack([np.zeros_like(angles), np.cos(angles), np.sin(angles)], axis=-1)
        profile_verts = profiles[..., None] * unit_circle[None]

    # pose profiles to get vert locations
    forward = np.zeros_like(axes)
    forward[:, 0] = 1
    rot_mats = rotate_match_directions(forward, axes)
    profile_verts = np.einsum('bij,bvj->bvi', rot_mats, profile_verts) + pos[:, None]

    return profile_verts

def loft(skeleton, skin, method='blender', face_size=0.01, debug=False, **kwargs):
    
    ctrlpts = compute_profile_verts(skeleton, skin.ts, skin.profiles, skin.angles, profile_as_points=skin.profile_as_points)
    obj = nurbs(ctrlpts, method, face_size, debug, **kwargs)

    if debug:
        skeleton_debug = butil.spawn_point_cloud('skeleton_debug', skeleton)
        skeleton_debug.parent = obj

    return obj

