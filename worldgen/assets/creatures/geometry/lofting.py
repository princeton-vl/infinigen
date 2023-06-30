from dataclasses import dataclass
import pdb

import bpy
import bmesh

import numpy as np

from util import blender as butil
from util.math import rotate_match_directions, lerp_sample, inverse_interpolate
from .nurbs import nurbs
from assets.creatures.geometry.cpp_utils import bnurbs

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
    '''

    ts: np.array # shape (N) float
    
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


    k = len(skeleton)

    # default angles point index 0 to -z by convention
        angles = default_profile_angles(m)

    # decide the axes of rotation for each integer distance along the skeleton
    axes = skeleton_to_tangents(skeleton)
    
    # user gives t in [0, 1] representing percent of distance along skeleton
    #ts = dist_pcts_to_ts(skeleton, ts)
    axes = lerp_sample(axes, ts * (k - 1))
    pos = lerp_sample(skeleton, ts * (k - 1))
    
    # compute profile shapes

    # pose profiles to get vert locations
    forward = np.zeros_like(axes)
    forward[:, 0] = 1
    rot_mats = rotate_match_directions(forward, axes)
    profile_verts = np.einsum('bij,bvj->bvi', rot_mats, profile_verts) + pos[:, None]

    return profile_verts

def loft(skeleton, skin, method='blender', face_size=0.01, debug=False, **kwargs):
    
    obj = nurbs(ctrlpts, method, face_size, debug, **kwargs)

    if debug:
        skeleton_debug = butil.spawn_point_cloud('skeleton_debug', skeleton)
        skeleton_debug.parent = obj

