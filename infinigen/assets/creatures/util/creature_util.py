# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from dataclasses import dataclass
import numbers

import bpy
import mathutils
from mathutils import Vector, Euler, Quaternion

import numpy as np

from infinigen.core.util.math import lerp
from infinigen.core.util import blender as butil

def euler(r, p, y):
    return mathutils.Euler(np.deg2rad([r, p, y])).to_quaternion()

def interp_dict(a: dict, b: dict, t: float, keys='assert', fill=0, recurse=True, lerp=lerp):

    '''
    keys: 'a', 'b', 'intersect', 'union', 'asset', 'switch'
    '''

    if keys == 'switch':
        keys = 'b' if t > 0.5 else 'a'

    if keys == 'assert':
        if not a.keys() == b.keys():
            raise ValueError(f'lerp_dict(..., {keys=}) recieved {a.keys()=}, {b.keys()}=')
        out_keys = a.keys()
    elif keys == 'a':
        out_keys = a.keys()
    elif keys == 'b':
        out_keys = b.keys()
    elif keys == 'union':
        out_keys = set(a.keys()).union(b.keys())
    elif keys == 'intersect':
        out_keys = set(a.keys()).intersection(b.keys())
    else:
        raise ValueError(f'Unrecognized lerp_dict(..., {keys=})')

    res = {}
    for k in out_keys:
        if k not in b:
            res[k] = a[k]
        elif k not in a:
            res[k] = b[k]
        elif recurse and isinstance(a[k], dict):
            res[k] = interp_dict(a[k], b[k], t, keys=keys, fill=fill, recurse=recurse, lerp=lerp)
        elif isinstance(a[k], numbers.Number) or isinstance(a[k], np.ndarray):
            res[k] = lerp(a[k], b[k], t)
        else:
            raise TypeError(f'interp_dict could not handle {type(a[k])=}')

    return res

def polar_skeleton(rads, eulers):

    assert len(rads.shape) == 1

    # if too few eulers are provided, we will assume the user only cares about the latter angles
    # IE 1 col = yaws, 2 col = pitches + yaws, 3 col = roll + pitches + yaws
    eulers = eulers.reshape(len(eulers), -1)
    eulers = np.deg2rad(eulers)
    if eulers.shape[1] < 3:
        zeros = np.zeros_like(eulers[:, [0]])
        eulers = np.concatenate([zeros] * (3 - eulers.shape[-1]) + [eulers], axis=-1)

    pos = Vector((0, 0, 0))
    rot = Quaternion()

    positions = [list(pos)]
    for r, euler in zip(rads, eulers):
        rot = Euler(euler).to_quaternion() @ rot
        pos += rot @ Vector((r, 0, 0))
        positions.append(list(pos))

    positions = np.array(positions)

    return positions

def offset_center(obj, x=True, z=True):
    
    # find all bbox corners
    vs = []
    for ob in butil.iter_object_tree(obj):
        for corner in ob.bound_box:
            vs.append(ob.matrix_world @ mathutils.Vector(corner))
    vs = np.array(vs)

    # offset to center x and align z to floor
    xoff = -(vs[:, 0].max() - vs[:, 0].min()) / 2 if x else 0
    zoff = -vs[:, -1].min() if z else 0
    offset = mathutils.Vector((xoff, 0, zoff))
    for ob in obj.children:
        ob.location += offset   