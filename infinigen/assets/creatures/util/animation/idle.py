# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import logging

import bpy
import bpy_types
import mathutils

import numpy as np
from numpy.random import uniform as U, normal as N

import pdb

from infinigen.assets.creatures.util import creature, creature_util as cutil
from infinigen.core.util.math import clip_gaussian, randomspacing, lerp
from infinigen.core.util import blender as butil

def compute_ik_length_height(targets):
    bounds = []
    for i in range(3):
        vmin = min(t.matrix_world.translation[i] for t in targets)
        vmax = max(t.matrix_world.translation[i] for t in targets)
        bounds.append([vmin, vmax])
    return np.array(bounds)

def snap_iks_to_floor(targets, floor_bvh, minweight=0.7):

    assert floor_bvh is not None 

    bpy.context.view_layer.update()

    get_targets = lambda k: [t for t in targets if k in t.name]

    bounds = compute_ik_length_height(targets)

    def find_floor_offset(t):
        ray_origin = mathutils.Vector((t.matrix_world.translation.x, t.matrix_world.translation.y, bounds[2, 1]))
        location, normal, index, dist = floor_bvh.ray_cast(ray_origin, mathutils.Vector((0, 0, -1)))
        if location is None:
            return None
        return location - t.matrix_world.translation
    
    feet = get_targets('foot')
    feet_offsets = [find_floor_offset(f) for f in feet]

    if any(off is None for off in feet_offsets):
        logging.warning(f'snap_iks_to_floor found {feet_offsets=}, aborting snap operation')
        return

    # dont allow the pose diff to be too large (ie, prevent weird behavior at cliffs)
    for i, o in enumerate(feet_offsets):
        if o.length > bounds[2, 1] - bounds[2, 0]:
            logging.warning(f'snap_iks_to_floor ignoring too-long offset {o.length=}')
            feet_offsets[i] = mathutils.Vector()

    for f, fo, in zip(feet, feet_offsets):
        f.location += fo        

    hips = get_targets('body')
    if len(feet) == len(hips) * 2:

        # hips seem coupled to pairs of feet, take that into consideration
        # TODO: Restructure to make detecting this more robust

        hip_offsets = []
        for i in range(len(feet) // 2):
            o1, o2 = feet_offsets[2*i], feet_offsets[2*i + 1]
            hip_off = minweight * min(o1, o2) + (1 - minweight) * max(o1, o2)
            hip_offsets.append(hip_off)

        for h, ho in zip(hips, hip_offsets):
            h.location += ho
        
        for o in get_targets('head'): # front-associated
            o.location += hip_offsets[-1]
        for o in get_targets('tail'): # back associated
            o.location += hip_offsets[0]

    else:
        logging.warning(f'Couldnt establish feet-hip mapping')
        off = mathutils.Vector(np.array(feet_offsets).mean(axis=0))
        for o in targets:
            if o in feet:
                continue
            o.location += off

def idle_body_noise_drivers(targets, foot_motion_chance=0.2, head_benddown=1.0, body_mag=1.0, wing_mag=1.0):

    # all magnitudes are determined as multiples of the creatures overall length/height/width
    bounds = compute_ik_length_height(targets)
    ls = bounds[:, 1] - bounds[:, 0]

    # scalars for the whole creature
    freq_scalar = N(1, 0.15)
    mag_scalar = N(1, 0.15)

    def add_noise(t, k, axis, mag, freq, off=0, mode='noise', seeds=None):
        d = t.driver_add(k, axis)
        p = getattr(t, k)[axis]

        if k == 'location':
            mag *= ls[axis]
        
        freq = freq / bpy.context.scene.render.fps

        freq *= freq_scalar
        mag *= mag_scalar
        
        if mode == 'noise':
            s1, s2 = seeds if seeds is not None else U(0, 1000, 2) # random offsets as 'seeds'
            varying = f'noise.noise(({freq:.6f}*frame, {s1:.2f}, {s2:.2f}))'
        elif mode == 'sin':
            varying = f'sin({freq:6f}*frame*2*pi)'
        else:
            raise ValueError(mode)

        d.driver.expression = f'{p:.4f}+{mag:.4f}*({off:.4f}+{varying})'
    
    get_targets = lambda k: [t for t in targets if k in t.name]

    for i, t in enumerate(get_targets('body')):
        add_noise(t, 'location', 0, mag=body_mag*0.025*N(1, 0.2), freq=0.25*N(1, 0.2))
        if i != 0:
            add_noise(t, 'location', 2, mag=body_mag*0.015*N(1, 0.2), freq=0.5*N(1, 0.2), mode='sin')

    for t in get_targets('foot'):
        if U() < foot_motion_chance:
            add_noise(t, 'location', 0, mag=0.07*N(1, 0.1), freq=U(0.2, 0.7))
            add_noise(t, 'location', 2, mag=0.04*N(1, 0.1), freq=U(0.2, 0.7))

    for t in get_targets('head'):
        headfreq = 0.4
        add_noise(t, 'location', 0, mag=0.07*N(1, 0.1), freq=headfreq, off=-0.5*head_benddown)
        add_noise(t, 'location', 1, mag=0.03*N(1, 0.1), freq=headfreq)
        add_noise(t, 'location', 2, mag=0.2*N(1, 0.1), freq=headfreq/2, off=-0.7*head_benddown)
        #add_noise(t, 'rotation_euler', 0, mag=0.4*N(1, 0.1), freq=U(0.1, 0.4))
        #add_noise(t, 'rotation_euler', 1, mag=0.4*N(1, 0.1), freq=U(0.1, 0.4))

    seeds = U(0, 1000, 2) # synchronize wing motion a little bit
    for t in get_targets('wingtip'):
        add_noise(t, 'location', 0, mag=wing_mag*0.1*N(1, 0.1), freq=U(0.6, 4), seeds=seeds+N(0, 0.2, 2))
        add_noise(t, 'location', 2, mag=wing_mag*0.2*N(1, 0.1), freq=U(0.6, 4), seeds=seeds+N(0, 0.2, 2))

    for t in get_targets('tail'):
        for i in range(3):
            add_noise(t, 'location', 0, mag=0.07*N(1, 0.1), freq=headfreq, off=-0.5)
            
def head_look_around(targets):
    pass