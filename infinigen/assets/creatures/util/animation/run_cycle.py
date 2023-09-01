# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import logging

import re
import bpy
import bpy_types
import mathutils

import numpy as np
from numpy.random import uniform as U, normal as N
from math import pi

import pdb

from infinigen.assets.creatures.util import creature, creature_util as cutil
from infinigen.core.util.math import clip_gaussian, randomspacing, lerp
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)

def foot_path(length, height, upturn, down_stroke, curve_resolution=8):

    curve = bpy.data.curves.new('foot_path', 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = curve_resolution
    curve.render_resolution_u = curve_resolution
    obj = bpy.data.objects.new('foot_path', curve)
    bpy.context.scene.collection.objects.link(obj)

    s = curve.splines.new(type='NURBS')
    s.use_cyclic_u = True    
    s.use_bezier_u = False    
    s.points.add(4 - len(s.points))
    s.points[0].co = (0,          0,  -down_stroke,          1)
    s.points[1].co = (-length/2,  0,  -down_stroke + upturn, 1)
    s.points[2].co = (0,          0,  height,                1)
    s.points[3].co = (length/2,   0,  -down_stroke + upturn, 1)

    curve.splines[0].order_u = 3

    return obj

def body_path(length, height, upturn, down_stroke, curve_resolution=8):

    curve = bpy.data.curves.new('body_path', 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = curve_resolution
    curve.render_resolution_u = curve_resolution
    obj = bpy.data.objects.new('body_path', curve)
    bpy.context.scene.collection.objects.link(obj)

    s = curve.splines.new(type='NURBS')
    s.use_cyclic_u = True    
    s.use_bezier_u = False    
    s.points.add(4 - len(s.points))
    s.points[0].co = (0,          -down_stroke,  0,         1)
    s.points[1].co = (-length/2,  -down_stroke + upturn, 0, 1)
    s.points[2].co = (0,          height,              0,   1)
    s.points[3].co = (length/2,   -down_stroke + upturn, 0, 1)

    curve.splines[0].order_u = 3

    return obj

def follow_path(target, path, duration: int, offset: float = 0, reset_rot=True, **kwargs):
    
    target.location = (0, 0, 0)
    if reset_rot:
        target.rotation_euler = (0, 0, 0)
    c = butil.constrain_object(target, 'FOLLOW_PATH', 
                               target=path, offset=offset, **kwargs)
    
    path.data.use_path = True
    path.data.path_duration = duration
        
def follow_gait_path(targets, path_dims, period, offset, spread):

    offsets = offset + np.linspace(-spread/2, spread/2, len(targets), endpoint=True)
    offsets *= period

    paths = []
    for target, offset in zip(targets, offsets):
        path = foot_path(*path_dims)
        path.location = target.location
        target.parent = None
        follow_path(target, path, duration=period, offset=offset, reset_rot=False)
        paths.append(path)

    return paths

def follow_body_path(targets, path_dims, period, offset, spread):

    offsets = offset + np.linspace(-spread/2, spread/2, len(targets), endpoint=True)
    offsets *= period

    paths = []
    for target, offset in zip(targets, offsets):
        path = body_path(*path_dims)
        path.location = target.location
        target.parent = None
        follow_path(target, path, period=period, offset=offset)
        paths.append(path)

    return paths

def animate_run(root, arma, targets, steps_per_sec=1, body=True, motion=True, squash_gait_pct=0.1):

    '''
    Animate creature by moving its IK targets
    '''
    
    assert arma.type == 'ARMATURE'

    stride_length = 0.7 * clip_gaussian(0.4, 0.3, 0.3, 0.6) * arma.dimensions.x
    spread = clip_gaussian(0.15, 0.1, 0, 0.5)
    stride_height = U(0.15, 0.4)
    body_height = stride_height * clip_gaussian(0.6, 0.4, 0.3, 1.2)
    
    base_offset = U(0, 1)
    offset = U(0.5, 0.7)

    frame_period = int(bpy.context.scene.render.fps / steps_per_sec)

    get_targets = lambda k: [t for t in targets if k in t.name]

    feet_targets = get_targets('foot')

    foot_paths = []    
    foot_paths += follow_gait_path(targets=feet_targets[:2], period=frame_period,
        path_dims=(stride_length, stride_height, 0.0, 0.0), offset=base_offset, spread=spread)
    foot_paths += follow_gait_path(targets=feet_targets[2:], period=frame_period,
        path_dims=(stride_length, stride_height, 0.0, 0.0), offset=base_offset+offset, spread=spread)

    for p in foot_paths:
        p.parent = root

    feet_targets = get_targets('knee')
    knee_paths = []    
    knee_paths += follow_gait_path(targets=feet_targets[:2], period=frame_period,
        path_dims=(0.1 * stride_length, 0.1 * stride_height, 0.0, 0.0), offset=base_offset, spread=spread)
    knee_paths += follow_gait_path(targets=feet_targets[2:], period=frame_period,
        path_dims=(0.1 * stride_length, 0.1 * stride_height, 0.0, 0.0), offset=base_offset+offset, spread=spread)
    for p in knee_paths:
        p.location.z = -0.1
        p.parent = root
    
    body_paths = []
    if body:
        body_paths += follow_gait_path(targets=get_targets('body_0'), period=frame_period,
            path_dims=(0, body_height, 0.0, 0.0), offset=base_offset+1-offset/2, spread=0)
        body_paths += follow_gait_path(targets=get_targets('body_1'), period=frame_period,
            path_dims=(0, body_height, 0.0, 0.0), offset=base_offset+offset/2, spread=0)
        #body_paths += animate_feet(targets=get_targets('tail'), period=frame_period,
        #    path_dims=(0.1, 0.4, 0.0, 0.0), offset=0, spread=0)
        body_paths += follow_gait_path(targets=get_targets('head'), period=frame_period, 
            path_dims=(0, body_height*0.5*N(1, 0.05), 0.0, 0.0), offset=0, spread=0)

    flap_height = U(0.3, 1)
    flap_speed_mult = 1 #uniform(0.7, 2)
    body_paths += follow_gait_path(targets=get_targets('wingtip'), period=int(frame_period/flap_speed_mult),
        path_dims=(0, flap_height, 0.0, flap_height), offset=base_offset+offset, spread=0)
    
    for p in body_paths:
        if len(foot_paths):
            p.location.z = (1 - squash_gait_pct) * p.location.z + squash_gait_pct * foot_paths[0].location.z
        p.parent = root

    all_paths = foot_paths + knee_paths + body_paths
    if motion:
        for p in all_paths:
           p.data.driver_add('eval_time').driver.expression = 'frame'

    return all_paths

def animate_wiggle_body_iks(root, arma, targets, compression_ratio=1.01, cycles_per_bodylen=1.5, fix_head=True):

    assert arma.type == 'ARMATURE'

    logger.info('Starting animate_wiggle')

    steps_per_sec = clip_gaussian(1.5, 0.5, 0.1, 3)

    width = 1.1 * clip_gaussian(0.15, 0.05, 0.15, 0.3) * arma.dimensions.x
    start_percent = U(0.1, 0.5)

    frame_period = int(bpy.context.scene.render.fps / steps_per_sec)

    body_paths = []
    targets = [t for t in targets if 'body_' in t.name]
    for i, t in enumerate(targets):
        offset = cycles_per_bodylen * (i/len(targets)) * frame_period
        w = lerp(start_percent * width, width, 1 - i/len(targets))
        dims = (w, 0, 0.0, 0.0)
        body_paths += follow_gait_path(targets=[t], path_dims=dims, period=frame_period,
                                       offset=offset, spread=0)

    bstart = body_paths[0].location.x 

    for p in body_paths:
        p.rotation_euler.z += np.pi / 2
        p.location.x = (p.location.x - bstart) * compression_ratio + bstart
        p.parent = root
        p.data.driver_add('eval_time').expression = 'frame'

    return body_paths

def sinusoid_driver(driver, mag, freq, off):
    driver.expression = f'{mag:.4f}*sin(({freq:.4f}*frame+{off:.4f})/(2*pi))'

def cosusoid_driver(driver, mag, freq, off):
    driver.expression = f'{mag:.4f}*cos(({freq:.4f}*frame+{off:.4f})/(2*pi))'

def animate_wiggle_bones(arma, bones, mag_deg, freq, off=0, wavelength=1, remove_iks=True, fixed_head=True):

    '''
    mag_deg = sum of magnitudes across al bones
    freq = flaps per second
    off = global time offset
    wavelength = how many flaps fit into one creature

    '''

    # remove any iks, we will be overriding them
    if remove_iks:
        for b in bones:
            for c in b.constraints:
                if hasattr(c, 'target'):
                    butil.delete(c.target)
                b.constraints.remove(c)

    mag = np.deg2rad(mag_deg) / len(bones)
    frame_period = int(bpy.context.scene.render.fps / freq)
    print('freq:', freq)

    for i, b in enumerate(bones):
        b_off = -(off + i / len(bones)) * frame_period / wavelength
        b.rotation_mode = 'XYZ'
        sinusoid_driver(b.driver_add('rotation_euler')[0].driver, mag, freq, b_off)
        if not fixed_head and i == 0: # move head
            cosusoid_driver(b.driver_add('location')[2].driver, mag / (freq / (2 * pi)), freq, b_off)
            # sinusoid_driver(b.driver_add('rotation_euler')[0].driver, -mag, freq, b_off)

def animate_running_front_leg(arma, bones, mag_deg, freq, off=0, wavelength=1, remove_iks=True, fixed_head=True):

    '''
    mag_deg = sum of magnitudes across al bones
    freq = flaps per second
    off = global time offset
    wavelength = how many flaps fit into one creature
    '''

    # remove any iks, we will be overriding them
    if remove_iks:
        for b in bones:
            for c in b.constraints:
                if hasattr(c, 'target'):
                    butil.delete(c.target)
                b.constraints.remove(c)

    mag = np.deg2rad(mag_deg) / len(bones)
    frame_period = int(bpy.context.scene.render.fps / freq)

    def number_finder(s):
        rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        return int(rr[0])

    left = 1000
    right = -1000
    for b in bones:
        rr = number_finder(b.name)
        left = min(left, rr)
        right = min(right, rr)

    for i, b in enumerate(bones):
        rr = number_finder(b.name)
        b_off = -(off + (1 / 2 * (rr == right))) * frame_period / wavelength
        b.rotation_mode = 'XYZ'
        sinusoid_driver(b.driver_add('rotation_euler')[0].driver, mag, freq, b_off)
        sinusoid_driver(b.driver_add('rotation_euler')[2].driver, 3 * mag, freq, b_off)
        b.driver_add('location')[2].driver.expression = '-0.1'
        # sinusoid_driver(b.driver_add('location')[0].driver, 5 * mag, freq, b_off)
        # sinusoid_driver(b.driver_add('location')[2].driver, 0.1 * mag, freq, b_off)

def animate_running_back_leg(arma, bones, mag_deg, freq, off=0, wavelength=1, remove_iks=True, fixed_head=True):

    '''
    mag_deg = sum of magnitudes across al bones
    freq = flaps per second
    off = global time offset
    wavelength = how many flaps fit into one creature
    '''

    # remove any iks, we will be overriding them
    if remove_iks:
        for b in bones:
            for c in b.constraints:
                if hasattr(c, 'target'):
                    butil.delete(c.target)
                b.constraints.remove(c)

    mag = np.deg2rad(mag_deg) / len(bones)
    frame_period = int(bpy.context.scene.render.fps / freq)

    def number_finder(s):
        rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        return int(rr[0])

    left = 1000
    right = -1000
    for b in bones:
        rr = number_finder(b.name)
        left = min(left, rr)
        right = min(right, rr)

    for i, b in enumerate(bones):
        rr = number_finder(b.name)
        b_off = -(off + (1 / 2 * (rr == right))) * frame_period / wavelength
        b.rotation_mode = 'XYZ'
        sinusoid_driver(b.driver_add('rotation_euler')[0].driver, 3 * mag, freq, b_off)
        # sinusoid_driver(b.driver_add('rotation_euler')[1].driver, 3 * mag, freq, b_off)
        b.driver_add('location')[2].driver.expression = '-0.1'
        # sinusoid_driver(b.driver_add('location')[0].driver, 5 * mag, freq, b_off)
        # sinusoid_driver(b.driver_add('location')[2].driver, 0.1 * mag, freq, b_off)

            


    