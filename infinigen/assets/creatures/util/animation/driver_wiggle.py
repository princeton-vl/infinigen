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

logger = logging.getLogger(__name__)

def sinusoid_driver(driver, mag, freq, off):
    driver.expression = f'{mag:.4f}*sin(({freq:.4f}*frame+{off:.4f})/(2*pi))'

def remove_ik_constraints(bones):
    for b in bones:
        for c in b.constraints:
            logger.debug(f'Removing {c.name} from {b.name=}')
            if hasattr(c, 'target'):
                butil.delete(c.target)
            b.constraints.remove(c)

def animate_wiggle_bones(arma, bones, mag_deg, freq, off=0, wavelength=1, remove_iks=True):

    '''
    mag_deg = sum of magnitudes across al bones
    freq = flaps per second
    off = global time offset
    wavelength = how many flaps fit into one creature

    '''

    logger.debug(f'animate_wiggle_bones on {len(bones)=} {mag_deg=}')

    # remove any iks, we will be overriding them
    if remove_iks:
        remove_ik_constraints(bones)
        
    mag = np.deg2rad(mag_deg) / len(bones)
    frame_period = int(bpy.context.scene.render.fps / freq)

    for i, b in enumerate(bones):
        b_off = -(off + i / len(bones)) * frame_period / wavelength
        b.rotation_mode = 'XYZ'
        sinusoid_driver(b.driver_add('rotation_euler')[0].driver, mag, freq, b_off)