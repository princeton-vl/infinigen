# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Zeyu Ma, Kaiyu Yang, Lingjie Mei


import bpy
import math
import numpy as np
import gin
from infinigen.core.util.random import random_general as rg
from numpy.random import uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.math import clip_gaussian

@gin.configurable
def nishita_lighting(
    nw,
    cam,
    dust_density=("clip_gaussian", 1, 1, 0.1, 2),
    air_density=("clip_gaussian", 1, 0.2, 0.7, 1.3),
    strength=("uniform", 0.18, 0.22),
    sun_intensity=("uniform", 0.8, 1),
    sun_elevation=("spherical_sample", 10, None),
    dynamic=False,
    rising_angle=90,
    camera_based_rotation=None,
):
    sky_texture = nw.new_node(Nodes.SkyTexture)
    sky_texture.sky_type = "NISHITA"
    sky_texture.sun_size = np.deg2rad(clip_gaussian(0.5, 0.3, 0.25, 5))
    sky_texture.sun_intensity = rg(sun_intensity)
    sky_texture.sun_elevation = np.radians(rg(sun_elevation))
    if camera_based_rotation is None:
        sky_texture.sun_rotation = np.random.uniform(0, 2 * math.pi)
    else:
        sky_texture.sun_rotation = 2 * math.pi - cam.parent.rotation_euler[2] + np.radians(camera_based_rotation)
    if dynamic:
        sky_texture.sun_rotation += (sky_texture.sun_elevation + np.radians(8)) / 2 * np.arctan(np.radians(rising_angle))
        sky_texture.keyframe_insert(data_path="sun_rotation", frame=bpy.context.scene.frame_end)
        sky_texture.sun_rotation -= (sky_texture.sun_elevation + np.radians(8)) * np.arctan(np.radians(rising_angle))
        sky_texture.keyframe_insert(data_path="sun_rotation", frame=bpy.context.scene.frame_start)

        sky_texture.keyframe_insert(data_path="sun_elevation", frame=bpy.context.scene.frame_end)
        sky_texture.sun_elevation = -np.radians(8)
        sky_texture.keyframe_insert(data_path="sun_elevation", frame=bpy.context.scene.frame_start)
        sky_texture.sun_elevation = -np.radians(5)
        sky_texture.keyframe_insert(data_path="sun_elevation", frame=bpy.context.scene.frame_start + 10)

    sky_texture.altitude = clip_gaussian(100, 400, 0, 2000)
    sky_texture.air_density =rg(air_density)
    sky_texture.dust_density = rg(dust_density)
    sky_texture.ozone_density = clip_gaussian(1, 1, 0.1, 10)
    strength = rg(strength)
    return nw.new_node(Nodes.Background, input_kwargs={'Color': sky_texture, 'Strength': strength})

def add_lighting(cam=None):
    nw = NodeWrangler(bpy.context.scene.world.node_tree)

    if True:
        surface = nishita_lighting(nw, cam)
    else:
        # TODO more options
        surface = None

    volume = None

    nw.new_node(Nodes.WorldOutput, input_kwargs={
        'Surface': surface,
        'Volume': volume
    })

@gin.configurable
def add_camera_based_lighting(energy=("log_uniform", 200, 500), spot_size=("uniform", np.pi / 6, np.pi / 4)):
    camera = bpy.context.scene.camera
    bpy.ops.object.light_add(type='SPOT', location=camera.location, rotation=camera.rotation_euler)
    spot = bpy.context.active_object
    spot.data.energy = rg(energy)
    spot.data.spot_size = rg(spot_size)
    spot.data.spot_blend = uniform(.6, .8)
