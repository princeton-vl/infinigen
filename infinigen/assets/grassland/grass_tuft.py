# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

import numpy as np
from numpy.random import uniform, normal

from infinigen.assets.creatures.util.geometry.curve import Curve
from infinigen.core.util.blender import deep_clone_obj

from infinigen.assets.materials import grass_blade_texture

from infinigen.core.placement.factory import AssetFactory

from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class GrassTuftFactory(AssetFactory):

    def __init__(self, seed):

        super(GrassTuftFactory, self).__init__(seed)

        self.n_seg = 4
        self.length_mean = uniform(0.05, 0.15)
        self.length_std = self.length_mean * uniform(0.2, 0.5)

        self.curl_mean = uniform(10, 70)
        self.curl_std = self.curl_mean * np.clip(normal(0.3, 0.1), 0.01, 0.6)
        self.curl_power = normal(1.2, 0.3)

        self.blade_width_pct_mean = uniform(0.01, 0.03)
        self.blade_width_var = uniform(0, 0.05)

        self.taper_var = uniform(0, 0.1)
        self.taper_y = np.linspace(1, 0, self.n_seg) * normal(1, self.taper_var, self.n_seg) 
        self.taper_x = np.linspace(0, 1, self.n_seg)
        self.taper_points = np.stack([self.taper_x, self.taper_y], axis=-1)

        self.base_spread = uniform(0, self.length_mean/4)
        self.base_angle_var = uniform(0, 15)

    def create_asset(self, **params) -> bpy.types.Object:
        
        n_blades = np.random.randint(30, 60)
        
        blade_lengths = normal(self.length_mean, self.length_std, (n_blades, 1))
        seg_lens = (blade_lengths / self.n_seg)
        
        seg_curls = normal(self.curl_mean, self.curl_std, (n_blades, self.n_seg)) 
        seg_curls *= np.power(np.linspace(0, 1, self.n_seg).reshape(1, self.n_seg), self.curl_power)
        seg_curls = np.deg2rad(seg_curls)

        point_rads = np.arange(self.n_seg).reshape(1, self.n_seg) * seg_lens
        point_angles = np.cumsum(seg_curls, axis=-1)
        point_angles -= point_angles[:, [0]]

        points = np.empty((n_blades, self.n_seg, 2))
        points[..., 0] = np.cumsum(point_rads * np.cos(point_angles), axis=-1)
        points[..., 1] = np.cumsum(point_rads * np.sin(point_angles), axis=-1)

        taper = Curve(self.taper_points).to_curve_obj()

        widths = blade_lengths.reshape(-1) * normal(self.blade_width_pct_mean, self.blade_width_var, n_blades)
        objs = []
        for i in range(n_blades):
            obj = Curve(points[i], taper=taper).to_curve_obj(name=f'_blade_{i}', extrude=widths[i], resu=2)
            objs.append(obj)

        with butil.SelectObjects(objs):
            bpy.ops.object.convert(target='MESH')
        butil.delete(taper)

        # Randomly pose and arrange the blades in a circle-ish cluster
        base_angles = uniform(0, 2 * np.pi, n_blades)
        base_rads = uniform(0, self.base_spread, n_blades)
        facing_offsets = np.rad2deg(normal(0, self.base_angle_var, n_blades))
        for a, r, off, obj in zip(base_angles, base_rads, facing_offsets, objs):
            obj.location = (-r * np.cos(a), r * np.sin(a), -0.05 * self.length_mean)
            obj.rotation_euler = (np.pi/2, -np.pi/2, -a + off)

        with butil.SelectObjects(objs):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        with butil.SelectObjects(objs):
            bpy.ops.object.join()
            bpy.ops.object.shade_flat()
            parent = objs[0]

        tag_object(parent, 'grass_tuft')
        
        return parent

    def finalize_assets(self, assets):
        grass_blade_texture.apply(assets)