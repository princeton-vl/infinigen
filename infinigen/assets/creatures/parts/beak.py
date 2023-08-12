# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import bpy 
import numpy as np
from math import sin, cos, pi, exp

from infinigen.assets.creatures.util.creature import PartFactory, Part
from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util import part_util
from infinigen.core.util import blender as butil

from infinigen.assets.creatures.util.geometry import nurbs as nurbs_util
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def square(x):
    return x * x

class Beak():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.hook_x = lambda x, theta: self.hook(self.hook_scale_x, self.hook_a, self.hook_b, self.hook_pos_x, self.hook_thickness_x, x, theta)
        self.hook_z = lambda x, theta: self.hook(self.hook_scale_z, self.hook_a, self.hook_b, self.hook_pos_z, self.hook_thickness_z, x, theta)
        
        self.crown_z = lambda x, theta: self.crown(self.crown_scale_z, self.crown_a, self.crown_b, self.crown_pos_z, x, theta)
        self.bump_z = lambda x, theta: self.bump(self.bump_scale_z, x, self.bump_l, self.bump_r) * max(sin(theta), 0)
    
    def cx(self, x):
        return x

    def cy(self, x):
        return 1 - exp(self.cy_a * (x - 1))

    def cz(self, x):
        return 1 - (x ** self.cz_a)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def exp(self, a, b, x):
        return a * exp(b * x)

    def hook(self, scale, a, b, p, t, x, theta):
        return scale * self.exp(a, b, x - p - (1 - x) * t * sin(theta))
        
    def bump(self, scale, x, l, r):
        if x < l or x > r:
            return 0
        x = (x - l) / (r - l) * pi
        return scale * sin(x)
        
    def crown(self, scale, a, b, p, x, theta):
        return scale * self.exp(a, b, p - x) * max(sin(theta), 0)
    
    def dx(self, x, theta):
        hook = self.hook_x(x, theta)
        sharp = self.sharpness * max(x - 0.95, 0)
        return hook + sharp

    def dy(self, x):
        return 0

    def dz(self, x, theta):
        hook = self.hook_z(x, theta)
        crown = self.crown_z(x, theta)
        bump = self.bump_z(x, theta)
        return hook + crown + bump
        
    def generate(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = self.sx * self.cx(p) + self.dx(p, theta)
                ctrls[i][j][1] = self.sy * self.cy(p) * self.r * cos(theta) + self.dy(p)
                ctrls[i][j][2] = self.reverse * (self.sz * self.cz(p) * self.r * max(sin(theta), 0) + self.dz(p, theta))

        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.02)


class BirdBeak(PartFactory):

    param_templates = {}
    tags = ['head_detail', 'rigid']
    unit_scale = (0.5, 0.5, 0.5)

    def sample_params(self, select=None, var=1):
        weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        params = part_util.rdict_comb(self.param_templates, weights)
        # params = np.random.choice(list(self.param_templates.values()))
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)
        # add additional noise to params
        for key in params['upper']:
            if key in params['range']:
                l, r = params['range'][key]
                noise = N(0, 0.05 * (r - l))
                params['upper'][key] += noise
                params['lower'][key] += noise
                params['upper'][key] = max(min(params['upper'][key], r), l)
                params['lower'][key] = max(min(params['lower'][key], r), l)
        params['lower']['sx'] = min(params['lower']['sx'], params['upper']['sx'] * (params['upper']['hook_pos_x'] - params['upper']['hook_thickness_x'] / 2))
        
        return params
    
    def rescale(self, params, scale):
        params['sx'] *= scale
        params['sy'] *= scale
        params['sz'] *= scale
        return params

    def make_part(self, params):
        
        obj = butil.spawn_vert('beak_parent_temp')
        upper = Beak(**params['upper']).generate()
        upper.parent = obj
        upper.name = 'BeakUpper'

        lower = Beak(**params['lower']).generate()
        lower.parent = obj
        lower.name = 'BeakLower'

        upper.scale = self.unit_scale
        lower.scale = self.unit_scale
        butil.apply_transform([upper, lower], scale=True)

        part = Part(skeleton=np.zeros((1, 3)), obj=obj, joints={}, iks={})
        tag_object(part.obj, 'bird_beak')
        
        return part


class FlyingBirdBeak(BirdBeak):
    def sample_params(self, select='normal', var=1):
        return super().sample_params(select=select)

    def make_part(self, params):
        obj = butil.spawn_vert('beak_parent_temp')
        params['upper'] = self.rescale(params['upper'], 0.4)
        params['lower'] = self.rescale(params['lower'], 0.4)
        upper = Beak(**params['upper']).generate()
        upper.parent = obj
        upper.name = 'BeakUpper'

        lower = Beak(**params['lower']).generate()
        lower.parent = obj
        lower.name = 'BeakLower'

        upper.scale = self.unit_scale
        lower.scale = self.unit_scale
        butil.apply_transform([upper, lower], scale=True)

        return Part(skeleton=np.zeros((1, 3)), obj=obj, joints={}, iks={})



default_beak = {
    'n': 20,
    'm': 20,
    'r': 1.0, 
    'sx': 1.0, 
    'sy': 1.0, 
    'sz': 1.0,
    'cy_a': 1.0,
    'cz_a': 2.0,
    'reverse': 1,
    'hook_a': 0.1,
    'hook_b': 5.0,
    'hook_scale_x': 0.0,
    'hook_pos_x': 0.0,
    'hook_thickness_x': 0.0,
    'hook_scale_z': 0.0,
    'hook_pos_z': 0.0,
    'hook_thickness_z': 0.0,
    'crown_scale_z': 0.0,
    'crown_a': 0.5,
    'crown_b': 0.5, 
    'crown_pos_z': 0.5,
    'bump_scale_z': 0.0,
    'bump_l': 0.5,
    'bump_r': 0.5,
    'sharpness': 0.0,
}

scales = {
    'r': [0.3, 1], 
    'sx': [0.2, 1], 
    'sy': [0.2, 1], 
    'sz': [0.2, 1],
    'cy_a': [1, 10],
    'cz_a': [1, 5],
    'hook_a': [0.1, 0.8],
    'hook_b': [1, 5],
    'hook_scale_x': [-0.5, 0.5],
    'hook_pos_x': [0.5, 1],
    'hook_thickness_x': [0, 0.5],
    'hook_scale_z': [-0.5, 0.5],
    'hook_pos_z': [0.5, 1],
    'hook_thickness_z': [0, 0.5],
    'crown_scale_z': [0, 0.3],
    'crown_a': [0.1, 0.8],
    'crown_b': [0, 2], 
    'crown_pos_z': [0, 0.5],
    'bump_scale_z': [0, 0.03],
    'bump_l': [0, 0.4],
    'bump_r': [0.6, 1],
    'sharpness': [-0.5, 0.5],
}
for k, v in scales.items():
    scales[k] = np.array(v)

eagle_upper = default_beak | {
    'r': 0.4, 
    'sx': 0.8, 
    'sy': 0.4, 
    'sz': 1.0,
    'hook_a': 0.1,
    'hook_b': 5.0,
    'hook_scale_x': -1.0,
    'hook_pos_x': 0.72,
    'hook_thickness_x': 0.35,
    'hook_scale_z': -0.8,
    'hook_pos_z': 0.7,
    'hook_thickness_z': 0.0,
}

eagle_lower = default_beak | {
    'r': 0.4, 
    'sx': 0.4, 
    'sy': 0.4, 
    'sz': 0.2,
    'reverse': -1,
    'hook_a': 0.1,
    'hook_b': 5.0,
    'hook_scale_x': 0.0,
    'hook_pos_x': 0.72,
    'hook_thickness_x': 0.35,
    'hook_scale_z': 0.1,
    'hook_pos_z': 0.6,
    'hook_thickness_z': -0.2,
}

normal_upper = default_beak | {
    'r': 0.4, 
    'sx': 0.7,
    'sy': 0.3, 
    'sz': 0.5,
    'hook_a': 0.1,
    'hook_b': 2.0,
    'hook_scale_x': 0.0,
    'hook_pos_x': 0.72,
    'hook_thickness_x': 0.35,
    'hook_scale_z': -0.8,
    'hook_pos_z': 0.7,
    'hook_thickness_z': 0.0,
}

normal_lower = default_beak | {
    'r': 0.4, 
    'sx': 0.7,
    'sy': 0.3, 
    'sz': 0.3,
    'reverse': -1,
    'hook_a': 0.1,
    'hook_b': 2.0,
    'hook_scale_x': 0.0,
    'hook_pos_x': 0.72,
    'hook_thickness_x': 0.35,
    'hook_scale_z': 0.8,
    'hook_pos_z': 0.7,
    'hook_thickness_z': 0.0,
}

duck_upper = default_beak | {
    'n': 50,
    'r': 0.4, 
    'sx': 1.0, 
    'sy': 0.4, 
    'sz': 0.5,
    'cy_a': 10.0,
    'hook_a': 0.1,
    'hook_b': 2.0,
    'hook_scale_x': -1.5,
    'hook_pos_x': 0.9,
    'hook_thickness_x': 0.0,
    'hook_scale_z': 0.4,
    'hook_pos_z': 0.6,
    'hook_thickness_z': 0.2,
    'crown_scale_z': 0.3,
    'crown_a': 0.1,
    'crown_b': 5.0, 
    'crown_pos_z': 0.3,
    'bump_scale_z': 0.02,
    'bump_l': 0.4,
    'bump_r': 1.0,
    'sharpness': -0.5
}

duck_lower = default_beak | {
    'n': 50,
    'r': 0.4, 
    'sx': 0.97, 
    'sy': 0.4, 
    'sz': 0.1,
    'cy_a': 10.0,
    'reverse': -1,
    'hook_a': 0.1,
    'hook_b': 2.0,
    'hook_scale_x': -1.5,
    'hook_pos_x': 0.9,
    'hook_thickness_x': 0.0,
    'hook_scale_z': -0.4,
    'hook_pos_z': 0.6,
    'hook_thickness_z': 0.0,
    'crown_scale_z': 0.1,
    'crown_a': 0.1,
    'crown_b': 5.0, 
    'crown_pos_z': 0.3,
    'bump_scale_z': 0.03,
    'bump_l': 0.3,
    'bump_r': 1.0,
    'sharpness': -0.5
}   

short_upper = default_beak | {
    'r': 0.4, 
    'sx': 0.25, 
    'sy': 0.3, 
    'sz': 0.3,
    'hook_a': 0.1,
    'hook_b': 2.0,
    'hook_scale_x': -0.5,
    'hook_pos_x': 0.8,
    'hook_thickness_x': 0.35,
    'hook_scale_z': -0.15,
    'hook_pos_z': 0.7,
    'hook_thickness_z': 0.0,
}
short_lower = default_beak | {
    'r': 0.4, 
    'sx': 0.25, 
    'sy': 0.3, 
    'sz': 0.3,
    'cy_a': 1.0,
    'cz_a': 1.1,
    'reverse': -1,
    'hook_a': 0.1,
    'hook_b': 2.0,
    'hook_scale_x': -0.5,
    'hook_pos_x': 0.8,
    'hook_thickness_x': 0.35,
    'hook_scale_z': 0.15,
    'hook_pos_z': 0.7,
    'hook_thickness_z': 0.0,
}  

BirdBeak.param_templates['normal'] = {'upper': normal_upper, 'lower': normal_lower, 'range': scales}
BirdBeak.param_templates['duck'] = {'upper': duck_upper, 'lower': duck_lower, 'range': scales}
BirdBeak.param_templates['eagle'] = {'upper': eagle_upper, 'lower': eagle_lower, 'range': scales}
BirdBeak.param_templates['short'] = {'upper': short_upper, 'lower': short_lower, 'range': scales}
