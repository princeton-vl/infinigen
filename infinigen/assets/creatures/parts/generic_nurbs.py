# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb

import bpy
from pathlib import Path
import numpy as np
from infinigen.assets.creatures.util.creature import Part, PartFactory

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util import part_util
from infinigen.assets.creatures.util.geometry import lofting

from infinigen.core.util import blender as butil
from infinigen.core.util.logging import Suppress
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

NURBS_BASE_PATH = Path(__file__).parent/'nurbs_data'
NURBS_KEYS = [p.stem for p in NURBS_BASE_PATH.iterdir()]
def load_nurbs(name:str):
    return np.load(NURBS_BASE_PATH/(name + '.npy'))[..., :3]

def decompose_nurbs_handles(handles):

    skeleton, ts, profiles = lofting.factorize_nurbs_handles(handles)
 
    rads = np.linalg.norm(profiles, axis=2, keepdims=True).mean(axis=1, keepdims=True)
    rads = np.clip(rads, 1e-3, 1e5)
    profiles_norm = profiles / rads 

    skeleton_root = skeleton[[0]]
    dirs = np.diff(skeleton, axis=0)

    lens = np.linalg.norm(dirs, axis=-1)
    length = lens.sum()
    proportions = lens / length

    thetas = np.arctan2(dirs[:, 2], dirs[:, 0])
    thetas = np.rad2deg(thetas)
    skeleton_yoffs = dirs[:, 1] / lens

    return {
        'ts': ts,
        'rads': rads,
        'skeleton_root': skeleton_root,
        'skeleton_yoffs': skeleton_yoffs,
        'length': length,
        'proportions': proportions,
        'thetas': thetas,
        'profiles_norm': profiles_norm
    }

def recompose_nurbs_handles(params):

    lens = params['length'] * params['proportions']
    thetas = np.deg2rad(params['thetas'])
    skeleton_offs = np.stack([
        lens * np.cos(thetas),
        lens * params['skeleton_yoffs'],
        lens * np.sin(thetas)
    ], axis=-1)
    skeleton = np.concatenate([params['skeleton_root'], skeleton_offs], axis=0)
    skeleton = np.cumsum(skeleton, axis=0)
    
    handles = lofting.compute_profile_verts(
        skeleton, params['ts'], 
        params['profiles_norm'] * params['rads'], profile_as_points=True)

    return handles

class NurbsPart(PartFactory):

    def __init__(self, params=None, prefix=None, tags=None, temperature=0.3, var=1, exps=None):
        self.prefix = prefix
        self.tags = tags or []
        self.temperature = temperature
        self.var = var
        self.exps = exps
        super(NurbsPart, self).__init__(params)
        
    def sample_params(self, select=None):

        if self.prefix is None:
            # for compatibility with interp which will not init prefix but does not need sample_params
            return {} # TODO hacky, replace

        N = lambda u, v, d=1: np.random.normal(u, np.array(v) * self.var, d)

        target_keys = [k for k in NURBS_KEYS if self.prefix is None or k.startswith(self.prefix)]
        weights = part_util.random_convex_coord(target_keys, select=select, temp=self.temperature)
        if self.exps is not None:
            for k, exp in self.exps.items():
                weights[k] = weights[k] ** exp
                
        handles = sum(w * load_nurbs(k) for k, w in weights.items())
        decomp = decompose_nurbs_handles(handles)

        sz = N(1, 0.1)
        decomp['length'] *= sz * N(1, 0.1)
        decomp['rads'] *= sz * N(1, 0.1) * N(1, 0.15, decomp['rads'].shape)
        decomp['proportions'] *= N(1, 0.15)

        ang_noise = N(0, 7, decomp['thetas'].shape)
        ang_noise -= ang_noise.mean()
        decomp['thetas'] += ang_noise

        n, m, d = decomp['profiles_norm'].shape
        profile_noise = N(1, 0.07, (1, m, 1)) * N(1, 0.15, (n, m, 1))
        profile_noise[:, :m//2-1] = profile_noise[:, m//2:-1][:, ::-1] # symmetrize noise
        decomp['profiles_norm'] *= profile_noise # profiles are 0-centered so multiplication is sensible
        
        return decomp

    def make_part(self, params):
        handles = recompose_nurbs_handles(params)
        part = part_util.nurbs_to_part(handles)
        with butil.ViewportMode(part.obj, mode='EDIT'), Suppress():
            bpy.ops.mesh.select_all()
            bpy.ops.mesh.remove_doubles()
            bpy.ops.mesh.normals_make_consistent(inside=False)
        return part

class NurbsBody(NurbsPart):

    def __init__(self, *args, shoulder_ik_ts=[0.0, 0.6], n_bones=8, rig_reverse_skeleton=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shoulder_ik_ts = shoulder_ik_ts
        self.n_bones = n_bones
        self.rig_reverse_skeleton = rig_reverse_skeleton

    def make_part(self, params):
        part = super().make_part(params)
        part.joints = {
            i: Joint((0,0,0), bounds=np.array([[-30, -30, -30], [30, 30, 30]]))
            for i in np.linspace(0, 1, self.n_bones, endpoint=True)
        }
        part.iks = {
            t: IKParams(name=f'body_{i}', mode='pin' if i == 0 else 'iksolve', 
                        rotation_weight=0, target_size=0.3)
            for i, t in enumerate(self.shoulder_ik_ts)
        }
        part.settings['rig_reverse_skeleton'] = self.rig_reverse_skeleton
        tag_object(part.obj, 'body')
        return part

class NurbsHead(NurbsPart):

    def make_part(self, params):
        part = super().make_part(params)
        part.iks = {
            1.0: IKParams(name='head', rotation_weight=0.1, target_size=0.4, chain_length=1)
        }
        part.settings['rig_extras'] = True
        tag_object(part.obj, 'head')
        return part