# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import math
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.corals.base import BaseCoralFactory
from infinigen.assets.corals.tentacles import make_radius_points_fn
from infinigen.assets.utils.decorate import separate_loose
from infinigen.assets.utils.object import mesh2obj, data2mesh
from infinigen.assets.utils.nodegroup import geo_radius
import infinigen.core.util.blender as butil
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.util.math import FixedSeed
from infinigen.core import surface
from infinigen.assets.trees.tree import build_radius_tree, recursive_path, FineTreeVertices
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class TreeBaseCoralFactory(BaseCoralFactory):
    default_scale = [1] * 3
    tentacle_prob = .8
    noise_strength = .01

    def __init__(self, factory_seed, coarse=False, method=None):
        super(TreeBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.tip = .4
        self.configs = {
            'twig': {'radius': .08, 'branch_config': self.twig_config},
            'bush': {'radius': .08, 'branch_config': self.bush_config}}
        self.weights = [.5, .5]
        with FixedSeed(self.factory_seed):
            if method is None:
                method = np.random.choice(list(self.configs.keys()), p=self.weights)
            self.radius, self.branch_config = map(self.configs[method].get, ['radius', 'branch_config'])
        self.points_fn = make_radius_points_fn(.05, .4)

    @property
    def bush_config(self):
        n_branch = np.random.randint(6, 8)
        n_major = np.random.randint(4, 5)
        n_minor = np.random.randint(4, 5)
        n_detail = np.random.randint(3, 4)
        span = uniform(.4, .5)
        detail_config = {
            'n': n_minor,
            'path_kargs': lambda idx: {
                'n_pts': n_detail + 1,
                'std': .4,
                'momentum': .6,
                'sz': .01 * (1.5 * n_detail - idx)},
            'spawn_kargs': lambda idx: {
                'rnd_idx': idx + 1,
                'ang_min': np.pi / 12,
                'ang_max': np.pi / 8,
                'axis2': [0, 0, 1]},
            'children': []}
        minor_config = {
            'n': n_major,
            'path_kargs': lambda idx: {
                'n_pts': n_minor + 1,
                'std': .4,
                'momentum': .4,
                'sz': .03 * (1.2 * n_minor - idx)},
            'spawn_kargs': lambda idx: {
                'rnd_idx': idx + 1,
                'ang_min': np.pi / 12,
                'ang_max': np.pi / 8,
                'axis2': [0, 0, 1]},
            'children': [detail_config]}
        major_config = {
            'n': n_branch,
            'path_kargs': lambda idx: {'n_pts': n_major + 1, 'std': .4, 'momentum': .4, 'sz': uniform(.08, .1)},
            'spawn_kargs': lambda idx: {
                'init_vec': [span * np.cos(2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)),
                    span * np.sin(2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)),
                    math.sqrt(1 - span * span)]},
            'children': [minor_config]}
        return major_config

    @property
    def twig_config(self):
        n_branch = np.random.randint(6, 8)
        n_major = np.random.randint(4, 5)
        n_minor = np.random.randint(4, 5)
        n_detail = np.random.randint(3, 4)
        span = uniform(.7, .8)
        detail_config = {
            'n': n_minor,
            'path_kargs': lambda idx: {
                'n_pts': n_detail * 2 + 1,
                'std': .4,
                'momentum': .6,
                'sz': .01 * (2.5 * n_detail - idx)},
            'spawn_kargs': lambda idx: {
                'rnd_idx': 2 * idx + 1,
                'ang_min': np.pi / 8,
                'ang_max': np.pi / 6,
                'axis2': [0, 0, 1]},
            'children': []}
        minor_config = {
            'n': n_major,
            'path_kargs': lambda idx: {
                'n_pts': n_minor * 2 + 1,
                'std': .4,
                'momentum': .4,
                'sz': .03 * (2.2 * n_minor - idx)},
            'spawn_kargs': lambda idx: {
                'rnd_idx': 2 * idx + 1,
                'ang_min': np.pi / 8,
                'ang_max': np.pi / 6,
                'axis2': [0, 0, 1]},
            'children': [detail_config]}
        major_config = {
            'n': n_branch,
            'path_kargs': lambda idx: {
                'n_pts': n_major * 2 + 1,
                'std': .4,
                'momentum': .4,
                'sz': uniform(.08, .1)},
            'spawn_kargs': lambda idx: {
                'init_vec': [span * np.cos(2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)),
                    span * np.sin(2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)),
                    math.sqrt(1 - span * span)]},
            'children': [minor_config]}
        return major_config

    @staticmethod
    def radius_fn(base_radius, size, resolution):
        radius_decay_root = .85
        radius_decay_leaf = uniform(.4, .6)
        radius = base_radius * radius_decay_root ** (np.arange(size * resolution) / resolution)
        radius[-resolution:] *= radius_decay_leaf ** (np.arange(resolution) / resolution)
        return radius

    def create_asset(self, face_size=0.01, **params) -> bpy.types.Object:
        resolution = 16
        obj = build_radius_tree(self.radius_fn, self.branch_config, self.radius, resolution)
        obj.scale = 2 * np.array(self.default_scale) / max(obj.dimensions[:2])
        butil.apply_transform(obj)
        surface.add_geomod(obj, geo_radius, apply=True, input_args=['radius', 32])
        tag_object(obj, 'tree_coral')
        return obj


class TwigBaseCoralFactory(TreeBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse, method='twig')


class BushBaseCoralFactory(TreeBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse, method='bush')
