# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import math

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.objects.corals.base import BaseCoralFactory
from infinigen.assets.objects.corals.tentacles import make_radius_points_fn
from infinigen.assets.objects.trees.tree import build_radius_tree
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.core.util.math import FixedSeed


class TreeBaseCoralFactory(BaseCoralFactory):
    default_scale = [1] * 3
    tentacle_prob = 0.8
    noise_strength = 0.01

    def __init__(self, factory_seed, coarse=False, method=None):
        super(TreeBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.tip = 0.4
        self.configs = {
            "twig": {"radius": 0.08, "branch_config": self.twig_config},
            "bush": {"radius": 0.08, "branch_config": self.bush_config},
        }
        self.weights = [0.5, 0.5]
        with FixedSeed(self.factory_seed):
            if method is None:
                method = np.random.choice(list(self.configs.keys()), p=self.weights)
            self.radius, self.branch_config = map(
                self.configs[method].get, ["radius", "branch_config"]
            )
        self.points_fn = make_radius_points_fn(0.05, 0.4)

    @property
    def bush_config(self):
        n_branch = np.random.randint(6, 8)
        n_major = np.random.randint(4, 5)
        n_minor = np.random.randint(4, 5)
        n_detail = np.random.randint(3, 4)
        span = uniform(0.4, 0.5)
        detail_config = {
            "n": n_minor,
            "path_kargs": lambda idx: {
                "n_pts": n_detail + 1,
                "std": 0.4,
                "momentum": 0.6,
                "sz": 0.01 * (1.5 * n_detail - idx),
            },
            "spawn_kargs": lambda idx: {
                "rnd_idx": idx + 1,
                "ang_min": np.pi / 12,
                "ang_max": np.pi / 8,
                "axis2": [0, 0, 1],
            },
            "children": [],
        }
        minor_config = {
            "n": n_major,
            "path_kargs": lambda idx: {
                "n_pts": n_minor + 1,
                "std": 0.4,
                "momentum": 0.4,
                "sz": 0.03 * (1.2 * n_minor - idx),
            },
            "spawn_kargs": lambda idx: {
                "rnd_idx": idx + 1,
                "ang_min": np.pi / 12,
                "ang_max": np.pi / 8,
                "axis2": [0, 0, 1],
            },
            "children": [detail_config],
        }
        major_config = {
            "n": n_branch,
            "path_kargs": lambda idx: {
                "n_pts": n_major + 1,
                "std": 0.4,
                "momentum": 0.4,
                "sz": uniform(0.08, 0.1),
            },
            "spawn_kargs": lambda idx: {
                "init_vec": [
                    span
                    * np.cos(
                        2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)
                    ),
                    span
                    * np.sin(
                        2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)
                    ),
                    math.sqrt(1 - span * span),
                ]
            },
            "children": [minor_config],
        }
        return major_config

    @property
    def twig_config(self):
        n_branch = np.random.randint(6, 8)
        n_major = np.random.randint(4, 5)
        n_minor = np.random.randint(4, 5)
        n_detail = np.random.randint(3, 4)
        span = uniform(0.7, 0.8)
        detail_config = {
            "n": n_minor,
            "path_kargs": lambda idx: {
                "n_pts": n_detail * 2 + 1,
                "std": 0.4,
                "momentum": 0.6,
                "sz": 0.01 * (2.5 * n_detail - idx),
            },
            "spawn_kargs": lambda idx: {
                "rnd_idx": 2 * idx + 1,
                "ang_min": np.pi / 8,
                "ang_max": np.pi / 6,
                "axis2": [0, 0, 1],
            },
            "children": [],
        }
        minor_config = {
            "n": n_major,
            "path_kargs": lambda idx: {
                "n_pts": n_minor * 2 + 1,
                "std": 0.4,
                "momentum": 0.4,
                "sz": 0.03 * (2.2 * n_minor - idx),
            },
            "spawn_kargs": lambda idx: {
                "rnd_idx": 2 * idx + 1,
                "ang_min": np.pi / 8,
                "ang_max": np.pi / 6,
                "axis2": [0, 0, 1],
            },
            "children": [detail_config],
        }
        major_config = {
            "n": n_branch,
            "path_kargs": lambda idx: {
                "n_pts": n_major * 2 + 1,
                "std": 0.4,
                "momentum": 0.4,
                "sz": uniform(0.08, 0.1),
            },
            "spawn_kargs": lambda idx: {
                "init_vec": [
                    span
                    * np.cos(
                        2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)
                    ),
                    span
                    * np.sin(
                        2 * np.pi * idx / n_branch + uniform(-np.pi / 9, np.pi / 9)
                    ),
                    math.sqrt(1 - span * span),
                ]
            },
            "children": [minor_config],
        }
        return major_config

    @staticmethod
    def radius_fn(base_radius, size, resolution):
        radius_decay_root = 0.85
        radius_decay_leaf = uniform(0.4, 0.6)
        radius = base_radius * radius_decay_root ** (
            np.arange(size * resolution) / resolution
        )
        radius[-resolution:] *= radius_decay_leaf ** (
            np.arange(resolution) / resolution
        )
        return radius

    def create_asset(self, face_size=0.01, **params) -> bpy.types.Object:
        resolution = 16
        obj = build_radius_tree(
            self.radius_fn, self.branch_config, self.radius, resolution
        )
        obj.scale = 2 * np.array(self.default_scale) / max(obj.dimensions[:2])
        butil.apply_transform(obj)
        surface.add_geomod(obj, geo_radius, apply=True, input_args=["radius", 32])
        tag_object(obj, "tree_coral")
        return obj


class TwigBaseCoralFactory(TreeBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse, method="twig")


class BushBaseCoralFactory(TreeBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse, method="bush")
