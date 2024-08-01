# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np

import infinigen.core.util.blender as butil
from infinigen.assets.objects.corals.base import BaseCoralFactory
from infinigen.assets.utils.decorate import geo_extension
from infinigen.assets.utils.mesh import build_convex_mesh
from infinigen.assets.utils.object import center, mesh2obj, new_icosphere
from infinigen.assets.utils.reaction_diffusion import (
    feed2kill,
    make_periodic_weight_fn,
    reaction_diffusion,
)
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.core.util.math import FixedSeed


class ReactionDiffusionBaseCoralFactory(BaseCoralFactory):
    tentacle_prob = 0.0
    noise_strength = 0.01

    def __init__(self, factory_seed, coarse=False):
        super(ReactionDiffusionBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.makers = [self.brain_make, self.honeycomb_make]
        self.weights = [0.5, 0.5]
        with FixedSeed(self.factory_seed):
            self.maker = np.random.choice(self.makers, p=self.weights)

    @staticmethod
    def reaction_diffusion_make(weight_fn, **kwargs):
        mesh = build_convex_mesh()
        wrapped = mesh2obj(mesh)

        subsurf_level = 2
        butil.modify_mesh(
            wrapped, "SUBSURF", levels=subsurf_level, render_levels=subsurf_level
        )

        obj = new_icosphere(subdivisions=8, radius=3)
        reaction_diffusion(obj, weight_fn, **kwargs)
        obj.location = center(wrapped)
        butil.apply_transform(obj, loc=True)
        butil.modify_mesh(
            obj,
            "SHRINKWRAP",
            target=wrapped,
            wrap_method="PROJECT",
            use_negative_direction=True,
        )

        obj.location[-1] = 1
        butil.apply_transform(obj, loc=True)
        surface.add_geomod(obj, geo_extension, apply=True)
        butil.modify_mesh(
            obj, "DISPLACE", vertex_group="B", strength=0.4, mid_level=0.0
        )
        butil.delete(wrapped)
        tag_object(obj, "reactiondiffusion_coral")
        return obj

    @staticmethod
    def brain_make():
        feed_rate = 0.055
        kill_rate = feed2kill(feed_rate)
        return ReactionDiffusionBaseCoralFactory.reaction_diffusion_make(
            make_periodic_weight_fn(100, 0.02), feed_rate=feed_rate, kill_rate=kill_rate
        )

    @staticmethod
    def honeycomb_make():
        feed_rate = 0.070
        kill_rate = feed2kill(feed_rate) - 0.001
        return ReactionDiffusionBaseCoralFactory.reaction_diffusion_make(
            make_periodic_weight_fn(5), feed_rate=feed_rate, kill_rate=kill_rate
        )

    def create_asset(self, face_size=0.01, **params):
        return self.maker()


class BrainBaseCoralFactory(ReactionDiffusionBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BrainBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.maker = self.brain_make


class HoneycombBaseCoralFactory(ReactionDiffusionBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(HoneycombBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.maker = self.honeycomb_make
