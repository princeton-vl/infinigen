# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from . import tentacles
from .base import BaseCoralFactory
from .diff_growth import (
    DiffGrowthBaseCoralFactory,
    LeatherBaseCoralFactory,
    TableBaseCoralFactory,
)
from .elkhorn import ElkhornBaseCoralFactory
from .fan import FanBaseCoralFactory
from .laplacian import CauliflowerBaseCoralFactory
from .reaction_diffusion import (
    BrainBaseCoralFactory,
    HoneycombBaseCoralFactory,
    ReactionDiffusionBaseCoralFactory,
)
from .star import StarBaseCoralFactory
from .tree import BushBaseCoralFactory, TreeBaseCoralFactory, TwigBaseCoralFactory
from .tube import TubeBaseCoralFactory


class CoralFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(CoralFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.factory_methods = [
                DiffGrowthBaseCoralFactory,
                ReactionDiffusionBaseCoralFactory,
                TubeBaseCoralFactory,
                TreeBaseCoralFactory,
                CauliflowerBaseCoralFactory,
                ElkhornBaseCoralFactory,
                StarBaseCoralFactory,
            ]
            weights = np.array([0.15, 0.2, 0.15, 0.2, 0.2, 0.15, 0.2])
            self.weights = weights / weights.sum()
            if factory_method is None:
                factory_method = np.random.choice(self.factory_methods, p=self.weights)
            self.factory: BaseCoralFactory = factory_method(factory_seed, coarse)
            self.base_hue = self.build_base_hue()
            self.material = surface.shaderfunc_to_material(
                self.shader_coral, self.base_hue
            )

    def create_asset(self, face_size=0.01, realize=True, **params):
        obj = self.factory.create_asset(**params)
        obj.scale = (
            2
            * np.array(self.factory.default_scale)
            / max(obj.dimensions[:2])
            * uniform(0.8, 1.2, 3)
        )
        butil.apply_transform(obj)
        remesh_with_attrs(obj, face_size)
        assign_material(obj, self.material)

        has_bump = uniform(0, 1) < self.factory.bump_prob
        if self.factory.noise_strength > 0:
            if has_bump:
                self.apply_noise_texture(obj)
            else:
                self.apply_bump(obj)

        tag_object(obj, "coral")

        if uniform(0, 1) < self.factory.tentacle_prob and not has_bump:
            t = tentacles.apply(
                obj,
                self.factory.points_fn,
                self.factory.density,
                realize,
                self.base_hue,
            )
            obj = join_objects([obj, t])

        return obj

    def apply_noise_texture(self, obj):
        t = np.random.choice(["STUCCI", "MARBLE"])
        texture = bpy.data.textures.new(name="coral", type=t)
        texture.noise_scale = log_uniform(0.01, 0.02)
        butil.modify_mesh(
            obj,
            "DISPLACE",
            True,
            strength=self.factory.noise_strength * uniform(0.9, 1.2),
            mid_level=0,
            texture=texture,
        )

    def apply_bump(self, obj):
        texture = bpy.data.textures.new(name="coral", type="VORONOI")
        texture.noise_scale = log_uniform(0.02, 0.03)
        texture.noise_intensity = log_uniform(1.5, 2)
        texture.distance_metric = "MINKOVSKY"
        texture.minkovsky_exponent = uniform(1, 1.5)
        butil.modify_mesh(
            obj,
            "DISPLACE",
            True,
            strength=-self.factory.noise_strength * uniform(1, 2),
            mid_level=1,
            texture=texture,
        )

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.25:
            base_hue = uniform(0, 1)
        else:
            base_hue = uniform(-0.2, 0.3) % 1
        return base_hue

    @staticmethod
    def shader_coral(nw: NodeWrangler, base_hue):
        shift = uniform(0.05, 0.1) * (-1) ** np.random.randint(2)
        subsurface_color = hsv2rgba(uniform(0, 1), uniform(0, 1), 1.0)
        bright_color = hsv2rgba((base_hue + shift) % 1, uniform(0.7, 0.9), 0.2)
        dark_color = hsv2rgba(base_hue, uniform(0.5, 0.7), 0.1)
        light_color = hsv2rgba(
            (base_hue + uniform(-0.2, 0.2)) % 1, uniform(0.2, 0.4), 0.4
        )
        specular = uniform(0.25, 0.5)

        color = build_color_ramp(
            nw,
            nw.musgrave(uniform(10, 20)),
            [0.0, 0.3, 0.7, 1.0],
            [dark_color, dark_color, bright_color, bright_color],
        )
        color = nw.new_node(
            Nodes.MixRGB,
            [
                nw.build_float_curve(
                    nw.musgrave(uniform(10, 20)),
                    [(0, 1), (uniform(0.3, 0.4), 0), (1, 0)],
                ),
                color,
                light_color,
            ],
        )

        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 1.0)])
        subsurface_ratio = uniform(0, 0.05) if uniform(0, 1) > 0.5 else 0
        subsurface_radius = [uniform(0.05, 0.2)] * 3
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
                "Subsurface Weight": subsurface_ratio,
                "Subsurface Radius": subsurface_radius,
                "Subsurface Color": subsurface_color,
            },
        )
        return bsdf


class LeatherCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(LeatherCoralFactory, self).__init__(
            factory_seed, coarse, LeatherBaseCoralFactory
        )


class TableCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TableCoralFactory, self).__init__(
            factory_seed, coarse, TableBaseCoralFactory
        )


class CauliflowerCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(CauliflowerCoralFactory, self).__init__(
            factory_seed, coarse, CauliflowerBaseCoralFactory
        )


class BrainCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BrainCoralFactory, self).__init__(
            factory_seed, coarse, BrainBaseCoralFactory
        )


class HoneycombCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(HoneycombCoralFactory, self).__init__(
            factory_seed, coarse, HoneycombBaseCoralFactory
        )


class BushCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BushCoralFactory, self).__init__(
            factory_seed, coarse, BushBaseCoralFactory
        )


class TwigCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TwigCoralFactory, self).__init__(
            factory_seed, coarse, TwigBaseCoralFactory
        )


class TubeCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TubeCoralFactory, self).__init__(
            factory_seed, coarse, TubeBaseCoralFactory
        )


class FanCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(FanCoralFactory, self).__init__(factory_seed, coarse, FanBaseCoralFactory)


class ElkhornCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ElkhornCoralFactory, self).__init__(
            factory_seed, coarse, ElkhornBaseCoralFactory
        )


class StarCoralFactory(CoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(StarCoralFactory, self).__init__(
            factory_seed, coarse, StarBaseCoralFactory
        )
