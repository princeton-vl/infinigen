# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials.fabrics.fabric_random import fabric_shader_list
from infinigen.assets.utils.decorate import read_uv, write_uv
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg

from . import rug, text


class Art(text.Text):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(self.factory_seed):
            self.n_barcodes = 0
            self.n_texts = 0
            self.n_patches = np.random.randint(10, 15)

    @staticmethod
    def scale_uniform(min_, max_):
        return (max_ - min_) * log_uniform(0.1, 0.5)


class DarkArt(Art):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(self.factory_seed):
            self.darken_scale = uniform(5, 10)
            self.darken_ratio = uniform(0.5, 1)

    def make_shader_func(self, bbox):
        art_shader_func = super(DarkArt, self).make_shader_func(bbox)

        def shader_dark_art(nw: NodeWrangler):
            art_shader_func(nw)
            art_bsdf = nw.find(Nodes.PrincipledBSDF)[0]
            art_color = nw.find_from(art_bsdf.inputs[0])[0].from_socket
            dark_color = nw.new_node(
                Nodes.NoiseTexture, input_kwargs={"Scale": self.darken_scale}
            ).outputs[0]
            art_color = nw.new_node(
                Nodes.MixRGB,
                [self.darken_ratio, art_color, dark_color],
                attrs={"blend_type": "DARKEN"},
            ).outputs[2]
            nw.connect_input(art_color, art_bsdf.inputs[0])

        return shader_dark_art


class ArtComposite(DarkArt):
    @property
    def base_shader(self):
        raise NotImplementedError

    def make_shader_func(self, bbox):
        art_shader_func = super(ArtComposite, self).make_shader_func(bbox)

        def shader_art_composite(nw: NodeWrangler, **kwargs):
            self.base_shader(nw, **kwargs)
            nw_, base_bsdf = nw.find_recursive(Nodes.PrincipledBSDF)[-1]
            art_shader_func(nw_)
            art_bsdf = nw_.find(Nodes.PrincipledBSDF)[-1]
            art_color = nw_.find_from(art_bsdf.inputs[0])[0].from_socket
            nw_.nodes.remove(art_bsdf)
            nw_.connect_input(art_color, base_bsdf.inputs[0])
            nw_.connect_input(
                base_bsdf.outputs[0],
                nw_.find(Nodes.MaterialOutput)[0].inputs["Surface"],
            )

        return shader_art_composite

    @classmethod
    def make_sphere(cls):
        return text.make_sphere()


class ArtRug(ArtComposite):
    @property
    def base_shader(self):
        return rug.shader_rug


class ArtFabric(ArtComposite):
    @property
    def base_shader(self):
        return rg(fabric_shader_list)


def apply(obj, selection=None, bbox=(0, 1, 0, 1), scale=None, **kwargs):
    if scale is not None:
        write_uv(obj, read_uv(obj) * scale)
    Art(np.random.randint(1e5)).apply(obj, selection, bbox, **kwargs)
