# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.object import new_plane
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.random import log_uniform

from . import common
from .table_materials import shader_wood
from .utils.surface_utils import perturb_coordinates


def shader_hardwood_floor(nw: NodeWrangler, rotation=None):
    vec = nw.new_node(
        Nodes.Mapping,
        [nw.new_node(Nodes.TextureCoord).outputs["Object"]],
        input_kwargs={"Rotation": rotation},
    )
    color, mortar = map(
        nw.new_node(
            Nodes.BrickTexture,
            [vec, (0, 0, 0, 1), (1, 1, 1, 1), (0, 0, 0, uniform(0.01, 0.02))],
            input_kwargs={
                "Scale": 1,
                "Row Height": log_uniform(0.06, 0.15),
                "Brick Width": log_uniform(0.6, 1),
                "Mortar Size": uniform(0.002, 0.002),
            },
        ).outputs.get,
        ["Color", "Fac"],
    )
    location = nw.combine(color, color, color)
    shader_wood(nw)
    perturb_coordinates(nw, nw.find(Nodes.TextureCoord)[1], location, 0)
    principled_bsdf = nw.find(Nodes.PrincipledBSDF)[0]
    wood_color = nw.find_from(principled_bsdf.inputs[0])[0].from_socket
    color = nw.new_node(Nodes.MixRGB, [mortar, wood_color, color])
    nw.links.remove(nw.find_from(principled_bsdf.inputs[0])[0])
    nw.connect_input(principled_bsdf.inputs[0], color)


def apply(obj, selection=None, rotation=None, **kwargs):
    if rotation is None:
        rotation = (0, 0, 0) if uniform() < 0.1 else (0, 0, np.pi / 2)
    return common.apply(obj, shader_hardwood_floor, selection, rotation, **kwargs)


def make_sphere():
    obj = new_plane()
    obj.rotation_euler[0] = np.pi / 2
    return obj
