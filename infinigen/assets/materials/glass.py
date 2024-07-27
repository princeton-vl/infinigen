# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.color import hsv2rgba


def shader_glass(nw: NodeWrangler, color=None, is_window=False, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler
    if color is None:
        color = get_glass_color(clear=False)

    # TODO windows are currently planes so refract and dont unrefract. ideally we just fix the geometry
    # warning: currently this IOR also accidentally just turns off reflections, the window plane is pretty much invisible.
    ior = 1.5 if not is_window else 1.0

    light_path = nw.new_node(Nodes.LightPath)

    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)

    shader = nw.new_node(
        Nodes.GlassBSDF, input_kwargs={"Roughness": 0.0200, "IOR": ior}
    )

    if is_window:
        shader = nw.new_node(
            Nodes.MixShader,
            input_kwargs={
                "Fac": light_path.outputs["Is Camera Ray"],
                1: transparent_bsdf,
                2: shader,
            },
        )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": shader},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, clear=False, **kwargs):
    color = get_glass_color(clear)
    common.apply(obj, shader_glass, selection, color, **kwargs)


def get_glass_color(clear):
    if uniform(0, 1) < 0.5:
        color = 1, 1, 1, 1
    else:
        color = hsv2rgba(uniform(0, 1), 0.01 if clear else uniform(0.05, 0.25), 1)
    return color
