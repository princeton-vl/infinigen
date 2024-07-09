# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from infinigen.core import surface
from infinigen.core.nodes import Nodes, NodeWrangler


def shader_mirror(nw: NodeWrangler, **kwargs):
    glossy_bsdf = nw.new_node(
        "ShaderNodeBsdfGlossy",
        input_kwargs={
            "Color": (1.0, 1.0, 1.0, 1.0),
            "Roughness": 0,
        },
    )

    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": glossy_bsdf})


class Mirror:
    shader = shader_mirror

    def generate():
        return surface.shaderfunc_to_material(shader_mirror)

    __call__ = generate
