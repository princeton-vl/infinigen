# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from infinigen.assets.materials import common
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


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_mirror, selection, **kwargs)
