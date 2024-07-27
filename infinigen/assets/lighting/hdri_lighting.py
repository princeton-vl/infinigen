# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei
import os

import bpy
import gin
import numpy as np
from numpy.random import uniform

from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.random import random_general as rg

HDRI_RESOURCES = f"{os.getcwd()}/resources/hdri"


@gin.configurable
def hdri_lighting(
    nw: NodeWrangler,
    strength=("uniform", 0.8, 1.2),
):
    suffixes = [f for f in os.listdir(HDRI_RESOURCES) if f.endswith(".exr")]
    suffix = np.random.choice(suffixes)
    image = bpy.data.images.load(
        filepath=f"{HDRI_RESOURCES}/{suffix}", check_existing=True
    )
    texture_coord = nw.new_node(Nodes.TextureCoord)
    coord = nw.new_node(
        Nodes.Mapping,
        [texture_coord],
        input_kwargs={"Rotation": (0, 0, uniform(np.pi * 2))},
    )
    texture = nw.new_node(Nodes.EnvironmentTexture, [coord], attrs={"image": image})
    return nw.new_node(
        Nodes.Background, input_kwargs={"Color": texture, "Strength": rg(strength)}
    )


def add_lighting():
    nw = NodeWrangler(bpy.context.scene.world.node_tree)
    surface = hdri_lighting(nw)
    nw.new_node(Nodes.WorldOutput, input_kwargs={"Surface": surface})
