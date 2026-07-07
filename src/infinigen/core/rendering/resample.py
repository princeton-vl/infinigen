# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import bpy

from infinigen.assets.lighting import sky_lighting
from infinigen.assets.objects import rocks, trees
from infinigen.core.nodes.node_utils import resample_node_group
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.logging import Timer
from infinigen.core.util.math import FixedSeed


def resample_all(factory_class):
    for placeholder_col in butil.get_collection("placeholders").children:
        classname, _ = placeholder_col.name.split("(")
        if classname != factory_class.__name__:
            continue

        placeholders = [o for o in placeholder_col.objects if o.parent is None]
        for pholder in placeholders:
            factory_class.quickly_resample(pholder)


def resample_scene(scene_seed):
    with FixedSeed(scene_seed), Timer("Resample noise nodes in materials"):
        for material in bpy.data.materials:
            nw = NodeWrangler(material.node_tree)
            resample_node_group(nw, scene_seed)

    with FixedSeed(scene_seed), Timer("Resample noise nodes in scatters"):
        for obj in bpy.data.objects:
            for modifier in obj.modifiers:
                if not any(
                    obj.name.startswith(s)
                    for s in ["BlenderRockFactory", "CloudFactory"]
                ):
                    if modifier.type == "NODES":
                        nw = NodeWrangler(modifier.node_group)
                        resample_node_group(nw, scene_seed)

    with (
        FixedSeed(scene_seed),
        Timer("Resample all placeholders"),
    ):  # CloudFactory too expensive
        resample_all(rocks.GlowingRocksFactory)
        resample_all(trees.TreeFactory)
        resample_all(trees.BushFactory)
        # resample_all(CreatureFactory)
    with FixedSeed(scene_seed):
        sky_lighting.add_lighting()
