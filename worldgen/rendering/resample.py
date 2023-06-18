
# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson
# Date Signed: May 2 2023

import bpy

from nodes.node_utils import resample_node_group
from nodes.node_wrangler import NodeWrangler

from lighting import lighting

from assets.trees.generate import TreeFactory, BushFactory
from assets.glowing_rocks import GlowingRocksFactory

from util.logging import Timer
from util.math import FixedSeed, int_hash
from util import blender as butil

def resample_all(factory_class):
    for placeholder_col in butil.get_collection('placeholders').children:
        classname, _ = placeholder_col.name.split('(')
        if classname != factory_class.__name__:
            continue

        placeholders = [o for o in placeholder_col.objects if o.parent is None]
        for pholder in placeholders:
            factory_class.quickly_resample(pholder)

def resample_scene(scene_seed):
    with FixedSeed(scene_seed), Timer('Resample noise nodes in materials'):
        for material in bpy.data.materials:
            nw = NodeWrangler(material.node_tree)
            resample_node_group(nw, scene_seed)
    with FixedSeed(scene_seed), Timer('Resample noise nodes in scatters'):
        for obj in bpy.data.objects:
            for modifier in obj.modifiers:
                if not any(obj.name.startswith(s) for s in ["BlenderRockFactory", "CloudFactory"]):
                    if modifier.type == 'NODES':
                        nw = NodeWrangler(modifier.node_group)
                        resample_node_group(nw, scene_seed)

    with FixedSeed(scene_seed), Timer('Resample all placeholders'):  # CloudFactory too expensive
        resample_all(GlowingRocksFactory)
        resample_all(TreeFactory)
        resample_all(BushFactory)
        #resample_all(CreatureFactory)
    with FixedSeed(scene_seed):
        lighting.add_lighting()