# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import bpy
from numpy.random import uniform as U

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import clip_gaussian


def shader_blackbody_temp(nw, params):
    blackbody = nw.new_node(
        Nodes.BlackBody, input_kwargs={"Temperature": params["Temperature"]}
    )
    emission = nw.new_node(Nodes.Emission, input_kwargs={"Color": blackbody})
    nw.new_node(Nodes.LightOutput, [emission])


class PointLampFactory(AssetFactory):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.params = {
                "Wattage": U(40, 100),
                "Radius": U(0.02, 0.03),
                "Temperature": clip_gaussian(4700, 700, 3500, 6500),
            }

    def create_placeholder(self, **_):
        cube = butil.spawn_cube(size=2)
        cube.scale = (self.params["Radius"],) * 3
        butil.apply_transform(cube)
        return cube

    def create_asset(self, **_) -> bpy.types.Object:
        bpy.ops.object.light_add(type="POINT")
        lamp = bpy.context.active_object
        lamp.data.energy = self.params["Wattage"]
        lamp.data.shadow_soft_size = self.params["Radius"]
        lamp.data.use_nodes = True

        nw = NodeWrangler(lamp.data.node_tree)
        shader_blackbody_temp(nw, params=self.params)

        return lamp
