# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: Primary author

import gin

from infinigen.assets.utils.joints import nodegroup_sliding_joint
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cube = nw.new_node(Nodes.MeshCube)

    cube_1 = nw.new_node(Nodes.MeshCube)

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Parent": cube.outputs["Mesh"],
            "Child": cube_1.outputs["Mesh"],
            "Position": (0.0000, 1.0000, 0.0000),
            "Axis": (0.0000, 1.0000, 0.0000),
            "Max": 1.0000,
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube)

    sliding_joint_1 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Geometry"],
            "Child": cube_2.outputs["Mesh"],
            "Position": (1.0000, 0.0000, 0.0000),
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 1.0000,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": sliding_joint_1.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


class Test9Factory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="Test9Factory")
    def sample_joint_parameters(
        cls,
    ):
        return {}

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        return {}

    def create_asset(self, asset_params=None, **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
        )

        return obj
