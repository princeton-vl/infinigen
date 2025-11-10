import gin

from infinigen.assets.utils.joints import (
    nodegroup_duplicate_joints_on_parent,
    nodegroup_hinge_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cube = nw.new_node(Nodes.MeshCube)

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (0.1000, 0.1000, 0.1000)}
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Parent": cube.outputs["Mesh"],
            "Child": cube_1.outputs["Mesh"],
            "Position": (0.0000, 0.0000, 0.5000),
        },
    )

    grid = nw.new_node(Nodes.MeshGrid)

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint.outputs["Parent"],
            "Child": hinge_joint.outputs["Child"],
            "Points": grid.outputs["Mesh"],
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": duplicate_joints_on_parent,
            "Translation": (0.0000, 0.0000, 0.5000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


class Test2Factory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="Test2Factory")
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
