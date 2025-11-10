import gin

from infinigen.assets.utils.joints import nodegroup_hinge_joint
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cube_2 = nw.new_node(Nodes.MeshCube)

    cube = nw.new_node(Nodes.MeshCube)

    cube_1 = nw.new_node(Nodes.MeshCube)

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.5000, 0.5000, 0.5000),
        },
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Parent": cube.outputs["Mesh"],
            "Child": transform_geometry,
            "Position": (0.5000, 0.5000, 0.5000),
            "Axis": (1.0000, 0.0000, 1.0000),
            "Show Joint": True,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Translation": (0.0000, 0.5000, 0.5000),
            "Rotation": (0.7854, 0.0000, 0.0000),
        },
    )

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Parent": cube_2.outputs["Mesh"],
            "Child": transform_geometry_1,
            "Position": (0.5000, 0.5000, 0.5000),
            "Show Joint": True,
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_1.outputs["Geometry"],
            "Translation": (0.0000, 0.0000, 0.5000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_2},
        attrs={"is_active_output": True},
    )


class Test8Factory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="Test8Factory")
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
