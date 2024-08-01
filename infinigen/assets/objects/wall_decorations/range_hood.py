# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.table_decorations.utils import nodegroup_lofting_poly
from infinigen.assets.objects.tables.table_utils import nodegroup_n_gon_profile
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed


class RangeHoodFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super(RangeHoodFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)
            self.surface, self.scratch, self.edge_wear = self.get_material_params()

    def get_material_params(self):
        material_assignments = AssetList["RangeHoodFactory"]()
        surface = material_assignments["surface"].assign_material()

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = np.random.uniform() < scratch_prob
        is_edge_wear = np.random.uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return surface, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # all in meters
        if dimensions is None:
            x = 0.55
            y = 0.75
            z = 1.0
            dimensions = (x, y, z)

        x, y, z = dimensions

        height_1 = uniform(0.05, 0.07)
        height_2 = uniform(0.1, 0.3)
        scale_2 = uniform(0.25, 0.4)

        parameters = {
            "Height_total": z,
            "Width": y,
            "Depth": x,
            "Height_1": height_1,
            "Scale_2": scale_2,
            "Height_2": height_2,
        }

        return parameters

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        surface.add_geomod(
            obj, geometry_generate_hood, apply=True, input_kwargs=self.params
        )
        butil.modify_mesh(obj, "SOLIDIFY", apply=True, thickness=0.002)
        butil.modify_mesh(obj, "SUBSURF", apply=True, levels=1, render_levels=1)

        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets)
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


def geometry_generate_hood(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    generatetabletop = nw.new_node(
        geometry_range_hood().name,
        input_kwargs={
            "Resolution": 64,
            "Height_total": kwargs["Height_total"],
            "Width": kwargs["Width"],
            "Depth": kwargs["Depth"],
            "Height_1": kwargs["Height_1"],
            "Scale_2": kwargs["Scale_2"],
            "Height_2": kwargs["Height_2"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": generatetabletop},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "geometry_range_hood", singleton=False, type="GeometryNodeTree"
)
def geometry_range_hood(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Resolution", 128),
            ("NodeSocketFloat", "Height_total", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketFloat", "Profile Fillet Ratio", 0.0100),
            ("NodeSocketFloat", "Height_1", 0.0000),
            ("NodeSocketFloat", "Scale_2", 0.0000),
            ("NodeSocketFloat", "Height_2", 0.3000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 1.4140},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: group_input.outputs["Width"]},
        attrs={"operation": "DIVIDE"},
    )

    ngonprofile = nw.new_node(
        nodegroup_n_gon_profile().name,
        input_kwargs={
            "Profile Width": multiply,
            "Profile Aspect Ratio": divide,
            "Profile Fillet Ratio": group_input.outputs["Profile Fillet Ratio"],
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": ngonprofile, "Count": group_input.outputs["Resolution"]},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": resample_curve, "Translation": combine_xyz},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height_1"]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry, "Translation": combine_xyz_1},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height_2"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry,
            "Translation": combine_xyz_2,
            "Scale": group_input.outputs["Scale_2"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Height_total"],
            1: group_input.outputs["Height_2"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Translation": combine_xyz_3},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_geometry_3,
                transform_geometry_2,
                transform_geometry_1,
                transform_geometry,
            ]
        },
    )

    lofting_poly = nw.new_node(
        nodegroup_lofting_poly().name,
        input_kwargs={
            "Profile Curves": join_geometry,
            "U Resolution": group_input.outputs["Resolution"],
            "V Resolution": group_input.outputs["Resolution"],
        },
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={
            "Geometry": lofting_poly.outputs["Geometry"],
            "Selection": lofting_poly.outputs["Top"],
        },
    )

    grid = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": group_input.outputs["Width"],
            "Size Y": group_input.outputs["Depth"],
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_2})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Translation": combine_xyz_4,
            "Rotation": (-0.0698, 0.0000, 0.0000),
            "Scale": (0.9800, 0.9800, 1.0000),
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Rotation": (0.1047, 0.0000, 0.0000),
            "Scale": (0.9500, 0.9700, 1.0000),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [delete_geometry, transform_geometry_5]},
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Rotation": (0.0, 0.0000, -np.pi / 2),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_6},
        attrs={"is_active_output": True},
    )
