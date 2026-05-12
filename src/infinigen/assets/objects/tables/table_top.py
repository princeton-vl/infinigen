# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy

from infinigen.assets.objects.tables.table_utils import (
    nodegroup_create_cap,
    nodegroup_n_gon_cylinder,
)
from infinigen.core import surface
from infinigen.core import tags as t
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_nodegroup
from infinigen.core.util.math import FixedSeed


@node_utils.to_nodegroup(
    "nodegroup_capped_cylinder", singleton=False, type="GeometryNodeTree"
)
def nodegroup_capped_cylinder(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Radius", 0.2000),
            ("NodeSocketFloat", "Cap Flatness", 4.0000),
            ("NodeSocketFloat", "Fillet Radius Vertical", 0.4000),
            ("NodeSocketFloat", "Cap Relative Scale", 1.0000),
            ("NodeSocketFloat", "Cap Relative Z Offset", 0.0000),
            ("NodeSocketInt", "Resolution", 64),
        ],
    )

    create_cap = nw.new_node(
        nodegroup_create_cap().name,
        input_kwargs={
            "Radius": group_input.outputs["Cap Flatness"],
            "Resolution": group_input.outputs["Resolution"],
        },
        label="CreateCap",
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["Cap Relative Z Offset"]},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 0.5},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input.outputs["Cap Relative Scale"]},
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": create_cap,
            "Translation": combine_xyz_5,
            "Scale": add_1,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 1.0},
        attrs={"operation": "MULTIPLY"},
    )

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": multiply,
            "N-gon": group_input.outputs["Resolution"],
            "Profile Width": multiply_2,
            "Aspect Ratio": 1.0000,
            "Fillet Ratio": 0.0000,
            "Fillet Radius Vertical": group_input.outputs["Fillet Radius Vertical"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_5, generatetabletop]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_generate_table_top", singleton=False, type="GeometryNodeTree"
)
def nodegroup_generate_table_top(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (1.0000, 0.0000, 1.0000),
            "End": (1.0000, 0.0000, -1.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketInt", "N-gon", 0),
            ("NodeSocketFloat", "Profile Width", 0.5000),
            ("NodeSocketFloat", "Aspect Ratio", 0.5000),
            ("NodeSocketFloat", "Fillet Ratio", 0.2000),
            ("NodeSocketFloat", "Fillet Radius Vertical", 0.0000),
        ],
    )

    ngoncylinder = nw.new_node(
        nodegroup_n_gon_cylinder().name,
        input_kwargs={
            "Radius Curve": curve_line,
            "Height": group_input.outputs["Thickness"],
            "N-gon": group_input.outputs["N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Aspect Ratio": group_input.outputs["Aspect Ratio"],
            "Fillet Ratio": group_input.outputs["Fillet Ratio"],
            "Profile Resolution": 512,
            "Resolution": 10,
        },
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={"Resolution": 4, "Radius": 0.7071, "Sweep Angle": 4.7124},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": arc.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, -0.7854),
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_2, "Translation": (0.0000, 0.5000, 0.0000)},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 1.0000,
            "Y": group_input.outputs["Fillet Radius Vertical"],
            "Z": 1.0000,
        },
    )

    transform_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_3, "Scale": combine_xyz}
    )

    fillet_curve = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={
            "Curve": transform_4,
            "Count": 8,
            "Radius": group_input.outputs["Fillet Radius Vertical"],
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": fillet_curve,
            "Rotation": (1.5708, 1.5708, 0.0000),
            "Scale": group_input.outputs["Thickness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": ngoncylinder.outputs["Profile Curve"],
            "Profile Curve": transform_6,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Translation": combine_xyz_1},
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={"A": index, "B": 0},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    cap = tag_nodegroup(
        nw, ngoncylinder.outputs["Caps"], t.Subpart.SupportSurface, selection=equal
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_5, cap]}
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": join_geometry})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Thickness"]}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": flip_faces, "Translation": combine_xyz_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": transform_1,
            "Curve": ngoncylinder.outputs["Profile Curve"],
        },
    )


def geometry_generate_table_top_wrapper(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Profile N-gon", kwargs["Profile N-gon"]),
            ("NodeSocketFloat", "Profile Width", kwargs["Profile Width"]),
            ("NodeSocketFloat", "Profile Aspect Ratio", kwargs["Profile Aspect Ratio"]),
            ("NodeSocketFloat", "Profile Fillet Ratio", kwargs["Profile Fillet Ratio"]),
            ("NodeSocketFloat", "Thickness", kwargs["Thickness"]),
            (
                "NodeSocketFloat",
                "Vertical Fillet Ratio",
                kwargs["Vertical Fillet Ratio"],
            ),
        ],
    )

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": group_input.outputs["Thickness"],
            "N-gon": group_input.outputs["Profile N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Aspect Ratio": group_input.outputs["Profile Aspect Ratio"],
            "Fillet Ratio": group_input.outputs["Profile Fillet Ratio"],
            "Fillet Radius Vertical": group_input.outputs["Vertical Fillet Ratio"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": generatetabletop},
        attrs={"is_active_output": True},
    )


class TableTopFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TableTopFactory, self).__init__(factory_seed, coarse=coarse)

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters()

    @staticmethod
    def sample_parameters():
        # all in meters
        return {
            "Profile N-gon": 4,
            "Profile Width": 1.0,
            "Profile Aspect Ratio": 1.0,
            "Profile Fillet Ratio": 0.2000,
            "Thickness": 0.1000,
            "Vertical Fillet Ratio": 0.2000,
        }

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
            obj,
            geometry_generate_table_top_wrapper,
            apply=False,
            input_kwargs=self.params,
        )

        return obj
