# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: primary author
# - Alexander Raistrick: implement placeholder

import bpy
from numpy.random import choice, uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.tables.legs.single_stand import (
    nodegroup_generate_single_stand,
)
from infinigen.assets.objects.tables.legs.straight import (
    nodegroup_generate_leg_straight,
)
from infinigen.assets.objects.tables.legs.wheeled import nodegroup_wheeled_leg
from infinigen.assets.objects.tables.strechers import nodegroup_strecher
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.tables.table_utils import (
    nodegroup_create_anchors,
    nodegroup_create_legs_and_strechers,
)
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util.math import FixedSeed


@node_utils.to_nodegroup(
    "geometry_create_legs", singleton=False, type="GeometryNodeTree"
)
def geometry_create_legs(nw: NodeWrangler, **kwargs):
    createanchors = nw.new_node(
        nodegroup_create_anchors().name,
        input_kwargs={
            "Profile N-gon": kwargs["Leg Number"],
            "Profile Width": kwargs["Leg Placement Top Relative Scale"]
            * kwargs["Top Profile Width"],
            "Profile Aspect Ratio": 1.0000,
        },
    )

    if kwargs["Leg Style"] == "single_stand":
        leg = nw.new_node(
            nodegroup_generate_single_stand(**kwargs).name,
            input_kwargs={
                "Leg Height": kwargs["Leg Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Resolution": 64,
            },
        )

        leg = nw.new_node(
            nodegroup_create_legs_and_strechers().name,
            input_kwargs={
                "Anchors": createanchors,
                "Keep Legs": True,
                "Leg Instance": leg,
                "Table Height": kwargs["Top Height"],
                "Leg Bottom Relative Scale": kwargs[
                    "Leg Placement Bottom Relative Scale"
                ],
                "Align Leg X rot": True,
            },
        )

    elif kwargs["Leg Style"] == "straight":
        leg = nw.new_node(
            nodegroup_generate_leg_straight(**kwargs).name,
            input_kwargs={
                "Leg Height": kwargs["Leg Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Resolution": 32,
                "N-gon": kwargs["Leg NGon"],
                "Fillet Ratio": 0.1,
            },
        )

        strecher = nw.new_node(
            nodegroup_strecher().name,
            input_kwargs={"Profile Width": kwargs["Leg Diameter"] * 0.5},
        )

        leg = nw.new_node(
            nodegroup_create_legs_and_strechers().name,
            input_kwargs={
                "Anchors": createanchors,
                "Keep Legs": True,
                "Leg Instance": leg,
                "Table Height": kwargs["Top Height"],
                "Strecher Instance": strecher,
                "Strecher Index Increment": kwargs["Strecher Increament"],
                "Strecher Relative Position": kwargs["Strecher Relative Pos"],
                "Leg Bottom Relative Scale": kwargs[
                    "Leg Placement Bottom Relative Scale"
                ],
                "Align Leg X rot": True,
            },
        )

    elif kwargs["Leg Style"] == "wheeled":
        leg = nw.new_node(
            nodegroup_wheeled_leg(**kwargs).name,
            input_kwargs={
                "Joint Height": kwargs["Leg Joint Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Top Height": kwargs["Top Height"],
                "Wheel Width": kwargs["Leg Wheel Width"],
                "Wheel Rotation": kwargs["Leg Wheel Rot"],
                "Pole Length": kwargs["Leg Pole Length"],
                "Leg Number": kwargs["Leg Pole Number"],
            },
        )

    else:
        raise NotImplementedError

    leg = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": leg, "Material": kwargs["LegMaterial"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": leg},
        attrs={"is_active_output": True},
    )


def geometry_assemble_table(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": kwargs["Top Thickness"],
            "N-gon": kwargs["Top Profile N-gon"],
            "Profile Width": kwargs["Top Profile Width"],
            "Aspect Ratio": kwargs["Top Profile Aspect Ratio"],
            "Fillet Ratio": kwargs["Top Profile Fillet Ratio"],
            "Fillet Radius Vertical": kwargs["Top Vertical Fillet Ratio"],
        },
    )

    tabletop_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generatetabletop,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    tabletop_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": tabletop_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [tabletop_instance, legs]}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": generatetabletop.outputs["Curve"]}
    )
    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": resample_curve})

    voff = kwargs["Top Height"] + kwargs["Top Thickness"]
    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": fill_curve, "Offset Scale": -voff, "Individual": False},
    )
    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], fill_curve]},
    )
    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": (0, 0, voff)},
    )
    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: kwargs["is_placeholder"],
            1: join_geometry,
            2: transform_geometry_1,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch},
        attrs={"is_active_output": True},
    )


class TableCocktailFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super(TableCocktailFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)

            # self.clothes_scatter = ClothesCover(factory_fn=blanket.BlanketFactory, width=log_uniform(.8, 1.2),
            #                                     size=uniform(.8, 1.2)) if uniform() < .3 else NoApply()
            self.clothes_scatter = NoApply()
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )

        self.params.update(self.material_params)

    def get_material_params(self):
        material_assignments = AssetList["TableCocktailFactory"]()
        params = {
            "TopMaterial": material_assignments["top"].assign_material(),
            "LegMaterial": material_assignments["leg"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = uniform() < scratch_prob
        is_edge_wear = uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # all in meters
        if dimensions is None:
            x = uniform(0.5, 0.8)
            z = uniform(1.0, 1.5)
            dimensions = (x, x, z)

        x, y, z = dimensions

        NGon = choice([4, 32])
        if NGon >= 32:
            round_table = True
        else:
            round_table = False

        leg_style = choice(["straight", "single_stand"])
        if leg_style == "single_stand":
            leg_number = 1
            leg_diameter = uniform(0.7 * x, 0.9 * x)

            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]

        elif leg_style == "straight":
            leg_diameter = uniform(0.05, 0.07)

            if round_table:
                leg_number = choice([3, 4])
            else:
                leg_number = NGon

            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, uniform(0.85, 0.95)),
                (1.0, uniform(0.4, 0.6)),
            ]

        else:
            raise NotImplementedError

        top_thickness = uniform(0.02, 0.05)

        parameters = {
            "Top Profile N-gon": 32 if round_table else 4,
            "Top Profile Width": x if round_table else 1.414 * x,
            "Top Profile Aspect Ratio": 1.0,
            "Top Profile Fillet Ratio": 0.499 if round_table else uniform(0.0, 0.05),
            "Top Thickness": top_thickness,
            "Top Vertical Fillet Ratio": uniform(0.1, 0.3),
            # 'Top Material': choice(['marble', 'tiled_wood', 'plastic', 'glass']),
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Number": leg_number,
            "Leg Style": leg_style,
            "Leg NGon": choice([4, 32]),
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": uniform(1.1, 1.3),
            "Leg Height": 1.0,
            "Leg Diameter": leg_diameter,
            "Leg Curve Control Points": leg_curve_ctrl_pts,
            # 'Leg Material': choice(['metal', 'wood', 'glass']),
            "Strecher Relative Pos": uniform(0.2, 0.6),
            "Strecher Increament": choice([0, 1, 2]),
        }

        return parameters

    def _execute_geonodes(self, is_placeholder):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        kwargs = {**self.params, "is_placeholder": is_placeholder}
        surface.add_geomod(
            obj, geometry_assemble_table, apply=True, input_kwargs=kwargs
        )
        tagging.tag_system.relabel_obj(obj)

        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return self._execute_geonodes(is_placeholder=True)

    def create_asset(self, **_):
        return self._execute_geonodes(is_placeholder=False)

    def finalize_assets(self, assets):
        self.clothes_scatter.apply(assets)
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
