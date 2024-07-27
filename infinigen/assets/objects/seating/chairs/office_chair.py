# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

import bpy
import numpy as np
from numpy.random import choice, uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.seating.chairs.seats.curvy_seats import (
    generate_curvy_seats,
)
from infinigen.assets.objects.tables.cocktail_table import geometry_create_legs
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def geometry_assemble_chair(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    generateseat = nw.new_node(
        generate_curvy_seats().name,
        input_kwargs={
            "Width": kwargs["Top Profile Width"],
            "Front Relative Width": kwargs["Top Front Relative Width"],
            "Front Bent": kwargs["Top Front Bent"],
            "Seat Bent": kwargs["Top Seat Bent"],
            "Mid Bent": kwargs["Top Mid Bent"],
            "Mid Relative Width": kwargs["Top Mid Relative Width"],
            "Back Bent": kwargs["Top Back Bent"],
            "Back Relative Width": kwargs["Top Back Relative Width"],
            "Mid Pos": kwargs["Top Mid Pos"],
            "Seat Height": kwargs["Top Thickness"],
        },
    )

    seat_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generateseat,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    seat_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": seat_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [seat_instance, legs]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


class OfficeChairFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super(OfficeChairFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params, leg_style = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params(leg_style)
            )
        self.params.update(self.material_params)

    def get_material_params(self, leg_style):
        material_assignments = AssetList["OfficeChairFactory"](leg_style)
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
            x = uniform(0.5, 0.6)
            z = uniform(1.0, 1.4)
            dimensions = (x, x, z)

        x, y, z = dimensions

        top_thickness = uniform(0.5, 0.7)

        # straight has the bug that seat and legs are disjoint, so disable for now.

        # leg_style = choice(['straight', 'single_stand', 'wheeled'])
        leg_style = choice(["single_stand", "wheeled"])

        parameters = {
            "Top Profile Width": x,
            "Top Thickness": top_thickness,
            "Top Front Relative Width": uniform(0.5, 0.8),
            "Top Front Bent": uniform(-1.5, -0.4),
            "Top Seat Bent": uniform(-1.5, -0.4),
            "Top Mid Bent": uniform(-2.4, -0.5),
            "Top Mid Relative Width": uniform(0.5, 0.9),
            "Top Back Bent": uniform(-1, -0.1),
            "Top Back Relative Width": uniform(0.6, 0.9),
            "Top Mid Pos": uniform(0.4, 0.6),
            # 'Top Material': choice(['leather', 'wood', 'plastic', 'glass']),
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Style": leg_style,
            "Leg NGon": choice([4, 32]),
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": uniform(1.1, 1.3),
            "Leg Height": 1.0,
        }

        if leg_style == "single_stand":
            leg_number = 1
            leg_diameter = uniform(0.7 * x, 0.9 * x)

            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]

            parameters.update(
                {
                    "Leg Number": leg_number,
                    "Leg Diameter": leg_diameter,
                    "Leg Curve Control Points": leg_curve_ctrl_pts,
                    # 'Leg Material': choice(['metal', 'wood'])
                }
            )

        elif leg_style == "straight":
            leg_diameter = uniform(0.04, 0.06)
            leg_number = 4

            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, uniform(0.85, 0.95)),
                (1.0, uniform(0.4, 0.6)),
            ]

            parameters.update(
                {
                    "Leg Number": leg_number,
                    "Leg Diameter": leg_diameter,
                    "Leg Curve Control Points": leg_curve_ctrl_pts,
                    # 'Leg Material': choice(['metal', 'wood']),
                    "Strecher Relative Pos": uniform(0.2, 0.6),
                    "Strecher Increament": choice([0, 1, 2]),
                }
            )

        elif leg_style == "wheeled":
            leg_diameter = uniform(0.03, 0.05)
            leg_number = 1
            pole_number = choice([4, 5])
            joint_height = uniform(0.5, 0.8) * (z - top_thickness)
            wheel_arc_sweep_angle = uniform(120, 240)
            wheel_width = uniform(0.11, 0.15)
            wheel_rot = uniform(0, 360)
            pole_length = uniform(1.6, 2.0)

            parameters.update(
                {
                    "Leg Number": leg_number,
                    "Leg Pole Number": pole_number,
                    "Leg Diameter": leg_diameter,
                    "Leg Joint Height": joint_height,
                    "Leg Wheel Arc Sweep Angle": wheel_arc_sweep_angle,
                    "Leg Wheel Width": wheel_width,
                    "Leg Wheel Rot": wheel_rot,
                    "Leg Pole Length": pole_length,
                    # 'Leg Material': choice(['metal'])
                }
            )

        else:
            raise NotImplementedError

        return parameters, leg_style

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
            obj, geometry_assemble_chair, apply=True, input_kwargs=self.params
        )
        tagging.tag_system.relabel_obj(obj)

        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
