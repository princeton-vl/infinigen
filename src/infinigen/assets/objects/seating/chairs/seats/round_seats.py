# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from numpy.random import uniform

from infinigen.assets.objects.tables.table_top import nodegroup_capped_cylinder
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "generate_round_seats", singleton=False, type="GeometryNodeTree"
)
def generate_round_seats(
    nw: NodeWrangler,
    thickness=None,
    radius=None,
    cap_radius=None,
    bevel_factor=None,
    seat_material=None,
):
    # Code generated using version 2.6.4 of the node_transpiler
    if thickness is None:
        thickness = uniform(0.05, 0.12)
    if radius is None:
        radius = uniform(0.35, 0.45)
    if cap_radius is None:
        cap_radius = uniform(2.0, 3.2)
    if bevel_factor is None:
        bevel_factor = uniform(0.01, 0.04)

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: thickness, 1: 1.0}, attrs={"operation": "MULTIPLY"}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: bevel_factor, 1: thickness},
        attrs={"operation": "DIVIDE"},
    )

    cappedcylinder = nw.new_node(
        nodegroup_capped_cylinder().name,
        input_kwargs={
            "Thickness": multiply,
            "Radius": radius,
            "Cap Flatness": cap_radius,
            "Fillet Radius Vertical": divide,
            "Cap Relative Scale": 0.0140,
            "Cap Relative Z Offset": -0.0020,
            "Resolution": 128,
        },
    )

    seat = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cappedcylinder, "Material": seat_material},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": seat},
        attrs={"is_active_output": True},
    )
