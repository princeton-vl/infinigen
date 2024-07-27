# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan


from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def duplicate(nw: NodeWrangler, obj):
    # Code generated using version 2.4.3 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": obj})

    object_info.transform_space = "RELATIVE"

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": object_info.outputs["Geometry"],
            "Location": object_info.outputs["Location"],
            "Rotation": object_info.outputs["Rotation"],
            "Scale": object_info.outputs["Scale"],
        },
    )


def apply(new_obj, old_obj, selection=None, **kwargs):
    surface.add_geomod(
        new_obj,
        duplicate,
        selection=selection,
        attributes=["Location", "Rotation", "Scale"],
        input_kwargs=dict(obj=old_obj),
    )
