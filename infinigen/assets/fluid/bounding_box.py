# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan


from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def geometry_geometry_nodes(nw: NodeWrangler, obj):
    # Code generated using version 2.4.3 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": obj})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": object_info.outputs["Geometry"]}
    )

    object_info.transform_space = "RELATIVE"

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": bounding_box.outputs["Bounding Box"]},
    )


def apply(bounding_box, obj, selection=None, **kwargs):
    surface.add_geomod(
        bounding_box,
        geometry_geometry_nodes,
        selection=selection,
        attributes=[],
        input_kwargs=dict(obj=obj),
    )
