# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_material(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute, attrs={"attribute_name": "col"})

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": attribute.outputs["Color"]}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf}
    )


def geo_face_colors(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    random_value = nw.new_node(Nodes.RandomValue, attrs={"data_type": "FLOAT"})

    combine = nw.new_node(
        Nodes.CombineColor,
        input_kwargs={
            0: random_value.outputs["Value"],
            1: 1.0,
            2: 0.4,
        },
        attrs={"mode": "HSV"},
    )

    geo = nw.new_node(Nodes.Triangulate, input_kwargs={"Mesh": group_input})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": geo,
            "Name": "col",
            "Value": combine,
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "FACE"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": store_named_attribute}
    )


class FaceSizeVisualizer:
    def apply(self, obj, selection=None, **kwargs):
        if not isinstance(obj, list):
            obj = [obj]

        obj = [o for o in obj if o.type == "MESH" and not o.hide_viewport]

        for o in obj:
            bpy.context.view_layer.objects.active = o
            for _ in range(len(o.material_slots)):
                bpy.ops.object.material_slot_remove()
            for m in o.modifiers:
                o.modifiers.remove(m)

        surface.add_geomod(obj, geo_face_colors, selection=selection, attributes=[])
        surface.add_material(obj, shader_material, selection=selection)
