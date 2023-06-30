# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Date Signed: May 30, 2023

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from nodes.color import color_category
from surfaces import surface

def shader_material(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'col'})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': attribute.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_face_colors(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    random_value = nw.new_node(Nodes.RandomValue,
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Name': 'col', 2: random_value.outputs["Value"]},
        attrs={'data_type': 'FLOAT_VECTOR', 'domain': 'FACE'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': store_named_attribute})



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_face_colors, selection=selection, attributes=[])
    surface.add_material(obj, shader_material, selection=selection)