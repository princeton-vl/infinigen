import bpy
import bpy
import mathutils
from numpy.random import uniform, normal, randint, choice

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.utils.object import new_bbox

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.shelves.kitchen_cabinet import KitchenCabinetFactory
from infinigen.assets.table_decorations.sink import SinkFactory
from infinigen.assets.wall_decorations.range_hood import RangeHoodFactory

from infinigen.core.util import blender as butil

from infinigen.assets.tables.table_top import nodegroup_generate_table_top

def nodegroup_tag_cube(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        
    index = nw.new_node(Nodes.Index)
    
    equal = nw.new_node(Nodes.Compare, input_kwargs={2: index, 3: 5}, attrs={'data_type': 'INT', 'operation': 'EQUAL'})
    
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': cube}, attrs={'is_active_output': True})

def geometry_nodes_add_cabinet_top(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0500
    
    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': group_input.outputs["Geometry"]})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': bounding_box.outputs["Max"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': bounding_box.outputs["Min"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: 1.4140}, attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: separate_xyz.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: subtract}, attrs={'operation': 'DIVIDE'})
    
    generatetabletop = nw.new_node(nodegroup_generate_table_top().name,
        input_kwargs={'Thickness': value, 'N-gon': 4, 'Profile Width': multiply, 'Aspect Ratio': divide, 'Fillet Ratio': 0.0100, 'Fillet Radius Vertical': 0.0100})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': generatetabletop, 'Material': surface.shaderfunc_to_material(shader_marble)})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: separate_xyz_1.outputs["Y"]})
    
    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': bounding_box.outputs["Max"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': divide_1, 'Z': separate_xyz_2.outputs["Z"]})
    
    transform_geometry = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': set_material, 'Translation': combine_xyz})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [group_input.outputs["Geometry"], transform_geometry]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

def geometry_node_to_tagged_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': group_input.outputs["Geometry"]})

    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': bounding_box, 'Scale': (0.9700, 0.9700, 1.000)})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_geometry}, attrs={'is_active_output': True})

def geometry_node_to_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': group_input.outputs["Geometry"]})

    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': bounding_box, 'Scale': (0.9700, 0.9700, 1.000)})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_geometry}, attrs={'is_active_output': True})

class KitchenSpaceFactory(AssetFactory):
        super(KitchenSpaceFactory, self).__init__(factory_seed, coarse=coarse)


            self.params = self.sample_parameters(dimensions)


    def sample_parameters(self, dimensions):
        self.cabinet_bottom_height = uniform(0.8, 1.0)
        self.cabinet_top_height = uniform(0.8, 1.0)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:

        return box

    def create_asset(self, **params):
        x, y, z = self.dimensions
        cabinet_bottom_height = self.cabinet_bottom_height
        cabinet_top_height = self.cabinet_top_height
        
        cabinet_bottom_factory = KitchenCabinetFactory(self.factory_seed, dimensions=(x, y-0.15, cabinet_bottom_height), drawer_only=True)
        cabinet_bottom = cabinet_bottom_factory(i=0)

        surface.add_geomod(cabinet_bottom, geometry_nodes_add_cabinet_top, apply=True)







        butil.apply_transform(kitchen_space)

        return kitchen_space
