# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Lingjie Mei

from numpy.random import normal as N

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category

from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory

from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object

def shader_material(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo_Shader)
    
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': object_info.outputs["Random"]})
    colorramp.color_ramp.elements[0].position = 0.0000
    colorramp.color_ramp.elements[0].color = color_category('pine_needle')
    colorramp.color_ramp.elements[1].position = 1.0000
    colorramp.color_ramp.elements[1].color = color_category('pine_needle')

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': colorramp})

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_pine_needle', singleton=False, type='GeometryNodeTree')
def nodegroup_pine_needle(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Scale', 0.0400),
            ('NodeSocketFloat', 'Bend', 0.0300),
            ('NodeSocketFloatDistance', 'Radius', 0.0010)])
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (-1.0000, 0.0000, 0.0000), 'Scale': group_input.outputs["Scale"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.0000, 1.0000, 0.0000), 'Scale': group_input.outputs["Bend"]},
        attrs={'operation': 'SCALE'})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (1.0000, 0.0000, 0.0000), 'Scale': group_input.outputs["Scale"]},
        attrs={'operation': 'SCALE'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 5, 'Start': scale.outputs["Vector"], 'Middle': scale_1.outputs["Vector"], 'End': scale_2.outputs["Vector"]})
    
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': 6, 'Radius': group_input.outputs["Radius"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': quadratic_bezier, 'Profile Curve': curve_circle.outputs["Curve"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': curve_to_mesh}, attrs={'is_active_output': True})

class PineNeedleFactory(AssetFactory):

    def sample_params(self):
        s = N(1, 0.2)
        return {
            'Scale': 0.04 * s,
            'Bend': 0.03 * s * N(1, 0.2),
            'Radius': 0.001 * s * N(1, 0.2)
        }

    def create_asset(self, **_):
        obj = butil.spawn_vert('pine_needle')
        butil.modify_mesh(obj, 'NODES', apply=True, node_group=nodegroup_pine_needle(), 
                          ng_inputs=self.sample_params())
        tag_object(obj, 'pine_needle')
        return obj

    def finalize_assets(self, objs):
        surface.add_material(objs, shader_material)