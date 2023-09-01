# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang


import random
import math

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

@node_utils.to_nodegroup('nodegroup_norm_value', singleton=False, type='GeometryNodeTree')
def nodegroup_norm_value(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Attribute', 0.0000),
            ('NodeSocketGeometry', 'Geometry', None)])
    
    attribute_statistic_1 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 2: group_input.outputs["Attribute"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Attribute"], 1: attribute_statistic_1.outputs["Min"]},
        attrs={'operation': 'SUBTRACT'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute_statistic_1.outputs["Max"], 1: attribute_statistic_1.outputs["Min"]},
        attrs={'operation': 'SUBTRACT'})
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: subtract_1}, attrs={'operation': 'DIVIDE'})
    
    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: divide}, attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2, 1: 2.0000}, attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Value': multiply}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_norm_vec', singleton=False, type='GeometryNodeTree')
def nodegroup_norm_vec(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketString', 'Name', ''),
            ('NodeSocketVector', 'Vector', (0.0000, 0.0000, 0.0000))])
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    normvalue = nw.new_node(nodegroup_norm_value().name,
        input_kwargs={'Attribute': separate_xyz_1.outputs["X"], 'Geometry': group_input.outputs["Geometry"]})
    
    normvalue_1 = nw.new_node(nodegroup_norm_value().name,
        input_kwargs={'Attribute': separate_xyz_1.outputs["Y"], 'Geometry': group_input.outputs["Geometry"]})
    
    normvalue_2 = nw.new_node(nodegroup_norm_value().name,
        input_kwargs={'Attribute': separate_xyz_1.outputs["Z"], 'Geometry': group_input.outputs["Geometry"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': normvalue, 'Y': normvalue_1, 'Z': normvalue_2})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Name': group_input.outputs["Name"], 2: combine_xyz},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': store_named_attribute}, attrs={'is_active_output': True})


def sample_range(x_min, x_max):
    y = random.random()
    y = y * (x_max - x_min) + x_min
    return y


def sample_ratio(x, sample_min=0.5, sample_max=2):
    if x == 0:
        return x
    neg = 1
    if x < 0:
        x = -x
        neg = -1
    x_min = x * sample_min
    x_max = x * sample_max
    exp = sample_range(math.log(x_min), math.log(x_max))
    return neg * math.exp(exp)


def clip(x, v_min=0, v_max=1):
    return max(min(x, v_max), v_min)


# sample a random rgb color
# if offset is not 0, the color is sampled from [color-offset, color+offset]
def sample_color(color, offset=0, keep_sum=False):
    if keep_sum:
        mean = (color[0]+color[1]+color[2])/3
        offset = min(mean, 1-mean)*random.random()
        idx = random.randint(0, 2)
        f = 1
        pcg = random.random()
        for i in range(3):
            if i == idx:
                color[i] = mean+offset
            else:
                color[i] = mean-offset*(f*pcg+(1-f)*(1-pcg))
                f = 0
        return
        
    for i in range(3):
        if offset == 0:
            color[i] = random.random()
        else:
            color[i] += (random.random() - 0.5) * 2 * offset
        color[i] = clip(color[i])

# generate a random voronoi offset
def geo_voronoi_noise(nw, rand=False, **input_kwargs):
    group_input = nw.new_node(Nodes.GroupInput)

    subdivide_mesh = nw.new_node('GeometryNodeSubdivideMesh',
                                 input_kwargs={'Mesh': group_input.outputs["Geometry"],
                                     'Level': input_kwargs.get('subdivide_mesh_level', 0)})

    position = nw.new_node(Nodes.InputPosition)

    scale = nw.new_node(Nodes.Value)
    scale.outputs["Value"].default_value = input_kwargs.get('scale', 2)

    vector_math = nw.new_node(Nodes.VectorMath, input_kwargs={0: position, 1: scale},
                              attrs={'operation': 'MULTIPLY'})

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Vector': vector_math.outputs["Vector"], 'Scale': 10.0})
    if rand:
        sample_max = input_kwargs['noise_scale_max'] if 'noise_scale_max' in input_kwargs else 3
        sample_min = input_kwargs['noise_scale_min'] if 'noise_scale_min' in input_kwargs else 1 / sample_max
        noise_texture.inputs["Scale"].default_value = sample_ratio(noise_texture.inputs["Scale"].default_value,
                                                                   sample_min, sample_max)

    mix = nw.new_node(Nodes.MixRGB, input_kwargs={'Fac': 0.8, 'Color1': noise_texture.outputs["Color"],
        'Color2': vector_math.outputs["Vector"]})
    if rand:
        mix.inputs["Fac"].default_value = sample_range(0.7, 0.9)

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': mix},
                                  attrs={'voronoi_dimensions': '4D'})
    if rand:
        sample_max = input_kwargs['voronoi_scale_max'] if 'voronoi_scale_max' in input_kwargs else 3
        sample_min = input_kwargs[
            'voronoi_scale_min'] if 'voronoi_scale_min' in input_kwargs else 1 / sample_max
        voronoi_texture.inputs["Scale"].default_value = sample_ratio(
            voronoi_texture.inputs["Scale"].default_value, sample_min, sample_max)
        voronoi_texture.inputs['W'].default_value = sample_range(-5, 5)

    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"]},
        attrs={'operation': 'SUBTRACT'})

    normal = nw.new_node(Nodes.InputNormal)

    vector_math_1 = nw.new_node(Nodes.VectorMath,
                                input_kwargs={0: subtract, 1: normal},
                                attrs={'operation': 'MULTIPLY'})

    offsetscale = nw.new_node(Nodes.Value)
    offsetscale.outputs["Value"].default_value = input_kwargs.get('offsetscale', 0.02)

    vector_math_2 = nw.new_node(Nodes.VectorMath,
                                input_kwargs={0: vector_math_1.outputs["Vector"], 1: offsetscale},
                                attrs={'operation': 'MULTIPLY'})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': subdivide_mesh,
        'Offset': vector_math_2.outputs["Vector"]})

    capture_attribute = nw.new_node(Nodes.CaptureAttribute, input_kwargs={'Geometry': set_position,
        1: voronoi_texture.outputs["Distance"]}, attrs={'data_type': 'FLOAT_VECTOR'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': capture_attribute.outputs["Geometry"],
                                   'Attribute': capture_attribute.outputs["Attribute"]})
