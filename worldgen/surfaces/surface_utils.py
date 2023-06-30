import random
import math
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from nodes.color import color_category
from surfaces import surface

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
    return y

def sample_ratio(x, sample_min=0.5, sample_max=2):
    if x == 0:
        return x
    neg = 1
    if x < 0:
        x = -x
        neg = -1
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
        color[i] = clip(color[i])

# generate a random voronoi offset
def geo_voronoi_noise(nw, rand=False, **input_kwargs):
    group_input = nw.new_node(Nodes.GroupInput)
    subdivide_mesh = nw.new_node('GeometryNodeSubdivideMesh',
    position = nw.new_node(Nodes.InputPosition)
    scale = nw.new_node(Nodes.Value)
    scale.outputs["Value"].default_value = input_kwargs.get('scale', 2)
    noise_texture = nw.new_node(Nodes.NoiseTexture,
    if rand:
        sample_max = input_kwargs['noise_scale_max'] if 'noise_scale_max' in input_kwargs else 3
    if rand:
        mix.inputs["Fac"].default_value = sample_range(0.7, 0.9)
    if rand:
        sample_max = input_kwargs['voronoi_scale_max'] if 'voronoi_scale_max' in input_kwargs else 3
        voronoi_texture.inputs['W'].default_value = sample_range(-5, 5)

    normal = nw.new_node(Nodes.InputNormal)
    vector_math_1 = nw.new_node(Nodes.VectorMath,
    offsetscale = nw.new_node(Nodes.Value)
    offsetscale.outputs["Value"].default_value = input_kwargs.get('offsetscale', 0.02)
    vector_math_2 = nw.new_node(Nodes.VectorMath,
    group_output = nw.new_node(Nodes.GroupOutput,
