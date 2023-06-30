import bpy
import numpy as np

from . import helper, mesh
from .materials import new_link

C = bpy.context
D = bpy.data

def add_node_modifier(obj):
  # Add geometry node modifier
  helper.set_active_obj(obj)
  # bpy.ops.node.new_geometry_nodes_modifier() # Blender 3.2
  bpy.ops.object.modifier_add(type='NODES') # Blender 3.1
  return obj.modifiers[-1]


def setup_inps(ng, inp, nodes):
  for k_idx, (k, node, attr) in enumerate(nodes):
    new_link(ng, inp, k_idx, node, attr)
    ng.inputs[k_idx].name = k


    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
                      ('NodeSocketBool', 'Selection', True),
                      ('NodeSocketCollection', 'Collection', None),
                      ('NodeSocketInt', 'Multi inst', 1),
                      ('NodeSocketFloat', 'Density', 0.5),
                      ('NodeSocketFloat', 'Min scale', 0.0),
                      ('NodeSocketFloat', 'Max scale', 1.0),
                      ('NodeSocketFloat', 'Pitch scaling', 0.2),
                      ('NodeSocketFloat', 'Pitch offset', 0.0),
                      ('NodeSocketFloat', 'Pitch variance', 0.4),
                      ('NodeSocketFloat', 'Yaw variance', 0.4),

    mesh_to_curve = nw.new_node('GeometryNodeMeshToCurve',
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Selection': group_input.outputs["Selection"]})

    curve_to_points = nw.new_node('GeometryNodeCurveToPoints',
        input_kwargs={'Curve': mesh_to_curve, 'Count': group_input.outputs["Multi inst"]})

    mesh_to_points = nw.new_node('GeometryNodeMeshToPoints',
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Selection': group_input.outputs["Selection"]})

    position = nw.new_node(Nodes.InputPosition)

    transfer_attribute = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={'Source': mesh_to_points, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR', 'mapping': 'NEAREST'})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_to_points.outputs["Points"], 'Position': transfer_attribute.outputs["Attribute"]})

    random_value = nw.new_node(Nodes.RandomValue)

    math = nw.new_node(Nodes.Math,
        input_kwargs={0: random_value.outputs[1], 1: group_input.outputs["Density"]},
        attrs={'operation': 'LESS_THAN'})
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': curve_to_points.outputs["Rotation"]})

    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 1.5708})

    math_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: group_input.outputs["Pitch scaling"]},
        attrs={'operation': 'MULTIPLY'})

    math_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_2, 1: group_input.outputs["Pitch offset"]})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': math_3, 'Z': separate_xyz.outputs["Z"]})

    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Pitch variance"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})

    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: math_4, 3: group_input.outputs["Pitch variance"]})

    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Yaw variance"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})

    random_value_2 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: math_5, 3: group_input.outputs["Yaw variance"]})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': random_value_1.outputs[1], 'Z': random_value_2.outputs[1]})

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: combine_xyz_1})

    random_value_3 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: group_input.outputs["Min scale"], 3: group_input.outputs["Max scale"]})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points})

    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["Realize Instance"], 14: instance_on_points, 15: realize_instances})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': switch.outputs[6]})


def phyllotaxis_distribute(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
                      ('NodeSocketInt', 'Count', 50),
                      ('NodeSocketFloat', 'Max radius', 2.0),
                      ('NodeSocketFloat', 'Radius exp', 0.5),
                      ('NodeSocketFloat', 'Inner pct', 0.0),
                      ('NodeSocketFloat', 'Min angle', -0.5236),
                      ('NodeSocketFloat', 'Max angle', 0.7854),
                      ('NodeSocketFloat', 'Min scale', 0.3),
                      ('NodeSocketFloat', 'Max scale', 0.3),
                      ('NodeSocketFloat', 'Min z', 0.0),
                      ('NodeSocketFloat', 'Max z', 1.0),
                      ('NodeSocketFloat', 'Clamp z', 1.0),
                      ('NodeSocketFloat', 'Yaw offset', -np.pi / 2)])

    mesh_line = nw.new_node('GeometryNodeMeshLine',
        input_kwargs={'Count': group_input.outputs["Count"]})

    mesh_to_points = nw.new_node('GeometryNodeMeshToPoints',
        input_kwargs={'Mesh': mesh_line})

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': mesh_to_points, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})

    index = nw.new_node('GeometryNodeInputIndex')

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0

    math = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: value},
        attrs={'operation': 'DIVIDE'})

    math_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: math},
        attrs={'operation': 'FLOOR'})

    math_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: 2.3998},
        attrs={'operation': 'MULTIPLY'})

    math_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: math},
        attrs={'operation': 'FRACT'})

    math_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_2, 1: 6.2832},
        attrs={'operation': 'MULTIPLY'})

    math_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_6, 1: math_5})

    math_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7},
        attrs={'operation': 'COSINE'})

    math_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7},
        attrs={'operation': 'SINE'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': math_8, 'Y': math_9})

    math_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Count"], 1: value},
        attrs={'operation': 'DIVIDE'})

    math_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_1, 1: math_3},
        attrs={'operation': 'DIVIDE'})

    math_10 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_4, 1: group_input.outputs["Radius exp"]},
        attrs={'operation': 'POWER'})

    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': math_10, 3: group_input.outputs["Inner pct"]})

    math_11 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: group_input.outputs["Max radius"]},
        attrs={'operation': 'MULTIPLY'})

    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': math_4, 3: 1.5708, 4: 1.5708})

    math_12 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"]},
        attrs={'operation': 'SINE'})

    math_13 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_11, 1: math_12},
        attrs={'operation': 'MULTIPLY'})

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: math_13},
        attrs={'operation': 'MULTIPLY'})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': vector_math.outputs["Vector"]})

    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': math_4, 2: group_input.outputs["Clamp z"], 3: group_input.outputs["Min z"], 4: group_input.outputs["Max z"]})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': map_range_2.outputs["Result"]})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': combine_xyz_1})

    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 2: map_range.outputs["Result"]})

    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': map_range.outputs["Result"], 1: attribute_statistic.outputs["Max"], 2: attribute_statistic.outputs["Min"], 3: group_input.outputs["Min angle"], 4: group_input.outputs["Max angle"]})

    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: -0.1, 3: 0.1})

    math_14 = nw.new_node(Nodes.Math,
        input_kwargs={0: math_7, 1: group_input.outputs["Yaw offset"]})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': map_range_3.outputs["Result"], 'Y': random_value_1.outputs[1], 'Z': math_14})

    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: group_input.outputs["Min scale"], 3: group_input.outputs["Max scale"]})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': set_position, 'Instance': group_input.outputs["Geometry"], 'Rotation': combine_xyz_2, 'Scale': random_value.outputs[1]})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': instance_on_points})


def follow_curve(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
                      ('NodeSocketGeometry', 'Curve', None),
                      ('NodeSocketFloat', 'Offset', 0.5)])

    position = nw.new_node(Nodes.InputPosition)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute.outputs["Attribute"]})

    math = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["Offset"]})

    sample_curve = nw.new_node('GeometryNodeSampleCurve',
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Length': math})

    vector_math = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Tangent"], 1: sample_curve.outputs["Normal"]},
        attrs={'operation': 'CROSS_PRODUCT'})

    vector_math_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math.outputs["Vector"], 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})

    vector_math_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Normal"], 'Scale': separate_xyz.outputs["Y"]},
        attrs={'operation': 'SCALE'})

    vector_math_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_math_1.outputs["Vector"], 1: vector_math_2.outputs["Vector"]})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': sample_curve.outputs["Position"], 'Offset': vector_math_3.outputs["Vector"]})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

def set_tree_radius(nw):
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'Selection': group_input.outputs["Selection"]})
        input_kwargs={0: group_input.outputs["Reverse depth"], 1: group_input.outputs["Scaling"]},
        attrs={'operation': 'MULTIPLY'})
        attrs={'operation': 'MULTIPLY'})
        attrs={'operation': 'POWER'})
        attrs={'operation': 'MAXIMUM'})
        attrs={'operation': 'MINIMUM'})
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Profile res"]})
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})
        input_kwargs={'Geometry': curve_to_mesh, 'Shade Smooth': False})
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': set_shade_smooth, 'Distance': group_input.outputs["Merge dist"]})
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': merge_by_distance})

def create_berry(sphere):
  # Create a sphere
  phyllotaxis_distribute('berry', sphere,
                         min_radius_pct=0, max_radius=1,
                         sin_max=2.5, sin_clamp_max=.8,
                         z_max=.8, z_clamp=.7)


def sample_points_and_normals(obj, max_density=3,
                              surface_dist=1, max_points=10000):
  # Need to instantiate point distribute
  m = add_node_modifier(obj)
  ng = m.node_group
  inp = ng.nodes.get('Group Input')
  out = ng.nodes.get('Group Output')
  dist = ng.nodes.new(type='GeometryNodeDistributePointsOnFaces')
  pos = ng.nodes.new('GeometryNodeInputPosition')
  scale_factor = ng.nodes.new('ShaderNodeValue')
  mult_normal = ng.nodes.new('ShaderNodeVectorMath')
  add_pos = ng.nodes.new('ShaderNodeVectorMath')
  set_pos = ng.nodes.new('GeometryNodeSetPosition')
  to_vtx = ng.nodes.new('GeometryNodePointsToVertices')

  new_link(ng, inp, 'Geometry', dist, 'Mesh')
  new_link(ng, dist, 'Normal', mult_normal, 0)
  new_link(ng, scale_factor, 0, mult_normal, 1)
  new_link(ng, pos, 0, add_pos, 0)
  new_link(ng, mult_normal, 0, add_pos, 1)
  new_link(ng, dist, 'Points', set_pos, 'Geometry')
  new_link(ng, add_pos, 0, set_pos, 'Position')
  new_link(ng, set_pos, 'Geometry', to_vtx, 'Points')
  new_link(ng, to_vtx, 'Mesh', out, 'Geometry')

  mult_normal.operation = 'MULTIPLY'
  scale_factor.outputs[0].default_value = surface_dist
  dist.distribute_method = 'POISSON'
  dist.inputs.get('Density Max').default_value = max_density

  # Get point coordinates
  dgraph = C.evaluated_depsgraph_get()
  obj_eval = obj.evaluated_get(dgraph)
  vtx = mesh.vtx2cds(obj_eval.data.vertices, obj_eval.matrix_world)

  # Get normals
  scale_factor.outputs[0].default_value = 1
  for l in ng.links:
    if l.from_node == pos:
      ng.links.remove(l)

  dgraph = C.evaluated_depsgraph_get()
  obj_eval = obj.evaluated_get(dgraph)
  normals = mesh.vtx2cds(obj_eval.data.vertices, np.eye(4))

  obj.modifiers.remove(obj.modifiers[-1])
  D.node_groups.remove(ng)

  idxs = mesh.subsample_vertices(vtx, max_num=max_points)
  return vtx[idxs], normals[idxs]

