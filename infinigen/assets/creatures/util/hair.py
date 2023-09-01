# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=dCIKH649gac by Hey Pictures

import pdb
import warnings
import logging

import bpy
import bmesh
import mathutils

import numpy as np
from scipy.spatial import KDTree

from infinigen.core.util import blender as butil
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes
from infinigen.core.nodes import node_utils

from infinigen.assets.creatures.util.nodegroups.hair import nodegroup_comb_direction, nodegroup_duplicate_to_clumps, \
    nodegroup_hair_position, nodegroup_comb_hairs, nodegroup_strand_noise, nodegroup_hair_length_rescale, \
    nodegroup_snap_roots_to_surface

logger = logging.getLogger(__name__)

def add_hair_particles(obj, params, props):

    _, mod = butil.modify_mesh(obj, 'PARTICLE_SYSTEM', apply=False, return_mod=True)

    settings = mod.particle_system.settings
    settings.type = 'HAIR'
    for k, v in params.items():
        setattr(settings, k, v)

    for k, v in props.items():
        setattr(mod.particle_system, k, v)

def as_hair_bsdf(mat, hair_bsdf_params):
    
    assert mat.use_nodes

    new_mat = mat.copy()
    new_mat.name = f'as_hair_bsdf({mat.name})'
    ng = new_mat.node_tree

    child = lambda inp: next(link.from_node for link in ng.links if link.to_socket == inp)

    try:
        out = ng.nodes['Material Output']
        shader = child(out.inputs['Surface'])
        rgb = child(shader.inputs['Base Color'])
    except StopIteration:
        # shader didnt match expected structure, abort and use original shader
        warnings.warn(f'as_hair_bsdf failed for {mat.name=}, did not match expected structure')
        return new_mat

    nw = NodeWrangler(ng)
    hair_bsdf = nw.new_node(Nodes.PrincipledHairBSDF, input_kwargs={'Color': rgb, **hair_bsdf_params})
    nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': hair_bsdf})

    return new_mat

def compute_hair_placement_vertgroup(obj, root, avoid_features_dist):

    avoid_types = ['Eyeball', 'Teeth', 'Tongue']#, 'Nose']
    extras = [o for o in butil.iter_object_tree(root) if 'extra' in o.name]
    avoid_extras = [o for o in extras if any(n in o.name for n in avoid_types)]
    
    avoid_verts = []
    for o in avoid_extras:
        for v in o.data.vertices:
            avoid_verts.append(o.matrix_world @ v.co)
    avoid_verts = np.array(avoid_verts).reshape(-1, 3)

    verts = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    if len(avoid_verts):
        kd = KDTree(avoid_verts)
        dists, _ = kd.query(verts, k=1)
    else:
        dists = np.full(len(verts), 1e5)

    tag_bald_mask = np.zeros(len(verts), dtype=np.float32)
    if 'tag_bald' in obj.data.attributes:
        obj.data.attributes['tag_bald'].data.foreach_get('value', tag_bald_mask)

    idxs = np.where((dists > avoid_features_dist) & (tag_bald_mask < 0.5))[0]

    group = obj.vertex_groups.new(name='hair_placement')
    group.add(idxs.tolist(), 1.0, 'ADD') # .tolist() necessary to avoid np.int64 type error

    return group
    
@node_utils.to_nodegroup('nodegroup_decode_noise', singleton=True, type='GeometryNodeTree')
def nodegroup_decode_noise(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'MinMaxScale', (0.0, 0.0, 0.0)),
            ('NodeSocketGeometry', 'Source', None),
            ('NodeSocketVector', 'Source Position', (0.0, 0.0, 0.0))])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["MinMaxScale"]})
    
    noise_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Scale': separate_xyz.outputs["Z"], 'Detail': 5.0})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 3: separate_xyz.outputs["X"], 4: separate_xyz.outputs["Y"]})
    
    transfer_attribute = nw.new_node(Nodes.SampleNearestSurface,
        input_kwargs={
            'Mesh': group_input.outputs["Source"], 
            'Value': map_range_1.outputs["Result"], 
            'Sample Position': group_input.outputs["Source Position"]
        })

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Attribute': (transfer_attribute, 'Value')})

@node_utils.to_nodegroup('nodegroup_hair_grooming', singleton=True, type='GeometryNodeTree')
def nodegroup_hair_grooming(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketObject', 'Object', None),
            ('NodeSocketVector', 'Length MinMaxScale', (0.014, 0.04, 40.0)),
            ('NodeSocketVector', 'Puff MinMaxScale', (0.14, 0.40, 40.0)),
            ('NodeSocketFloat', 'Combing', 0.0),
            ('NodeSocketFloat', 'Strand Random Mag', 0.001),
            ('NodeSocketFloat', 'Strand Perlin Mag', 0.05),
            ('NodeSocketFloat', 'Strand Perlin Scale', 33.38),
            ('NodeSocketInt', 'Tuft Amount', 1),
            ('NodeSocketFloat', 'Tuft Spread', 0.005),
            ('NodeSocketFloat', 'Tuft Clumping', 0.5),
            ('NodeSocketFloat', 'Root Radius', 0.01),
            ('NodeSocketFloat', 'Post Clump Noise Mag', 0.0),
            ('NodeSocketFloat', 'Hair Length Pct Min', 0.7)])
    
    hairposition = nw.new_node(nodegroup_hair_position().name,
        input_kwargs={'Curves': group_input.outputs["Geometry"]})
    
    object_info = nw.new_node(Nodes.ObjectInfo,
        input_kwargs={'Object': group_input.outputs["Object"]})
    
    combdirection = nw.new_node(nodegroup_comb_direction().name,
        input_kwargs={'Surface': object_info.outputs["Geometry"], 'Root Positiion': hairposition.outputs["Root Position"]})
    
    decode_length = nw.new_node(nodegroup_decode_noise().name,
        input_kwargs={'MinMaxScale': group_input.outputs["Length MinMaxScale"], 'Source': object_info.outputs["Geometry"], 'Source Position': hairposition.outputs["Root Position"]},
        label='Decode Length')
    
    decode_puff = nw.new_node(nodegroup_decode_noise().name,
        input_kwargs={'MinMaxScale': group_input.outputs["Puff MinMaxScale"], 'Source': object_info.outputs["Geometry"], 'Source Position': hairposition.outputs["Root Position"]},
        label='Decode Puff')
    
    combhairs = nw.new_node(nodegroup_comb_hairs().name,
        input_kwargs={'Curves': group_input.outputs["Geometry"], 'Root Position': hairposition.outputs["Root Position"], 'Comb Dir': combdirection.outputs["Combing Direction"], 'Surface Normal': combdirection.outputs["Surface Normal"], 'Length': decode_length, 'Puiff': group_input.outputs["Combing"], 'Comb': decode_puff})
    
    strandnoise = nw.new_node(nodegroup_strand_noise().name,
        input_kwargs={'Geometry': combhairs, 'Random Mag': group_input.outputs["Strand Random Mag"], 'Perlin Mag': group_input.outputs["Strand Perlin Mag"], 'Perlin Scale': group_input.outputs["Strand Perlin Scale"]})
    
    duplicatetoclumps = nw.new_node(nodegroup_duplicate_to_clumps().name,
        input_kwargs={'Geometry': strandnoise, 'Surface Normal': combdirection.outputs["Surface Normal"], 'Amount': group_input.outputs["Tuft Amount"], 'Tuft Spread': group_input.outputs["Tuft Spread"], 'Tuft Clumping': group_input.outputs["Tuft Clumping"]})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: (-1.0, -1.0, -1.0)},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: random_value.outputs["Value"], 'Scale': group_input.outputs["Post Clump Noise Mag"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': duplicatetoclumps, 'Offset': scale.outputs["Vector"]})
    
    hairlengthrescale = nw.new_node(nodegroup_hair_length_rescale().name,
        input_kwargs={'Curves': set_position, 'Min': group_input.outputs['Hair Length Pct Min']})
    
    snaprootstosurface = nw.new_node(nodegroup_snap_roots_to_surface().name,
        input_kwargs={'Target': object_info.outputs["Geometry"], 'Curves': hairlengthrescale})

    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: group_input.outputs["Root Radius"], 4: 0.0})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': snaprootstosurface, 'Radius': map_range.outputs["Result"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_curve_radius})

def mat_attr_dependencies(node_tree):
    attrs = set()
    for node in node_tree.nodes:
        if node.bl_idname == Nodes.Attribute:
            attrs.add(node.attribute_name)
        elif node.bl_idname == "ShaderNodeGroup":
            attrs = attrs | mat_attr_dependencies(node.node_tree)

    return attrs

def geo_transfer_hair_attributes(nw, obj, attrs):

    group_input = nw.new_node(Nodes.GroupInput)

    hairposition = nw.new_node(nodegroup_hair_position().name,
        input_kwargs={'Curves': group_input.outputs["Geometry"]})

    object_info = nw.new_node(Nodes.ObjectInfo,
        input_kwargs={'Object': obj})

    attrs_out = {}
    for attr_name in attrs:
        if not attr_name in obj.data.attributes:
            logger.warn(f'Attempted to geo_transfer_hair_attributes() including {attr_name=} which is not present on {obj=}. Available are {list(obj.data.attributes.keys())}')
            continue

        obj_attr = obj.data.attributes[attr_name]

        named_attr = nw.new_node(Nodes.NamedAttribute,
            attrs={'data_type': obj_attr.data_type},
            input_kwargs={'Name': attr_name})
        transfer = nw.new_node(Nodes.SampleNearestSurface, 
            attrs={'data_type': obj_attr.data_type},
            input_kwargs={
                'Mesh': object_info.outputs['Geometry'], 
                "Value": named_attr, 
                'Sample Position': hairposition
            })
        attrs_out[attr_name] = (transfer, 'Value')

    nw.new_node(Nodes.GroupOutput, input_kwargs={
        'Geometry': group_input.outputs['Geometry'], **attrs_out})

def configure_hair(obj, root, hair_genome: dict, apply=True, is_dynamic=None):

    if is_dynamic is None:
        is_dynamic = any(m.type == 'ARMATURE' for m in obj.modifiers)

    # re-parameterize density params
    sa = butil.surface_area(obj)
    count = int(sa * hair_genome['density'])
    n_guide_hairs = count // hair_genome['clump_n']
    hair_genome['grooming']['Tuft Amount'] = hair_genome['clump_n']

    logger.debug(f'Computing hair placement vertex group')
    avoid_group = compute_hair_placement_vertgroup(obj, root, 
        avoid_features_dist=hair_genome['avoid_features_dist'])

    logger.debug(f'Add particle system with {n_guide_hairs=}')
    add_hair_particles(obj, params={'count': n_guide_hairs}, 
        props={'vertex_group_density': avoid_group.name})

    logger.debug(f'Converting particles to curves')
    with butil.SelectObjects(obj):
        for m in obj.modifiers:
            if m.type == 'PARTICLE_SYSTEM':
                m.show_viewport = True
        bpy.ops.curves.convert_from_particle_system()
        curves = bpy.context.active_object

    with butil.SelectObjects(obj):
        bpy.ops.object.particle_system_remove()

    logger.debug(f'Performing geonodes hair grooming')
    with butil.DisableModifiers(obj):
        _, mod = butil.modify_mesh(curves, 'NODES', apply=False, return_mod=True)
        mod.node_group = nodegroup_hair_grooming()
        surface.set_geomod_inputs(mod, {'Object': obj, **hair_genome['grooming']})

        if apply:
            butil.apply_modifiers(curves, mod=mod)
    
    curves.parent = obj
    curves.matrix_parent_inverse = obj.matrix_world.inverted() # keep prexisting transform
    curves.data.surface = obj

    if len(obj.material_slots) == 0:
        return

    if obj.active_material is not None:
        
        hair_mat = as_hair_bsdf(obj.active_material, hair_genome['material'])
        
        logger.debug(f'Transfer material attr dependencies from surf to curves')
        attr_deps = mat_attr_dependencies(hair_mat.node_tree)
        attr_deps = [a for a in attr_deps if a in obj.data.attributes]
        surface.add_geomod(curves, geo_transfer_hair_attributes, apply=apply,
            input_kwargs=dict(obj=obj, attrs=attr_deps), attributes=attr_deps)
        curves.active_material = hair_mat

    if is_dynamic:
        attach_hair_to_surface(curves, obj)

    curves.name = obj.name + '.hair_curves'

    return curves

@node_utils.to_nodegroup('nodegroup_transfer_uvs_to_curves_vec3', singleton=True)
def nodegroup_transfer_uvs_to_curves_vec3(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketObject', 'Object', None),
            ('NodeSocketString', 'from_uv', None),
            ('NodeSocketString', 'to_attr', None)])
    
    object_info = nw.new_node(Nodes.ObjectInfo,
        input_kwargs={'Object': group_input.outputs["Object"]},
        attrs={'transform_space': 'RELATIVE'})
    obj = object_info.outputs["Geometry"]
    
    uv = nw.new_node(Nodes.NamedAttribute,
        input_kwargs={'Name': group_input.outputs["from_uv"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    capture = nw.new_node(Nodes.CaptureAttribute, 
                     input_kwargs={'Geometry': obj, 'Value': uv},
                     attrs={'data_type': 'FLOAT_VECTOR', 'domain': 'FACE'})
    
    root_pos = nw.new_node(nodegroup_hair_position().name, [group_input.outputs['Geometry']])
    transfer_attribute = nw.new_node(Nodes.TransferAttribute,
        input_kwargs={
            'Source': capture.outputs['Geometry'], 
            1: capture.outputs["Attribute"],
            #'Source Position': root_pos
        },
        attrs={'data_type': 'FLOAT_VECTOR'})#, 'mapping': 'NEAREST'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={
            'Geometry': group_input.outputs["Geometry"], 
            'Name': group_input.outputs['to_attr'], 
            'Value': transfer_attribute.outputs["Attribute"]},
        attrs={'data_type': 'FLOAT_VECTOR', 'domain': 'CURVE'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': store_named_attribute})

def transfer_uvs_to_curves(curves, target, uv_name):

    # blender doesnt seem to support writing directly to FLOAT2 uv attributes.
    # lets write to a FLOAT_VECTOR then change it over to a FLOAT2

    curve_uv_attr = 'surface_uv_coordinate'
    butil.modify_mesh(curves, 'NODES', node_group=nodegroup_transfer_uvs_to_curves_vec3(), 
        ng_inputs={'Object': target, 'from_uv': uv_name, 'to_attr': curve_uv_attr}, apply=True)

    # rip uvs to np array
    n = len(curves.data.curves)
    uvs = np.empty(3 * n, dtype=np.float32)
    attr = curves.data.attributes[curve_uv_attr]
    assert attr.domain == 'CURVE' and attr.data_type == 'FLOAT_VECTOR'
    attr.data.foreach_get('vector', uvs)
    curves.data.attributes.remove(attr)

    # write back as FLOAT2
    uvs = uvs.reshape(n, 3)[:, :2].reshape(-1)
    attr = curves.data.attributes.new(curve_uv_attr, type='FLOAT2', domain='CURVE')
    attr.data.foreach_set('vector', uvs)

@node_utils.to_nodegroup('nodegroup_deform_curves_on_surface', singleton=True)
def nodegroup_deform_curves_on_surface(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    deform_curves_on_surface = nw.new_node('GeometryNodeDeformCurvesOnSurface',
        input_kwargs={'Curves': group_input.outputs["Geometry"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': deform_curves_on_surface})


def attach_hair_to_surface(curves, target):

    # target object needs UVMap and rest_position attribute,
    # curves obj needs surface_uv_coordinate attribute
    # defined in https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/curve/deform_curves_on_surface.html

    surface.write_attribute(target, lambda nw: nw.new_node(Nodes.InputPosition), 'rest_position', apply=True)
    with butil.ViewportMode(target, mode='EDIT'):
        bpy.ops.mesh.select_all(action='SELECT')        
        bpy.ops.uv.smart_project(island_margin=0.03)
    assert len(target.data.uv_layers) > 0

    curves.data.surface = target
    curves.data.surface_uv_map = target.data.uv_layers[-1].name
    transfer_uvs_to_curves(curves, target, curves.data.surface_uv_map)

    butil.modify_mesh(curves, 'NODES', apply=False, show_viewport=True,
                      node_group=nodegroup_deform_curves_on_surface())
    
    

    
