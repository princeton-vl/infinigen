# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alex Raistrick: primary author
# - Lahav Lipson: Surface mixing


import string
from collections import defaultdict
import importlib
from inspect import signature

import bpy
from mathutils import Vector
import gin
import numpy as np
from tqdm import trange

from infinigen.core.util import blender as butil
from infinigen.core.util.blender import set_geomod_inputs # got moved, left here for import compatibility
from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes, isnode, infer_output_socket, geometry_node_group_empty_new
from infinigen.core.nodes import node_info

def remove_materials(obj):
    with butil.SelectObjects(obj):
        obj.active_material_index = 0
        for i in range(len(obj.material_slots)):
            bpy.ops.object.material_slot_remove({'object': obj})


def write_attribute(objs, node_func, name=None, data_type=None, apply=False):
    if name is None:
        name = node_func.__name__

    def attr_writer(nw, **kwargs):
        value = node_func(nw)

        nonlocal data_type
        if data_type is None:
            data_type = node_info.NODETYPE_TO_DATATYPE[infer_output_socket(value).type]

        capture = nw.new_node(Nodes.CaptureAttribute,
                              attrs={'data_type': data_type},
                              input_kwargs={
                                  'Geometry': nw.new_node(Nodes.GroupInput),
                                  'Value': value
                              })
        output = nw.new_node(Nodes.GroupOutput, input_kwargs={
            'Geometry': (capture, 'Geometry'),
            name: (capture, 'Attribute')
        })

    mod = add_geomod(objs, attr_writer, name=f'write_attribute({name})', apply=apply, attributes=[name])
    return name 

def read_attr_data(obj, attr, domain='POINT') -> np.array:
    if isinstance(attr, str):
        attr = obj.data.attributes[attr]

    if domain == 'POINT':
        n = len(obj.data.vertices)
    elif domain == 'EDGE':
        n = len(obj.data.edges)
    elif domain == 'FACE':
        n = len(obj.data.polygons)
    else:
        raise NotImplementedError
    dtype = attr.data_type
    dim = node_info.DATATYPE_DIMS[dtype]
    field = node_info.DATATYPE_FIELDS[dtype]

    data = np.empty(n * dim)
    attr.data.foreach_get(field, data)
    return data.reshape(-1, dim)


def write_attr_data(obj, attr, data: np.array, type='FLOAT', domain='POINT'):
    if isinstance(attr, str):
        if attr in obj.data.attributes:
            attr = obj.data.attributes[attr]
        else:
            attr = obj.data.attributes.new(attr, type, domain)

    field = node_info.DATATYPE_FIELDS[attr.data_type]
    attr.data.foreach_set(field, data.reshape(-1))

def new_attr_data(obj, attr, type, domain, data: np.array):
    assert(isinstance(attr, str))
    assert(attr not in obj.data.attributes)

    obj.data.attributes.new(name=attr, type=type, domain=domain)
    attr = obj.data.attributes[attr]
    field = node_info.DATATYPE_FIELDS[attr.data_type]
    attr.data.foreach_set(field, data.reshape(-1))


def smooth_attribute(obj, name, iters=20, weight=0.05, verbose=False):
    data = read_attr_data(obj, name)

    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int)
    obj.data.edges.foreach_get('vertices', edges)
    edges = edges.reshape(-1, 2)

    r = range(iters) if not verbose else trange(iters)
    for _ in r:
        vertex_weight = np.ones(len(obj.data.vertices))
        data_out = data.copy()

        data_out[edges[:, 0]] += data[edges[:, 1]] * weight
        vertex_weight[edges[:, 0]] += weight

        data_out[edges[:, 1]] += data[edges[:, 0]] * weight
        vertex_weight[edges[:, 1]] += weight

        data = data_out / vertex_weight[:, None]

    write_attr_data(obj, name, data)


def attribute_to_vertex_group(obj, attr, name=None, min_thresh=0, binary=False):
    if name is None:
        name = attr if isinstance(attr, str) else attr.name

    attr_data = read_attr_data(obj, attr)

    if attr_data.shape[-1] != 1:
        raise ValueError(
            f'Could not convert non-scalar attribute {attr} to vertex group, expected 1 data dimension but got {attr_data.shape=}')

    group = obj.vertex_groups.new(name=name)

    if binary:
        group.add(np.where(attr_data > min_thresh)[0], 1.0, 'ADD')
    else:
        for i, v in enumerate(attr_data):
            if v > min_thresh:
                group.add([i], v, 'ADD')

    return group


def eval_argument(nw, argument, default_value=1.0, **kwargs):
    if argument is None:
        # return selection encompassing everything
        v = nw.new_node(Nodes.Value)
        v.outputs[0].default_value = default_value
        return v
    elif callable(argument):
        allowed_keys = list(signature(argument).parameters.keys())[1:]
        return argument(nw, **{k: v for k, v in kwargs.items() if k in allowed_keys})
    elif isinstance(argument, str):
        return nw.expose_input(name=argument, attribute=argument, val=default_value)
    elif isinstance(argument, (float, int)):
        v = nw.new_node(Nodes.Value)
        v.outputs[0].default_value = argument
        return v
    elif isinstance(argument, (tuple, Vector)):
        v = nw.new_node(Nodes.Vector)
        v.vector = argument
        return v
    elif nw.is_socket(argument):
        return argument
    else:
        raise ValueError(f'surface.eval_argument couldnt parse {argument}')


def shaderfunc_to_material(shader_func, *args, name=None, **kwargs):
    '''
    Convert a shader_func(nw) directly to a bpy.data.material

    Used in add_material and transpiler's Nodes.SetMaterial handler
    '''

    if name is None:
        name = shader_func.__name__

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.remove(material.node_tree.nodes['Principled BSDF'])  # remove the default BSDF

    nw = NodeWrangler(material.node_tree)
    new_node_tree = shader_func(nw, *args, **kwargs)

    if new_node_tree is not None:
        if isinstance(new_node_tree, tuple) and isnode(new_node_tree[1]):
            new_node_tree, volume = new_node_tree
            nw.new_node(Nodes.MaterialOutput, input_kwargs={'Volume': volume})
        nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': new_node_tree})

    return material


def seed_generator(size=8, chars=string.ascii_uppercase):
    return ''.join(np.random.choice(list(chars)) for _ in range(size))


def add_material(objs, shader_func, selection=None, input_args=None, input_kwargs=None, name=None, reuse=False):
    if input_args is None:
        input_args = []
    if input_kwargs is None:
        input_kwargs = {}
    if not isinstance(objs, list):
        objs = [objs]

    if selection is None:
        if name is None:
            name = shader_func.__name__
        if (not reuse) and (name in bpy.data.materials):
            name += f"_{seed_generator(8)}"
        material = shaderfunc_to_material(shader_func, *input_args, **input_kwargs)
    elif isinstance(selection, str):

        name = "MixedSurface"
        if name in objs[0].data.materials:
            material = objs[0].data.materials[name]
        else:
            material = bpy.data.materials.new(name=name)
            material.use_nodes = True
            material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (1, 0, 1, 1)  # Set Magenta
            objs[0].active_material = material

        nw = NodeWrangler(material.node_tree)

        new_attribute_node = nw.new_node(Nodes.Attribute, [], {"attribute_name": selection})
        if "Attribute Sum" in material.node_tree.nodes:
            old_attribute_sum_node = material.node_tree.nodes["Attribute Sum"]
            if old_attribute_sum_node.type == "ATTRIBUTE":
                socket_index_old = 2
            else:
                socket_index_old = 0
            new_attribute_sum_node = nw.scalar_add((old_attribute_sum_node, socket_index_old), (new_attribute_node, 2))
            old_attribute_sum_node.name = "Attribute Sum Old"
            new_attribute_sum_node.name = "Attribute Sum"
        else:
            new_attribute_node.name = "Attribute Sum"
            new_attribute_sum_node = new_attribute_node
        # grab a reference to whatever is currently linked to output
        links_to_output = [link for link in nw.links if (link.to_node.bl_idname == Nodes.MaterialOutput)]
        assert len(links_to_output) == 1, links_to_output
        penultimate_node = links_to_output.pop().from_node
        if new_attribute_sum_node.type == "ATTRIBUTE":
            socket_index_new = 2
        else:
            socket_index_new = 0
        selection_weight = nw.divide2(
            (new_attribute_node, 2),
            (new_attribute_sum_node, socket_index_new)
        )

        # spawn in the node tree to mix with it
        new_node_tree = shader_func(nw, **input_kwargs)
        if new_node_tree is None:
            raise ValueError(
                f'{shader_func} returned None while attempting add_material(selection=...). Shaderfunc must return its output to be mixable')
        if isinstance(new_node_tree, tuple) and isnode(new_node_tree[1]):
            new_node_tree, volume = new_node_tree
            nw.new_node(Nodes.MaterialOutput, input_kwargs={'Volume': volume})

        # mix the two together
        mix_shader = nw.new_node(Nodes.MixShader, [selection_weight, penultimate_node, new_node_tree])
        nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': mix_shader})
    else:
        raise ValueError(f"{type(selection)=} not handled.")

    for obj in objs:
        obj.active_material = material
    return material

def add_geomod(objs, geo_func,
               name=None, apply=False, reuse=False, input_args=None,
               input_kwargs=None, attributes=None, show_viewport=True, selection=None,
               domains=None, input_attributes=None, ):
    
    if input_args is None:
        input_args = []
    if input_kwargs is None:
        input_kwargs = {}
    if attributes is None:
        attributes = []
    if domains is None:
        domains = ['POINT'] * len(attributes)
    if input_attributes is None:
        input_attributes = [None] * 128

    if name is None:
        name = geo_func.__name__
    if not isinstance(objs, list):
        objs = [objs]
    elif len(objs) == 0:
        return None

    if selection is not None:
        input_kwargs['selection'] = selection

    ng = None
    for obj in objs:

        mod = obj.modifiers.new(name=name, type='NODES')
        mod.show_viewport = False

        if mod is None:
            raise ValueError(f'Attempted to surface.add_geomod({obj=}), yet created modifier was None. '
                             f'Check that {obj.type=} supports geo modifiers')

        mod.show_viewport = show_viewport
        if ng is None:  # Create a unique node_group for the first one only
            if reuse and name in bpy.data.node_groups:
                mod.node_group = bpy.data.node_groups[name]
            else:
                # print("input_kwargs", input_kwargs, geo_func.__name__)
                if mod.node_group == None:
                    group = geometry_node_group_empty_new()
                    mod.node_group = group
                nw = NodeWrangler(mod)
                geo_func(nw, *input_args, **input_kwargs)
            ng = mod.node_group
            ng.name = name
        else:
            mod.node_group = ng

        outputs = mod.node_group.outputs
        identifiers = [outputs[i].identifier for i in range(len(outputs)) if outputs[i].type != 'GEOMETRY']
        if len(identifiers) != len(attributes):
            raise Exception(
                f"has {len(identifiers)} identifiers, but {len(attributes)} attributes. Specifically, {identifiers=} and {attributes=}")
        for id, att_name in zip(identifiers, attributes):
            # attributes are a 1-indexed list, and Geometry is the first element, so we start from 2
            # while f'Output_{i}_attribute_name' not in
            mod[id + '_attribute_name'] = att_name
        os = [outputs[i] for i in range(len(outputs)) if outputs[i].type != 'GEOMETRY']
        for o, domain in zip(os, domains):
            o.attribute_domain = domain

        inputs = mod.node_group.inputs
        if not any(att_name is None for att_name in input_attributes):
            raise Exception('None should be provided for Geometry inputs.')
        for i, att_name in zip(inputs, input_attributes):
            id = i.identifier
            if att_name is not None:
                mod[f'{id}_use_attribute'] = True
                mod[f'{id}_attribute_name'] = att_name

    if apply:
        for obj in objs:
            butil.apply_modifiers(obj, name)
        return None

    return mod


class NoApply:
    @staticmethod
    def apply(objs, *args, **kwargs):
        return objs, []


class Registry:

    def __init__(self):
        self._registry = None

    @staticmethod
    def get_surface(name):
        if name == '':
            return NoApply
        
        prefixes = [
            'infinigen.infinigen_gpl.surfaces', 
            'infinigen.assets.materials', 
            'infinigen.assets.scatters'
        ]
        for prefix in prefixes:
            try:
                return importlib.import_module('.' + name, prefix)
            except ModuleNotFoundError as e:
                continue

        raise ValueError(f'Could not find {name=} in any of {prefixes}')

    @staticmethod
    def sample_registry(registry):
        mods, probs = zip(*registry)
        return np.random.choice(mods, p=np.array(probs) / sum(probs))

    @gin.configurable('registry')
    def initialize_from_gin(self, smooth_categories=0, **gin_category_info):

        if smooth_categories != 0:
            raise NotImplementedError

        with gin.unlock_config():
            self._registry = defaultdict(list)
            for k, v in gin_category_info.items():
                self._registry[k] = [(self.get_surface(name), weight) for name, weight in v]

    def __call__(self, category_key):
        if self._registry is None:
            raise ValueError(
                'Surface registry has not been initialized! Have you loaded gin and called .initialize()?'
                'Note, this step cannot happen at module initialization time, as gin is not yet loaded'
            )

        if category_key not in self._registry:
            raise KeyError(
                f'registry recieved request with {category_key=}, but no gin_config for this key was provided. {self._registry.keys()=}')

        return self.sample_registry(self._registry[category_key])        


registry = Registry()
