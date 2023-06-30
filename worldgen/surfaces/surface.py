import string
from collections import defaultdict
import importlib

import bpy
from mathutils import Vector
import gin
import numpy as np
from tqdm import trange

from util.blender import set_geomod_inputs # got moved, left here for import compatibility
from nodes import node_info

def remove_materials(obj):
        obj.active_material_index = 0
        for i in range(len(obj.material_slots)):
            bpy.ops.object.material_slot_remove({'object': obj})


    if name is None:
        name = node_func.__name__

    def attr_writer(nw, **kwargs):
        value = node_func(nw)

        nonlocal data_type
        if data_type is None:
            data_type = node_info.NODETYPE_TO_DATATYPE[infer_output_socket(value).type]

        capture = nw.new_node(Nodes.CaptureAttribute,
        output = nw.new_node(Nodes.GroupOutput, input_kwargs={
            'Geometry': (capture, 'Geometry'),
        })


    if isinstance(attr, str):
        attr = obj.data.attributes[attr]

    dtype = attr.data_type
    dim = node_info.DATATYPE_DIMS[dtype]
    field = node_info.DATATYPE_FIELDS[dtype]

    data = np.empty(n * dim)
    attr.data.foreach_get(field, data)
    return data.reshape(-1, dim)


    if isinstance(attr, str):

    field = node_info.DATATYPE_FIELDS[attr.data_type]
    attr.data.foreach_set(field, data.reshape(-1))

    attr.data.foreach_set(field, data.reshape(-1))

def smooth_attribute(obj, name, iters=20, weight=0.05, verbose=False):
    data = read_attr_data(obj, name)

    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int)
    obj.data.edges.foreach_get('vertices', edges)
    edges = edges.reshape(-1, 2)

    r = range(iters) if not verbose else trange(iters)
        vertex_weight = np.ones(len(obj.data.vertices))
        data_out = data.copy()

        data_out[edges[:, 0]] += data[edges[:, 1]] * weight
        vertex_weight[edges[:, 0]] += weight

        data_out[edges[:, 1]] += data[edges[:, 0]] * weight
        vertex_weight[edges[:, 1]] += weight

        data = data_out / vertex_weight[:, None]
    write_attr_data(obj, name, data)


    if name is None:
        name = attr if isinstance(attr, str) else attr.name

    attr_data = read_attr_data(obj, attr)

    if attr_data.shape[-1] != 1:

    group = obj.vertex_groups.new(name=name)

    if binary:
        group.add(np.where(attr_data > min_thresh)[0], 1.0, 'ADD')
    else:
        for i, v in enumerate(attr_data):
            if v > min_thresh:
                group.add([i], v, 'ADD')
    return group

    if argument is None:
        # return selection encompassing everything
        v = nw.new_node(Nodes.Value)
        v.outputs[0].default_value = default_value
        return v
    elif callable(argument):
    elif isinstance(argument, str):
    elif isinstance(argument, (float, int)):
        v = nw.new_node(Nodes.Value)
        v.outputs[0].default_value = argument
        return v
    elif isinstance(argument, (tuple, Vector)):
        v = nw.new_node(Nodes.Vector)
        v.vector = argument
        return v
    else:
        raise ValueError(f'surface.eval_argument couldnt parse {argument}')

def shaderfunc_to_material(shader_func, *args, name=None, **kwargs):
    '''
    Convert a shader_func(nw) directly to a bpy.data.material

    Used in add_material and transpiler's Nodes.SetMaterial handler
    '''

    if name is None:

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nw = NodeWrangler(material.node_tree)
    new_node_tree = shader_func(nw, *args, **kwargs)

    if new_node_tree is not None:
        if isinstance(new_node_tree, tuple) and isnode(new_node_tree[1]):
            new_node_tree, volume = new_node_tree
            nw.new_node(Nodes.MaterialOutput, input_kwargs={'Volume': volume})
        nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': new_node_tree})
    return material

    if input_kwargs is None:
        input_kwargs = {}
        objs = [objs]


        name = "MixedSurface"
        if name in objs[0].data.materials:
            material = objs[0].data.materials[name]
        else:
            objs[0].active_material = material
        nw = NodeWrangler(material.node_tree)
        # grab a reference to whatever is currently linked to output

        # spawn in the node tree to mix with it
        if new_node_tree is None:

        # mix the two together
    else:
        raise ValueError(f"{type(selection)=} not handled.")
    for obj in objs:
        obj.active_material = material
    return material
    
    if input_kwargs is None:
        input_kwargs = {}
    if attributes is None:
        attributes = []
    if name is None:
        name = geo_func.__name__
    if not isinstance(objs, list):
        objs = [objs]
    elif len(objs) == 0:
        return None

    ng = None
    for obj in objs:

        mod = obj.modifiers.new(name=name, type='NODES')
        mod.show_viewport = False

        if mod is None:
            raise ValueError(f'Attempted to surface.add_geomod({obj=}), yet created modifier was None. '

        mod.show_viewport = show_viewport
            if reuse and name in bpy.data.node_groups:
                mod.node_group = bpy.data.node_groups[name]
            else:
                nw = NodeWrangler(mod)
            ng = mod.node_group
            ng.name = name
        else:
            mod.node_group = ng
        for id, att_name in zip(identifiers, attributes):
            # attributes are a 1-indexed list, and Geometry is the first element, so we start from 2

    if apply:
        for obj in objs:
            butil.apply_modifiers(obj, name)
        return None
    return mod

class Registry:

    def __init__(self):
        self._registry = None

    def get_surface(name):
        if name == '':
        
        prefixes = ['surfaces.templates', 'surfaces.scatters']
        for prefix in prefixes:
                return importlib.import_module(prefix + '.' + name)
            except ModuleNotFoundError:
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
                self._registry[k] = [(self.get_surface(name), weight) for name, weight in v]

    def __call__(self, category_key):
        if self._registry is None:
            raise ValueError(
                f'Surface registry has not been initialized! Have you loaded gin and called .initialize()?'
                'Note, this step cannot happen at module initialization time, as gin is not yet loaded'
            )

        if not category_key in self._registry:

        return self.sample_registry(self._registry[category_key])        


