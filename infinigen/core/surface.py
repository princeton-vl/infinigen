# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alex Raistrick: primary author
# - Lahav Lipson: Surface mixing
# - Lingjie Mei: attributes and geo nodes

import string
from inspect import signature

import bpy
import numpy as np
from mathutils import Vector
from tqdm import trange

from infinigen.core import tags as t
from infinigen.core.nodes import node_info
from infinigen.core.nodes.node_wrangler import (
    Nodes,
    NodeWrangler,
    geometry_node_group_empty_new,
    ng_inputs,
    ng_outputs,
)
from infinigen.core.nodes.utils import infer_output_socket, isnode
from infinigen.core.util import blender as butil


def remove_materials(obj):
    with butil.SelectObjects(obj):
        obj.active_material_index = 0
        for i in range(len(obj.material_slots)):
            bpy.ops.object.material_slot_remove({"object": obj})


def write_attribute(objs, node_func, name=None, data_type=None, apply=False):
    if name is None:
        name = node_func.__name__

    def attr_writer(nw, **kwargs):
        value = node_func(nw)

        nonlocal data_type
        if data_type is None:
            data_type = node_info.NODETYPE_TO_DATATYPE[infer_output_socket(value).type]

        capture = nw.new_node(
            Nodes.CaptureAttribute,
            attrs={"data_type": data_type},
            input_kwargs={"Geometry": nw.new_node(Nodes.GroupInput), "Value": value},
        )
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={
                "Geometry": (capture, "Geometry"),
                name: (capture, "Attribute"),
            },
        )

    add_geomod(
        objs,
        attr_writer,
        name=f"write_attribute({name})",
        apply=apply,
        attributes=[name],
    )
    return name


def read_attr_data(obj, attr, domain="POINT", result_dtype=None) -> np.array:
    if isinstance(attr, str):
        attr = obj.data.attributes[attr]
        domain = attr.domain

    if domain == "POINT":
        n = len(obj.data.vertices)
    elif domain == "EDGE":
        n = len(obj.data.edges)
    elif domain == "FACE":
        n = len(obj.data.polygons)
    else:
        raise ValueError(f"Unknown domain {domain}")

    dim = node_info.DATATYPE_DIMS[attr.data_type]
    field = node_info.DATATYPE_FIELDS[attr.data_type]

    if result_dtype is None:
        result_dtype = node_info.DATATYPE_TO_PYTYPE[attr.data_type]

    data = np.empty(n * dim, dtype=result_dtype)
    attr.data.foreach_get(field, data)

    if dim > 1:
        data = data.reshape(-1, dim)

    return data


def set_active(obj, name):
    attributes = obj.data.attributes
    attributes.active_index = next(
        (i for i, a in enumerate(attributes) if a.name == name)
    )
    attributes.active = attributes[attributes.active_index]


def write_attr_data(obj, attr, data: np.array, type="FLOAT", domain="POINT"):
    if isinstance(attr, str):
        if attr in obj.data.attributes:
            attr = obj.data.attributes[attr]
        else:
            attr = obj.data.attributes.new(attr, type, domain)

    field = node_info.DATATYPE_FIELDS[attr.data_type]
    attr.data.foreach_set(field, data.reshape(-1))


def new_attr_data(obj, attr, type, domain, data: np.array):
    assert isinstance(attr, str)
    assert attr not in obj.data.attributes

    obj.data.attributes.new(name=attr, type=type, domain=domain)
    attr = obj.data.attributes[attr]
    field = node_info.DATATYPE_FIELDS[attr.data_type]
    attr.data.foreach_set(field, data.reshape(-1))


def smooth_attribute(obj, name, iters=20, weight=0.05, verbose=False):
    data = read_attr_data(obj, name)

    edges = np.empty(len(obj.data.edges) * 2, dtype=int)
    obj.data.edges.foreach_get("vertices", edges)
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
            f"Could not convert non-scalar attribute {attr} to vertex group, expected 1 data dimension but "
            f"got {attr_data.shape=}"
        )

    group = obj.vertex_groups.new(name=name)

    if binary:
        group.add(np.where(attr_data > min_thresh)[0], 1.0, "ADD")
    else:
        for i, v in enumerate(attr_data):
            if v > min_thresh:
                group.add([i], v, "ADD")

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
        raise ValueError(f"surface.eval_argument couldnt parse {argument}")


def shaderfunc_to_material(shader_func, *args, name=None, **kwargs):
    """
    Convert a shader_func(nw) directly to a bpy.data.material

    Used in add_material and transpiler's Nodes.SetMaterial handler
    """

    if name is None:
        name = shader_func.__name__

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.remove(
        material.node_tree.nodes["Principled BSDF"]
    )  # remove the default BSDF

    nw = NodeWrangler(material.node_tree)

    new_node_tree = shader_func(nw, *args, **kwargs)

    if new_node_tree is not None:
        if isinstance(new_node_tree, tuple) and isnode(new_node_tree[1]):
            new_node_tree, volume = new_node_tree
            nw.new_node(Nodes.MaterialOutput, input_kwargs={"Volume": volume})
        nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": new_node_tree})

    return material


def seed_generator(size=8, chars=string.ascii_uppercase):
    return "".join(np.random.choice(list(chars)) for _ in range(size))


def add_material(
    objs,
    shader_func,
    selection=None,
    input_args=None,
    input_kwargs=None,
    name=None,
    reuse=False,
):
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
    elif isinstance(selection, (str, t.Semantics)):
        if isinstance(selection, t.Semantics):
            selection = selection.value
        name = "MixedSurface"
        if name in objs[0].data.materials:
            material = objs[0].data.materials[name]
        else:
            material = bpy.data.materials.new(name=name)
            material.use_nodes = True
            material.node_tree.nodes["Principled BSDF"].inputs[
                "Base Color"
            ].default_value = (1, 0, 1, 1)  # Set Magenta
            objs[0].active_material = material

        nw = NodeWrangler(material.node_tree)

        new_attribute_node = nw.new_node(
            Nodes.Attribute, [], {"attribute_name": selection}
        )
        if "Attribute Sum" in material.node_tree.nodes:
            old_attribute_sum_node = material.node_tree.nodes["Attribute Sum"]
            if old_attribute_sum_node.type == "ATTRIBUTE":
                socket_index_old = 2
            else:
                socket_index_old = 0
            new_attribute_sum_node = nw.scalar_add(
                (old_attribute_sum_node, socket_index_old), (new_attribute_node, 2)
            )
            old_attribute_sum_node.name = "Attribute Sum Old"
            new_attribute_sum_node.name = "Attribute Sum"
        else:
            new_attribute_node.name = "Attribute Sum"
            new_attribute_sum_node = new_attribute_node
        # grab a reference to whatever is currently linked to output
        links_to_output = [
            link
            for link in nw.links
            if (link.to_node.bl_idname == Nodes.MaterialOutput)
        ]
        assert len(links_to_output) == 1, links_to_output
        penultimate_node = links_to_output.pop().from_node
        if new_attribute_sum_node.type == "ATTRIBUTE":
            socket_index_new = 2
        else:
            socket_index_new = 0
        selection_weight = nw.divide2(
            (new_attribute_node, 2), (new_attribute_sum_node, socket_index_new)
        )

        # spawn in the node tree to mix with it
        new_node_tree = shader_func(nw, **input_kwargs)
        if new_node_tree is None:
            raise ValueError(
                f"{shader_func} returned None while attempting add_material(selection=...). Shaderfunc must "
                f"return its output to be mixable"
            )
        if isinstance(new_node_tree, tuple) and isnode(new_node_tree[1]):
            new_node_tree, volume = new_node_tree
            nw.new_node(Nodes.MaterialOutput, input_kwargs={"Volume": volume})

        # mix the two together
        mix_shader = nw.new_node(
            Nodes.MixShader, [selection_weight, penultimate_node, new_node_tree]
        )
        nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": mix_shader})
    else:
        raise ValueError(f"{type(selection)=} not handled.")

    for obj in objs:
        obj.active_material = material
    return material


def assign_material(
    objs: bpy.types.Object | list[bpy.types.Object],
    material: bpy.types.Material,
    selection: str | np.ndarray | None = None,
):
    if not isinstance(objs, list):
        objs = [objs]

    for obj in objs:
        if selection is None:
            obj.active_material = material
            return

        if len(obj.material_slots) == 0:
            raise ValueError(
                f"{assign_material.__name__} recieved {selection=} but existing materials to combine with"
            )

        if isinstance(selection, str):
            if selection.startswith("!"):
                selection_arr = ~read_attr_data(
                    obj, selection[1:], domain="FACE", result_dtype=bool
                )
            else:
                selection_arr = read_attr_data(
                    obj, selection, domain="FACE", result_dtype=bool
                )
        elif isinstance(selection, np.ndarray):
            selection_arr = selection
        else:
            raise ValueError(
                f"Expected str or np.ndarray for selection, got {selection=}"
            )

        if selection_arr.dtype != bool:
            raise ValueError(
                f"Got unexpected {selection_arr.dtype=} for {selection=}, expected `bool`"
            )

        if "face_mask" not in obj.data.attributes:
            active_mat = obj.active_material
            active_mat_idx = next(
                (s.slot_index for s in obj.material_slots if s.material == active_mat),
                None,
            )
            if active_mat is None:
                raise ValueError(
                    f"Could not find active material {active_mat=}. Should be impossible since {len(obj.material_slots)=} is non empty"
                )

            material_index = np.full(len(obj.data.polygons), active_mat_idx, dtype=int)
        else:
            material_index = read_attr_data(obj, "material_index", domain="FACE")

        with butil.SelectObjects(obj):
            bpy.ops.object.material_slot_add()
        slot = obj.material_slots[-1]
        assert slot.material is None

        slot.material = material
        material_index[selection_arr] = slot.slot_index
        write_attr_data(
            obj, "material_index", material_index, type="INT", domain="FACE"
        )


def add_geomod(
    objs,
    geo_func,
    name=None,
    apply=False,
    reuse=False,
    input_args=None,
    input_kwargs=None,
    attributes=None,
    show_viewport=True,
    selection=None,
    domains=None,
    input_attributes=None,
):
    if input_args is None:
        input_args = []
    if input_kwargs is None:
        input_kwargs = {}
    if attributes is None:
        attributes = []
    if domains is None:
        domains = ["POINT"] * len(attributes)
    if input_attributes is None:
        input_attributes = [None] * 128

    if name is None:
        name = geo_func.__name__
    if not isinstance(objs, list):
        objs = [objs]
    elif len(objs) == 0:
        return None

    if selection is not None:
        input_kwargs["selection"] = selection

    ng = None
    for obj in objs:
        mod = obj.modifiers.new(name=name, type="NODES")
        mod.show_viewport = False

        if mod is None:
            raise ValueError(
                f"Attempted to surface.add_geomod({obj=}), yet created modifier was None. "
                f"Check that {obj.type=} supports geo modifiers"
            )

        mod.show_viewport = show_viewport
        if ng is None:  # Create a unique node_group for the first one only
            if reuse and name in bpy.data.node_groups:
                mod.node_group = bpy.data.node_groups[name]
            else:
                # print("input_kwargs", input_kwargs, geo_func.__name__)
                if mod.node_group is None:
                    group = geometry_node_group_empty_new()
                    mod.node_group = group
                nw = NodeWrangler(mod)
                geo_func(nw, *input_args, **input_kwargs)
            ng = mod.node_group
            ng.name = name
        else:
            mod.node_group = ng

        non_geometries = [
            o
            for o in ng_outputs(mod.node_group).values()
            if o.socket_type != "NodeSocketGeometry"
        ]
        if len(non_geometries) != len(attributes):
            raise Exception(
                f"has {len(non_geometries)} identifiers, but {len(attributes)} attributes. Specifically, "
                f"{non_geometries=} and {attributes=}"
            )
        for o, att_name in zip(non_geometries, attributes):
            # attributes are a 1-indexed list, and Geometry is the first element, so we start from 2
            # while f'Output_{i}_attribute_name' not in
            mod[o.identifier + "_attribute_name"] = att_name
        for o, domain in zip(non_geometries, domains):
            o.attribute_domain = domain

        inputs = ng_inputs(mod.node_group)
        if not any(att_name is None for att_name in input_attributes):
            raise Exception("None should be provided for Geometry inputs.")
        for i, att_name in zip(inputs.values(), input_attributes):
            o = i.identifier
            if att_name is not None:
                mod[f"{o}_use_attribute"] = True
                mod[f"{o}_attribute_name"] = att_name

    if apply:
        for obj in objs:
            butil.apply_modifiers(obj, name)
        return None

    return mod


class NoApply:
    @staticmethod
    def apply(objs, *args, **kwargs):
        return objs, []
