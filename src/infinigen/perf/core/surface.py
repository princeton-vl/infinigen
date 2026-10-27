# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: primary author of the original Infinigen surface utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/surface.py)
# - Lahav Lipson: surface mixing in the original Infinigen surface utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/surface.py)
# - Lingjie Mei: attributes and geometry nodes in the original Infinigen surface utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/surface.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np

from infinigen.core.nodes import node_info
from infinigen.core.nodes.node_wrangler import (
    NodeWrangler,
    geometry_node_group_empty_new,
    ng_inputs,
    ng_outputs,
)
from infinigen.core.util import blender as butil

# foreach_get / foreach_set copy the buffer directly only when the numpy dtype
# matches the property; any other dtype falls back to a much slower per-element loop.
ATTR_BUFFER_DTYPES = {
    "FLOAT": np.float32,
    "INT": np.int32,
    "FLOAT_VECTOR": np.float32,
    "FLOAT2": np.float32,
    "FLOAT_COLOR": np.float32,
    "BOOLEAN": np.bool_,
    "INT32_2D": np.int32,
}


def _attr_foreach_set(attr, data):
    field = node_info.DATATYPE_FIELDS[attr.data_type]
    data = np.asarray(data).reshape(-1)
    buffer_dtype = ATTR_BUFFER_DTYPES.get(attr.data_type)
    if buffer_dtype is not None and data.dtype != buffer_dtype:
        same_kind = (
            (buffer_dtype is np.bool_ and data.dtype.kind == "b")
            or (buffer_dtype is np.int32 and data.dtype.kind in "iu")
            or (buffer_dtype is np.float32 and data.dtype.kind == "f")
        )
        if same_kind:
            data = data.astype(buffer_dtype)
    attr.data.foreach_set(field, data)


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

        # A modifier applied right away needs no viewport evaluation: the apply
        # operator evaluates it itself, so showing it would only add one more
        # evaluation in the view-layer update before that operator.
        mod.show_viewport = show_viewport and not apply
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


def new_attr_data(obj, attr, type, domain, data: np.array):
    assert isinstance(attr, str)
    assert attr not in obj.data.attributes

    obj.data.attributes.new(name=attr, type=type, domain=domain)
    _attr_foreach_set(obj.data.attributes[attr], data)


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

    buffer_dtype = ATTR_BUFFER_DTYPES.get(attr.data_type, result_dtype)
    data = np.empty(n * dim, dtype=buffer_dtype)
    attr.data.foreach_get(field, data)
    if data.dtype != np.dtype(result_dtype):
        data = data.astype(result_dtype)

    if dim > 1:
        data = data.reshape(-1, dim)

    return data


def write_attr_data(obj, attr, data: np.array, type="FLOAT", domain="POINT"):
    if isinstance(attr, str):
        if attr in obj.data.attributes:
            attr = obj.data.attributes[attr]
        else:
            attr = obj.data.attributes.new(attr, type, domain)

    _attr_foreach_set(attr, data)
