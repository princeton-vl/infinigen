# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original Infinigen attribute transfer node groups (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/nodes/nodegroups/transfer_attributes.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.nodes.nodegroups import transfer_attributes as _default
from infinigen.core.util import blender as butil


def uvs_to_attribute(obj, name="uv_map"):
    assert obj.type == "MESH"
    n = len(obj.data.vertices)
    n_loops = len(obj.data.loops)
    # zeros: a vertex with no loop is never written below
    data = np.zeros((n, 3), dtype=np.float32)
    # for a vertex with several loops the last loop's uv wins, as in the default
    loop_vertex = np.empty(n_loops, dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loop_vertex)
    uv = np.empty(n_loops * 2, dtype=np.float32)
    obj.data.uv_layers.active.data.foreach_get("uv", uv)
    last = np.full(n, -1, dtype=np.int64)
    np.maximum.at(last, loop_vertex, np.arange(n_loops))
    has = last >= 0
    data[has, :2] = uv.reshape(n_loops, 2)[last[has]]
    data[has, 2] = 0
    attr = obj.data.attributes.new(name, type="FLOAT_VECTOR", domain="POINT")
    attr.data.foreach_set("vector", data.reshape(-1))
    return attr


def attribute_to_uvs(obj, attr_name):
    assert obj.type == "MESH"
    obj.data.uv_layers.active = obj.data.uv_layers.new()
    n = len(obj.data.vertices)
    data = np.empty(n * 3, dtype=np.float32)
    obj.data.attributes[attr_name].data.foreach_get("vector", data)
    data = data.reshape((n, 3))
    # one bulk write of the same float32 values the default writes per loop
    n_loops = len(obj.data.loops)
    loop_vertex = np.empty(n_loops, dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loop_vertex)
    uv = np.ascontiguousarray(data[loop_vertex, :2])
    obj.data.uv_layers.active.data.foreach_set("uv", uv.reshape(-1))


def transfer_all_with_nearest(
    source,
    target,
    attributes=None,
    uvs=False,
    nearest_point_attributes=None,
):
    """`transfer_all`, copying the POINT attributes in `nearest_point_attributes` from
    the nearest source vertex instead of sampling the nearest surface. With none listed
    it builds the same nodes as `transfer_all`, which is not gated."""
    assert source.type == "MESH"
    assert target.type == "MESH"

    if attributes is None:
        attributes = [
            a.name for a in source.data.attributes if not butil.blender_internal_attr(a)
        ]

    if len(source.data.uv_layers) == 0:
        uvs = False

    if uvs:
        uv_att_name = _default.uvs_to_attribute(source).name
        attributes.append(uv_att_name)

    dtypes = [source.data.attributes[n].data_type for n in attributes]
    domains = [source.data.attributes[n].domain for n in attributes]
    nearest_point_attributes = set(nearest_point_attributes or ())
    unknown = nearest_point_attributes.difference(attributes)
    if unknown:
        raise ValueError(
            f"nearest-point attributes are not being transferred: {sorted(unknown)}"
        )
    invalid_domains = {
        name: source.data.attributes[name].domain
        for name in nearest_point_attributes
        if source.data.attributes[name].domain != "POINT"
    }
    if invalid_domains:
        raise ValueError(
            f"nearest-point transfer requires POINT attributes: {invalid_domains}"
        )

    surface.add_geomod(
        source,
        _transfer_att_node,
        input_kwargs={
            "source": source,
            "target": target,
            "attribute_to_transfer_list": list(zip(attributes, dtypes, domains)),
            "nearest_point_attributes": nearest_point_attributes,
        },
        attributes=attributes,
        apply=True,
        domains=domains,
    )

    surface.add_geomod(
        target,
        _default.copy_geom_info,
        input_kwargs={"source": source, "target": target},
        apply=True,
    )

    if uvs:
        _default.attribute_to_uvs(target, uv_att_name)


def _transfer_att_node(
    nw,
    source,
    target,
    attribute_to_transfer_list=(),
    nearest_point_attributes=frozenset(),
):
    # create a geom node in the non-remeshed version of the mesh (i.e., source)
    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": target})
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
        ],
    )

    for att_name, att_type, _ in attribute_to_transfer_list:
        nw.expose_input(att_name, attribute=att_name)

    position = nw.new_node(Nodes.InputPosition)

    group_output_sockets = {"Geometry": object_info.outputs["Geometry"]}

    nearest_index = None
    for att_name, att_type, domain in attribute_to_transfer_list:
        if att_name in nearest_point_attributes:
            if nearest_index is None:
                nearest_index = nw.new_node(
                    Nodes.SampleNearest,
                    attrs={"domain": "POINT"},
                    input_kwargs={
                        "Geometry": group_input.outputs["Geometry"],
                        "Sample Position": position,
                    },
                )
            transfer_attribute = nw.new_node(
                Nodes.SampleIndex,
                attrs={"data_type": att_type, "domain": domain},
                input_kwargs={
                    "Geometry": group_input.outputs["Geometry"],
                    "Value": group_input.outputs[att_name],
                    "Index": nearest_index.outputs["Index"],
                },
            )
        else:
            transfer_attribute = nw.new_node(
                Nodes.SampleNearestSurface,
                attrs={"data_type": att_type},
                input_kwargs={
                    "Mesh": group_input.outputs["Geometry"],
                    "Value": group_input.outputs[att_name],
                    "Sample Position": position,
                },
            )

        group_output_sockets[att_name] = (transfer_attribute, "Value")

    nw.new_node(Nodes.GroupOutput, input_kwargs=group_output_sockets)
