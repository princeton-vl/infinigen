# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick, David Yan

import logging
from collections import OrderedDict

from .node_info import (
    NODECLASS_TO_DATATYPE,
    Nodes,
)
from .utils import infer_output_socket

logger = logging.getLogger(__name__)


def map_dict_keys(d, m):
    for m_from, m_to in m.items():
        if m_from not in d:
            continue
        if m_to in d:
            raise ValueError(
                f"{m_from} would map to {m_to} but {d} already contains that key"
            )
        d[m_to] = d.pop(m_from)
    return d


def make_virtual_mixrgb(nw, orig_type, input_args, attrs, input_kwargs):
    attrs["data_type"] = "RGBA"

    key_mapping = OrderedDict({"Fac": "Factor", "Color1": "A", "Color2": "B"})
    map_dict_keys(input_kwargs, key_mapping)

    # any previous uses of input_args are no longer valid, since the node has lots of hidden type-based sockets now
    # we will convert any input_args present into input_kwargs instead
    for k, a in zip(key_mapping.values(), input_args):
        if k in input_kwargs:
            raise ValueError(
                f"In {make_virtual_mixrgb}, encountered {orig_type} with conflicting {len(input_args)=} and {input_kwargs.keys()}"
            )
        input_kwargs[k] = a
    input_args = []

    return nw.new_node(
        node_type=Nodes.Mix,
        input_args=input_args,
        attrs=attrs,
        input_kwargs=input_kwargs,
        compat_mode=False,
    )


def make_virtual_transfer_attribute(nw, orig_type, input_args, attrs, input_kwargs):
    if attrs is None:
        raise ValueError(
            f"{attrs=} in make_virtual_transfer_attribute, cannot infer correct node type mapping"
        )

    if attrs["mapping"] == "NEAREST_FACE_INTERPOLATED":
        mapped_type = Nodes.SampleNearestSurface
        map_dict_keys(
            input_kwargs,
            {
                "Source": "Mesh",
                "Attribute": "Value",
                "Source Position": "Sample Position",
            },
        )
    elif attrs["mapping"] == "NEAREST":
        raise ValueError(
            "Compatibility mapping for mode='NEAREST' is not supported, please modify the code to resolve this outdated instance of TransferAttribute"
        )
    elif attrs["mapping"] == "INDEX":
        mapped_type = Nodes.SampleIndex
        map_dict_keys(input_kwargs, {"Source": "Geometry", "Attribute": "Value"})
    else:
        assert False

    logger.warning(
        f"Converting request for Nodes.TransferAttribute to {mapped_type}"
        f"to ensure compatibility with bl3.3 code, but this is unsafe. Please update to avoid {Nodes.TransferAttribute}"
    )

    return nw.new_node(
        node_type=mapped_type,
        input_args=input_args,
        attrs=attrs,
        input_kwargs=input_kwargs,
        compat_mode=False,
    )


def compat_args_sample_curve(nw, orig_type, input_args, attrs, input_kwargs):
    map_dict_keys(input_kwargs, {"Curve": "Curves"})
    return nw.new_node(
        node_type=orig_type,
        input_args=input_args,
        attrs=attrs,
        input_kwargs=input_kwargs,
        compat_mode=False,
    )


def compat_musgrave_texture(nw, orig_type, input_args, attrs, input_kwargs):
    # https://docs.blender.org/manual/en/4.2/render/shader_nodes/textures/musgrave.html
    old_names = [
        "Vector",
        "W",
        "Scale",
        "Detail",
        "Dimension",
        "Lacunarity",
        "Offset",
        "Gain",
    ]
    default_values = {"Dimension": 2, "Lacunarity": 2, "Detail": 2}
    for name, value in zip(old_names, input_args):
        input_kwargs[name] = value
    for name, value in default_values.items():
        if name not in input_kwargs:
            input_kwargs[name] = value
    # handle roughness
    if nw.is_socket(input_kwargs["Dimension"]) or nw.is_socket(
        input_kwargs["Lacunarity"]
    ):
        input_kwargs["Roughness"] = nw.math(
            "POWER",
            input_kwargs["Lacunarity"],
            nw.scalar_sub(0, input_kwargs["Dimension"]),
        )
    else:
        input_kwargs["Roughness"] = input_kwargs["Lacunarity"] ** (
            -input_kwargs["Dimension"]
        )
    input_kwargs.pop("Dimension")
    # handle detail
    if nw.is_socket(input_kwargs["Detail"]):
        input_kwargs["Detail"] = nw.scalar_sub(input_kwargs["Detail"], 1)
    else:
        input_kwargs["Detail"] = input_kwargs["Detail"] - 1
    if "musgrave_dimensions" in attrs:
        attrs["noise_dimensions"] = attrs.pop("musgrave_dimensions")
    if "musgrave_type" in attrs:
        attrs["noise_type"] = attrs.pop("musgrave_type")
    attrs["normalize"] = False
    return nw.new_node(
        node_type=Nodes.NoiseTexture,
        input_args=[],
        attrs=attrs,
        input_kwargs=input_kwargs,
        compat_mode=False,
    )


def compat_capture_attribute(nw, orig_type, input_args, attrs, input_kwargs):
    if "Geometry" in input_kwargs:
        geometry = input_kwargs.pop("Geometry")
    elif len(input_args) >= 1:
        geometry = input_args.pop(0)
    else:
        raise ValueError(
            f"Geometry is not given for {orig_type=} and {input_args=} and {input_kwargs=}"
        )

    if "data_type" in attrs:
        data_type = attrs.pop("data_type")
    else:
        data_type = None
    data_types = {}

    inputs = {}

    def get_name(k):
        if isinstance(k, int):
            if "Attribute" in inputs:
                return f"Attribute_{k}"
            return "Attribute"
        return k

    for k, v in input_kwargs.items():
        k = get_name(k)
        inputs[k] = v
        data_types[k] = (
            NODECLASS_TO_DATATYPE[infer_output_socket(v).bl_idname]
            if data_type is None
            else data_type
        )
    for k, v in enumerate(input_args):
        k += 1
        inputs[k] = v
        data_types[k] = (
            NODECLASS_TO_DATATYPE[infer_output_socket(v).bl_idname]
            if data_type is None
            else data_type
        )
    node = nw.new_node(
        node_type=orig_type,
        input_args=[geometry],
        attrs=attrs,
        input_kwargs=inputs,
        compat_mode=False,
    )
    for i, d in enumerate(data_types.values()):
        node.capture_items[i].data_type = d
    return node


def compat_principled_bsdf(nw, orig_type, input_args, attrs, input_kwargs):
    input_kwargs["Subsurface Scale"] = 1
    if "Subsurface Color" in input_kwargs:
        logger.warning(f"Subsurface Color no longer in use for {orig_type}")
        input_kwargs.pop("Subsurface Color")
    return nw.new_node(
        node_type=orig_type,
        input_args=input_args,
        attrs=attrs,
        input_kwargs=input_kwargs,
        compat_mode=False,
    )


COMPATIBILITY_MAPPINGS = {
    Nodes.MixRGB: make_virtual_mixrgb,
    Nodes.TransferAttribute: make_virtual_transfer_attribute,
    Nodes.SampleCurve: compat_args_sample_curve,
    Nodes.MusgraveTexture: compat_musgrave_texture,
    Nodes.CaptureAttribute: compat_capture_attribute,
    Nodes.PrincipledBSDF: compat_principled_bsdf,
}
