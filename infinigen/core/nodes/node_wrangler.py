# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.
# Authors:
# - Alexander Raistrick: NodeWrangler class, node linking, expose_input, nodegroup support
# - Zeyu Ma: initial version, fixes, arithmetic utilties
# - Lingjie Mei: NodeWrangler compare, switch, build and other utilities
# - Karhan Kayan: geometry_node_group_empty_new()


import logging
import re
import sys
import traceback
import warnings
from collections.abc import Iterable

import bpy
import numpy as np

from infinigen.core.nodes import node_info
from infinigen.core.nodes.node_info import Nodes, map_socket
from infinigen.core.util.random import random_vector3

from .compatibility import COMPATIBILITY_MAPPINGS
from .utils import infer_input_socket, infer_output_socket

logger = logging.getLogger(__name__)


class NodeMisuseWarning(UserWarning):
    pass


def ng_inputs(node_group):
    return {s.name: s for s in node_group.interface.items_tree if s.in_out == "INPUT"}


def ng_outputs(node_group):
    return {s.name: s for s in node_group.interface.items_tree if s.in_out == "OUTPUT"}


# This is for Blender 3.3 because of the nodetree change
def geometry_node_group_empty_new():
    group = bpy.data.node_groups.new("Geometry Nodes", "GeometryNodeTree")
    group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")
    output_node.is_active_output = True

    input_node.select = False
    output_node.select = False

    input_node.location.x = -200 - input_node.width
    output_node.location.x = 200

    group.links.new(output_node.inputs[0], input_node.outputs[0])

    return group


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    traceback_str = " ".join(traceback.format_stack())
    traceback_files = re.findall('/([^/]*\.py)", line ([0-9]+)', traceback_str)
    traceback_files = [
        f"{f}:{l}"
        for f, l in traceback_files
        if all(s not in f for s in {"warnings.py", "node_wrangler.py"})
    ]
    if len(traceback_files):
        message.args = (
            f"{message.args[0]}. The issue is probably coming from {traceback_files.pop()}",
        )
    log = file if hasattr(file, "write") else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


class NodeWrangler:
    def __init__(self, node_group):
        if issubclass(type(node_group), bpy.types.NodeTree):
            self.modifier = None
            self.node_group = node_group
        elif issubclass(type(node_group), bpy.types.NodesModifier):
            self.modifier = node_group
            self.node_group = self.modifier.node_group
        else:
            raise ValueError(
                f"Couldnt initialize NodeWrangler with {node_group=}, {type(node_group)=}"
            )

        self.nodes = self.node_group.nodes
        self.links = self.node_group.links

        self.nodegroup_input_data = {}

        self.input_attribute_data = {}
        self.position_translation_seed = {}

        self.input_consistency_forced = 0

    def force_input_consistency(self):
        self.input_consistency_forced = 1

    def new_value(self, v, label=None):
        node = self.new_node(Nodes.Value, label=label)
        node.outputs[0].default_value = v
        return node

    def new_node(
        self,
        node_type,
        input_args=None,
        attrs=None,
        input_kwargs=None,
        label=None,
        expose_input=None,
        compat_mode=True,
    ):
        if input_args is None:
            input_args = []
        if input_kwargs is None:
            input_kwargs = {}

        if attrs is None:
            attrs = {}

        compat_map = COMPATIBILITY_MAPPINGS.get(node_type)
        if compat_mode and compat_map is not None:
            # logger.debug(f'Using {compat_map.__name__=} for {node_type=}')
            return compat_map(self, node_type, input_args, attrs, input_kwargs)

        node = self._make_node(node_type)

        if label is not None:
            node.label = label
            node.name = label

        if attrs is not None:
            for key_path, val in attrs.items():
                keys = key_path.split(".")
                obj = node
                for key in keys[:-1]:
                    obj = getattr(obj, key)
                setattr(obj, keys[-1], val)

        if node_type in [
            Nodes.VoronoiTexture,
            Nodes.NoiseTexture,
            Nodes.WaveTexture,
            Nodes.WhiteNoiseTexture,
            Nodes.MusgraveTexture,
        ]:
            if not (input_args != [] or "Vector" in input_kwargs):
                w = f"{self.node_group=}, no vector input for noise texture in specified"
                if self.input_consistency_forced:
                    logger.debug(
                        f"{w}, it is fixed automatically by using position for consistency"
                    )
                    if self.node_group.type == "SHADER":
                        input_kwargs["Vector"] = self.new_node("ShaderNodeNewGeometry")
                    else:
                        input_kwargs["Vector"] = self.new_node(Nodes.InputPosition)
                else:
                    pass  # print(f"{w}, please fix it if you found it causes inconsistency")

        input_keyval_list = list(enumerate(input_args)) + list(input_kwargs.items())
        for input_socket_name, input_item in input_keyval_list:
            if input_item is None:
                continue
            if node_type == Nodes.GroupOutput:
                assert not isinstance(input_socket_name, int), (
                    f"Attribute inputs to group output nodes must be given a string "
                    f"name, integer name "
                    f"{input_socket_name} will not suffice"
                )
                assert not isinstance(
                    input_item, list
                ), "Multi-input sockets to GroupOutput nodes are impossible"
                if input_socket_name not in node.inputs:
                    nodeclass = map_socket(infer_output_socket(input_item).bl_idname)
                    self.node_group.interface.new_socket(
                        name=input_socket_name, in_out="OUTPUT", socket_type=nodeclass
                    )
                    assert (
                        input_socket_name in node.inputs
                        and node.inputs[input_socket_name].enabled
                    )

            input_socket = infer_input_socket(node, input_socket_name)
            self.connect_input(input_socket, input_item)

        if expose_input is not None:
            names = [v[1] for v in expose_input]
            uniq, counts = np.unique(names, return_counts=True)
            if (counts > 1).any():
                raise ValueError(
                    f"expose_input with {names} features duplicate entries. in bl3.5 this is invalid."
                )
            for inp in expose_input:
                nodeclass, name, val = inp
                self.expose_input(name, val=val, dtype=nodeclass)

        return node

    def expose_input(
        self, name, val=None, attribute=None, dtype=None, use_namednode=False
    ):
        """
        Expose an input to the nodegroups interface, making it able to be specified externally

        If this nodegroup is
        """

        if attribute is not None:
            if self.modifier is None and val is None:
                raise ValueError(
                    "Attempted to use expose_input(attribute=...) on NodeWrangler constructed from "
                    "node_tree.\n"
                    "Please construct by passing in the modifier instead, or specify expose_input(val=..., "
                    "attribute=...) to provide a fallback"
                )

        if use_namednode:
            assert dtype is not None
            return self.new_node(
                Nodes.NamedAttribute, [name], attrs={"data_type": dtype}
            )

        group_input = self.new_node(Nodes.GroupInput)  # will reuse singleton

        if name in ng_inputs(self.node_group):
            assert len([o for o in group_input.outputs if o.name == name]) == 1
            return group_input.outputs[name]

        # Infer from args what type of node input to make (NodeSocketFloat / NodeSocketVector / etc)
        nodeclass = self._infer_nodeclass_from_args(dtype, val)
        inp = self.node_group.interface.new_socket(
            name, in_out="INPUT", socket_type=nodeclass
        )

        def prepare_cast(to_type, val):
            # cast val only when necessary, and only when type(val) wont crash
            if to_type not in [bpy.types.bpy_prop_array, bpy.types.bpy_prop]:
                val = to_type(val)
            return val

        if val is not None:
            if not hasattr(inp, "default_value") or inp.default_value is None:
                raise ValueError(
                    f"expose_input() recieved {val=} but inp {inp} does not expect a default_value"
                )
            inp.default_value = prepare_cast(type(inp.default_value), val)

        if self.modifier is not None:
            id = inp.identifier
            if val is not None:
                curr_mod_inp_val = self.modifier[id]
                if hasattr(curr_mod_inp_val, "real"):
                    self.modifier[id] = prepare_cast(type(curr_mod_inp_val.real), val)
            if attribute is not None:
                self.modifier[f"{id}_attribute_name"] = attribute
                self.modifier[f"{id}_use_attribute"] = 1

        assert len([o for o in group_input.outputs if o.name == name]) == 1
        return group_input.outputs[name]

    @staticmethod
    def _infer_nodeclass_from_args(dtype, val=None):
        """
        We will allow the user to request a 'dtype' that is a python type, blender datatype, or blender
        nodetype.

        All of these must be mapped to some node_info.NODECLASS in order to create a node.
        Optionally, we can try to infer a nodeclass from the type of a provided `val`
        """

        if dtype is None:
            if val is not None:
                datatype = node_info.PYTYPE_TO_DATATYPE[type(val)]
            else:
                # assert attribute is not None
                datatype = "FLOAT_VECTOR"
            return node_info.DATATYPE_TO_NODECLASS[datatype]
        else:
            if dtype in node_info.NODECLASSES:
                return dtype
            else:
                if dtype in node_info.NODETYPE_TO_DATATYPE:
                    datatype = node_info.NODETYPE_TO_DATATYPE[dtype]
                elif dtype in node_info.PYTYPE_TO_DATATYPE:
                    datatype = node_info.PYTYPE_TO_DATATYPE[dtype]
                else:
                    raise ValueError(f"Could not parse {dtype=}")
                return node_info.DATATYPE_TO_NODECLASS[datatype]

    def _update_socket(self, input_socket, input_item):
        output_socket = infer_output_socket(input_item)

        if output_socket is None and hasattr(input_socket, "default_value"):
            # we couldnt parse the inp to be any kind of node, it must be a default_value for us to assign
            try:
                input_socket.default_value = input_item
                return
            except TypeError as e:
                print(
                    f"TypeError while assigning {input_item=} as default_value for {input_socket.name}"
                )
                raise e

        self.links.new(output_socket, input_socket)

    def connect_input(self, input_socket, input_item):
        if isinstance(input_item, list) and any(
            infer_output_socket(i) is not None for i in input_item
        ):
            if not input_socket.is_multi_input:
                raise ValueError(
                    f"list of sockets {input_item} is not valid to connect to {input_socket} as it is not a "
                    f"valid multi-input socket"
                )
            for inp in input_item:
                self._update_socket(input_socket, inp)
        else:
            self._update_socket(input_socket, input_item)

    def _make_node(self, node_type):
        if node_type in node_info.SINGLETON_NODES:
            # for nodes in the singletons list, we should reuse an existing one if it exists
            try:
                node = next(n for n in self.nodes if n.bl_idname == node_type)
            except StopIteration:
                node = self.nodes.new(node_type)
        elif node_type in bpy.data.node_groups:
            assert node_type not in [
                getattr(Nodes, k) for k in dir(Nodes) if not k.startswith("__")
            ], (
                f"Someone has made a node_group named {node_type}, which is also the name of a "
                f"regular node"
            )

            nodegroup_type = {
                "ShaderNodeTree": "ShaderNodeGroup",
                "GeometryNodeTree": "GeometryNodeGroup",
                "CompositorNodeTree": "CompositorNodeGroup",
            }[bpy.data.node_groups[node_type].bl_idname]

            node = self.nodes.new(nodegroup_type)
            node.node_tree = bpy.data.node_groups[node_type]
        else:
            node = self.nodes.new(node_type)

        return node

    def get_position_translation_seed(self, i):
        if i not in self.position_translation_seed:
            self.position_translation_seed[i] = random_vector3()
        return self.position_translation_seed[i]

    def find(self, name):
        return [n for n in self.nodes if name in type(n).__name__]

    def find_recursive(self, name):
        return [(self, n) for n in self.find(name)] + sum(
            (
                NodeWrangler(n.node_tree).find_recursive(name)
                for n in self.nodes
                if n.type == "GROUP"
            ),
            [],
        )

    def find_from(self, to_socket):
        return [l for l in self.links if l.to_socket == to_socket]

    def find_from_recursive(self, name):
        return [(self, n) for n in self.find(name)] + sum(
            (
                NodeWrangler(n.node_tree).find_from_recursive(name)
                for n in self.nodes
                if n.type == "GROUP"
            ),
            [],
        )

    def find_to(self, from_socket):
        return [l for l in self.links if l.from_socket == from_socket]

    def find_to_recursive(self, name):
        return [(self, n) for n in self.find(name)] + sum(
            (
                NodeWrangler(n.node_tree).find_to_recursive(name)
                for n in self.nodes
                if n.type == "GROUP"
            ),
            [],
        )

    @staticmethod
    def is_socket(node):
        return isinstance(node, bpy.types.NodeSocket) or isinstance(
            node, bpy.types.Node
        )

    @staticmethod
    def is_vector_socket(node):
        if isinstance(node, bpy.types.Node):
            node = [o for o in node.outputs if o.enabled][0]
        if isinstance(node, bpy.types.NodeSocket):
            return "VECTOR" in node.type
        return isinstance(node, Iterable)

    def add2(self, *nodes):
        return self.new_node(Nodes.VectorMath, list(nodes))

    def multiply2(self, *nodes):
        return self.new_node(Nodes.VectorMath, list(nodes), {"operation": "MULTIPLY"})

    def scalar_add2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes))

    def scalar_max2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes), {"operation": "MAXIMUM"})

    def scalar_multiply2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes), {"operation": "MULTIPLY"})

    def sub2(self, *nodes):
        return self.new_node(Nodes.VectorMath, list(nodes), {"operation": "SUBTRACT"})

    def divide2(self, *nodes):
        return self.new_node(Nodes.VectorMath, list(nodes), {"operation": "DIVIDE"})

    def scalar_sub2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes), {"operation": "SUBTRACT"})

    def scalar_divide2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes), {"operation": "DIVIDE"})

    def power(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes), {"operation": "POWER"})

    def add(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.add2(*nodes)
        return self.add2(nodes[0], self.add(*nodes[1:]))

    def multiply(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.multiply2(*nodes)
        return self.multiply2(nodes[0], self.multiply(*nodes[1:]))

    def scalar_add(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.scalar_add2(*nodes)
        return self.scalar_add2(nodes[0], self.scalar_add(*nodes[1:]))

    def scalar_max(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.scalar_max2(*nodes)
        return self.scalar_max2(nodes[0], self.scalar_max(*nodes[1:]))

    def scalar_multiply(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.scalar_multiply2(*nodes)
        return self.scalar_multiply2(nodes[0], self.scalar_multiply(*nodes[1:]))

    sub = sub2
    scalar_sub = scalar_sub2
    divide = divide2
    scalar_divide = scalar_divide2

    def scale(self, *nodes):
        x, y = nodes
        if self.is_vector_socket(y):
            x, y = y, x
        elif isinstance(y, Iterable):
            x, y = y, x
        return self.new_node(
            Nodes.VectorMath,
            input_kwargs={"Vector": x, "Scale": y},
            attrs={"operation": "SCALE"},
        )

    def dot(self, *nodes):
        return self.new_node(
            Nodes.VectorMath, attrs={"operation": "DOT_PRODUCT"}, input_args=nodes
        )

    def math(self, node_type, *nodes):
        return self.new_node(
            Nodes.Math, attrs={"operation": node_type}, input_args=nodes
        )

    def vector_math(self, node_type, *nodes):
        return self.new_node(
            Nodes.VectorMath, attrs={"operation": node_type}, input_args=nodes
        )

    def boolean_math(self, node_type, *nodes):
        return self.new_node(
            Nodes.BooleanMath, attrs={"operation": node_type}, input_args=nodes
        )

    def compare(self, node_type, *nodes):
        return self.new_node(
            Nodes.Compare, attrs={"operation": node_type}, input_args=nodes
        )

    def compare_direction(self, node_type, x, y, angle):
        return self.new_node(
            Nodes.Compare,
            input_kwargs={"A": x, "B": y, "Angle": angle},
            attrs={"data_type": "VECTOR", "mode": "DIRECTION", "operation": node_type},
        )

    def bernoulli(self, prob, seed=None):
        if seed is None:
            seed = np.random.randint(1e5)
        return self.new_node(
            Nodes.RandomValue,
            input_kwargs={"Probability": prob, "Seed": seed},
            attrs={"data_type": "BOOLEAN"},
        )

    def uniform(self, low=0.0, high=1.0, seed=None, data_type="FLOAT"):
        if seed is None:
            seed = np.random.randint(1e5)
        if isinstance(low, Iterable):
            data_type = "FLOAT_VECTOR"
        return self.new_node(
            Nodes.RandomValue,
            input_kwargs={"Min": low, "Max": high, "Seed": seed},
            attrs={"data_type": data_type},
        )

    def combine(self, x, y, z):
        return self.new_node(Nodes.CombineXYZ, [x, y, z])

    def separate(self, x):
        return self.new_node(Nodes.SeparateXYZ, [x]).outputs

    def switch(self, pred, true, false, input_type="FLOAT"):
        return self.new_node(
            Nodes.Switch,
            input_kwargs={"Switch": pred, "True": true, "False": false},
            attrs={"input_type": input_type},
        )

    def vector_switch(self, pred, true, false):
        return self.new_node(
            Nodes.Switch,
            input_kwargs={"Switch": pred, "True": true, "False": false},
            attrs={"input_type": "VECTOR"},
        )

    def geometry2point(self, geometry):
        return self.new_node(
            Nodes.MergeByDistance,
            input_kwargs={"Geometry": geometry, "Distance": 100.0},
        )

    def position2point(self, position):
        return self.new_node(
            Nodes.MeshLine, input_kwargs={"Count": 1, "Start Location": position}
        )

    def capture(self, geometry, attribute, attrs=None):
        if attrs is None:
            attrs = {}
        capture = self.new_node(
            Nodes.CaptureAttribute,
            input_kwargs={"Geometry": geometry, "Value": attribute},
            attrs=attrs,
        )
        return capture.outputs["Geometry"], capture.outputs["Attribute"]

    def musgrave(self, scale=10, vector=None):
        return self.new_node(
            Nodes.MapRange,
            [
                self.new_node(
                    Nodes.MusgraveTexture, [vector], input_kwargs={"Scale": scale}
                ),
                -1,
                1,
                0,
                1,
            ],
        )

    def curve2mesh(self, curve, profile_curve=None):
        return self.new_node(
            Nodes.SetShadeSmooth,
            [
                self.new_node(Nodes.CurveToMesh, [curve, profile_curve, True]),
                None,
                False,
            ],
        )

    def build_float_curve(self, x, anchors, handle="VECTOR"):
        float_curve = self.new_node(Nodes.FloatCurve, input_kwargs={"Value": x})
        c = float_curve.mapping.curves[0]
        for i, p in enumerate(anchors):
            if i < 2:
                c.points[i].location = p
            else:
                c.points.new(*p)
            c.points[i].handle_type = handle
        float_curve.mapping.use_clip = False
        return float_curve

    def build_case(self, value, inputs, outputs, input_type="FLOAT"):
        node = outputs[-1]
        for i, o in zip(inputs[:-1], outputs[:-1]):
            node = self.switch(self.compare("EQUAL", value, i), o, node, input_type)
        return node

    def build_index_case(self, inputs):
        return self.build_case(
            self.new_node(Nodes.Index), inputs + [-1], [True] * len(inputs) + [False]
        )
