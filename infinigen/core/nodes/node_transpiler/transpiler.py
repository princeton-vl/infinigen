# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: primary author
# - Alejandro Newell, Lingjie Mei: bugfixes

import importlib
import keyword
import logging
import re
import runpy
from collections import OrderedDict
from pathlib import Path

import bpy
import mathutils
import numpy as np

from infinigen.assets.sim_objects import mapping
from infinigen.core.nodes.node_info import OUTPUT_NODE_IDS, SINGLETON_NODES, Nodes
from infinigen.core.nodes.node_wrangler import ng_inputs

logger = logging.getLogger(__name__)

VERSION = "2.7.1"
indent_string = " " * 4
LINE_LEN = 100

COMMON_ATTR_NAMES = ["data_type", "mode", "operation"]
VALUE_NODES = [Nodes.Value, Nodes.Vector, Nodes.RGB, Nodes.InputColor, Nodes.Integer]

UNIVERSAL_ATTR_NAMES = set(
    [
        "show_preview",
        "__module__",
        "is_registered_node_type",
        "bl_rna",
        "poll",
        "name",
        "internal_links",
        "dimensions",
        "parent",
        "bl_width_max",
        "label",
        "input_template",
        "show_texture",
        "rna_type",
        "width_hidden",
        "show_options",
        "location",
        "outputs",
        "use_custom_color",
        "__doc__",
        "width",
        "bl_width_default",
        "inputs",
        "bl_idname",
        "socket_value_update",
        "bl_width_min",
        "color",
        "bl_height_max",
        "__slots__",
        "select",
        "mute",
        "bl_height_default",
        "bl_static_type",
        "bl_height_min",
        "height",
        "bl_label",
        "bl_icon",
        "hide",
        "output_template",
        "poll_instance",
        "draw_buttons_ext",
        "type",
        "bl_description",
        "draw_buttons",
        "update",
    ]
)

SPECIAL_CASE_ATTR_NAMES = set(
    [
        "color_ramp",
        "mapping",
        "vector",
        "color",
        "integer",
        "texture_mapping",
        "color_mapping",
        "image_user",
        "interface",
        "node_tree",
        "tag_need_exec",
    ]
)


def node_attrs_available(node):
    attrs = set(node.__dir__())
    attrs = attrs.difference(UNIVERSAL_ATTR_NAMES)
    attrs = attrs.difference(SPECIAL_CASE_ATTR_NAMES)
    logging.info(node.name, attrs)
    return attrs


def indent(s):
    return indent_string + s.strip().replace("\n", f"\n{indent_string}")


def prefix(dependencies_used) -> str:
    fixed_prefix = (
        "import bpy\n"
        "import mathutils\n"
        "from numpy.random import uniform, normal, randint\n"
        "from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler\n"
        "from infinigen.core.nodes import node_utils\n"
        "from infinigen.core import surface\n"
        "from infinigen.core.placement.factory import AssetFactory\n"
        "from infinigen.core.util import blender as butil\n"
        "from infinigen.core.util.paths import blueprint_path_completion\n"
        "from infinigen.core.sim.exporters import factory\n"
    )

    deps_table = [
        (ng_name, name_used[0])
        for ng_name, name_used in dependencies_used.items()
        if name_used[1]
    ]
    module_names = set(d[1] for d in deps_table)
    deps_by_module = {n: [d[0] for d in deps_table if d[1] == n] for n in module_names}
    deps_prefix_lines = [
        f"from {name} import {', '.join(ngnames)}"
        for name, ngnames in deps_by_module.items()
    ]

    return fixed_prefix + "\n" + "\n".join(deps_prefix_lines)


def postfix(funcnames, targets):
    header = "def apply(obj, selection=None, apply=False, **kwargs):\n"
    body = ""

    for funcname, target in zip(funcnames, targets):
        idname = get_node_tree(target).bl_idname
        if idname == "GeometryNodeTree":
            body += f"surface.add_geomod(obj, {funcname}, apply=apply, selection=selection, attributes=[], **kwargs)\n"
        elif idname == "ShaderNodeTree":
            body += f"surface.add_material(obj, {funcname}, selection=selection)\n"
        else:
            raise ValueError(
                f"Postfix couldnt handle {idname=}, please contact the developer"
            )

    return header + indent(body)


def repr_iter_val(v):
    match v:
        case list():
            return represent_list(v)
        case str():
            return v  # String are assumed to be code variables to get passed through
        case _:
            return represent_default_value(v, simple=True)


def represent_default_value(val, simple=True):
    """
    Attempt to create a python expression to represent val, which was the .default_value of some .input node

    Unless simple=True, we may encounter things such as Materials which require transpiling.
    """

    code = ""
    new_transpiler_targets = {}

    match val:
        case str() | int() | bool() | set():
            code = repr(val)
        case float():
            code = f"{val:.4f}"
        case (
            tuple()
            | bpy.types.bpy_prop_array()
            | mathutils.Vector()
            | mathutils.Euler()
        ):
            code = represent_tuple(tuple(val))
        case bpy.types.Object() | bpy.types.Image():
            code = repr(val)
        case bpy.types.Collection():
            logger.warning(
                f"Encountered collection {repr(val.name)} as a default_value - please edit the code to remove this dependency on a collection already existing"
            )
            code = f"bpy.data.collections[{repr(val.name)}]"
        case bpy.types.Material():
            if val.use_nodes:
                funcname = get_func_name(val)
                new_transpiler_targets[funcname] = val
                code = f"surface.shaderfunc_to_material({funcname})"
            else:
                logger.warning(f"Encountered material {val} but it has use_nodes=False")
                code = repr(val)
        case None:
            logger.warning(
                "Transpiler introduced a None into result script, this may not have been intended by the user"
            )
            code = "None"
        case _:
            raise ValueError(
                f"represent_default_value was unable to handle {val=} with type {type(val)}, please contact the developer"
            )

    assert isinstance(code, str)

    if simple:
        if len(new_transpiler_targets) != 0:
            raise ValueError(
                f"Encountered {val=} while trying to represent_default_value with simple=True, please contact the developer"
            )
        return code
    else:
        return code, new_transpiler_targets


def has_default_value_changed(node_tree, node, value):
    """
    Utility to check whether a given `value` of a `node` has been changed at all
    from its default, and hence to check whether we need to bother to add code to
    set its value.

    `value` is either an input socket of the node with a default_value, or just a
    python variable name string to check
    """

    def compare(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.all(a == b)

    temp_default_node = node_tree.nodes.new(node.bl_idname)
    if node.bl_idname.endswith("NodeGroup"):
        temp_default_node.node_tree = node.node_tree

    if isinstance(value, bpy.types.NodeSocket):
        assert get_connected_link(node_tree, input_socket=value) is None
        assert hasattr(value, "default_value")

        observed_val = value.default_value
        default_socket = [
            i for i in temp_default_node.inputs if i.identifier == value.identifier
        ][0]
        default_val = default_socket.default_value
        has_changed = not compare(observed_val, default_val)
    elif isinstance(value, str):
        assert hasattr(node, value)
        has_changed = not compare(
            getattr(node, value), getattr(temp_default_node, value)
        )
    else:
        node_tree.nodes.remove(temp_default_node)
        raise ValueError(f"Unexpected input {value=} in has_default_value_changed")

    node_tree.nodes.remove(temp_default_node)

    return has_changed


def special_case_colorramp(node, varname):
    assert node.bl_idname == Nodes.ColorRamp

    code = ""
    cramp = node.color_ramp

    if cramp.interpolation != "LINEAR":  # dont bother if left at default
        code += f'{varname}.color_ramp.interpolation = "{cramp.interpolation}"\n'

    # add code to add new elements if need be
    if len(cramp.elements) > 2:
        n_elements_needed = len(cramp.elements) - 2  # starts with 2 by default
        for _ in range(n_elements_needed):
            code += f"{varname}.color_ramp.elements.new(0)\n"

    for i, ele in enumerate(cramp.elements):
        code += f"{varname}.color_ramp.elements[{i}].position = {ele.position:.4f}\n"
        code += (
            f"{varname}.color_ramp.elements[{i}].color = {represent_list(ele.color)}\n"
        )

    return code


def special_case_curve(node, varname):
    assert node.bl_idname in [Nodes.FloatCurve, Nodes.RGBCurve, Nodes.VectorCurve]

    code = ""
    for i, c in enumerate(node.mapping.curves):
        points = [tuple(p.location) for p in c.points]
        args = [f"{varname}.mapping.curves[{i}]", represent_list(points)]
        if not all(p.handle_type == "AUTO" for p in c.points):
            args.append(f"handles={repr([p.handle_type for p in c.points])}")
        code += f"node_utils.assign_curve({', '.join(args)})\n"
    return code


def represent_label_value_expression(expression):
    """
    When the user puts "var ~ N(0, 1)" or something of the like as the label of their node,
    this function parses everything after the ~ into a python expression

    Must be of form {operation}({argument})

    Valid operations:
    - U, uniform
    - N, normal

    Valid arguments: str, float, list of float

    """

    def parse_arg(arg):
        arg = arg.strip(" ,")

        if arg.strip("'\"").isalpha():
            return arg.strip("'\"")

        try:
            return float(arg)
        except ValueError:
            pass

        if arg.startswith("["):
            vals = arg.strip("[]").split(",")
            return [parse_arg(v) for v in vals]
        else:
            raise ValueError(
                f"represent_label_value_expression had invalid argument {arg}"
            )

    matched_chars = {"'": "'", '"': '"', "[": "]"}

    def parse_args(arg_str):
        args = []

        remaining = arg_str
        while len(remaining) != 0:
            remaining = remaining.strip(", ")

            search_for = matched_chars.get(remaining[0], ",")
            next_idx = (
                remaining[1:].index(search_for) + 1
                if search_for in remaining
                else len(remaining)
            )
            arg = remaining[: next_idx + 1]
            remaining = remaining[next_idx + 1 :]

            try:
                args.append(parse_arg(arg))
            except ValueError:
                raise ValueError(
                    f"Could not parse node label expression {repr(arg_str)}, item {repr(arg)} was not a valid argument"
                )

        return args

    op, args = expression.split("(")
    op = op.strip()
    args = parse_args(args.strip(")"))

    if op in ["N", "normal", "U", "uniform", "R", "randint"]:
        if not len(args) == 2:
            raise ValueError(
                f"In {expression=}, expected 2 arguments, got {len(args)} instead"
            )
        funcname = {
            "N": "normal",
            "normal": "normal",
            "U": "uniform",
            "uniform": "uniform",
            "R": "randint",
            "randint": "randint",
        }[op]
        args = ", ".join(repr(a) for a in args)
        return f"{funcname}({args})"
    else:
        raise ValueError(
            f"Failed to represent_label_value_expression({expression=}), unrecognized {op=}"
        )


def special_case_value(node, varname):
    code = ""

    # Determine value expression
    if node.label and "~" in node.label:
        labelname, expression = node.label.split("~")
        value_expr = represent_label_value_expression(expression)
    else:
        if node.bl_idname in [Nodes.Value, Nodes.RGB]:
            val = node.outputs[0].default_value
        elif node.bl_idname == Nodes.InputColor:
            val = node.color
        elif node.bl_idname == Nodes.Vector:
            val = node.vector
        elif node.bl_idname == Nodes.Integer:
            val = node.integer
        else:
            raise ValueError(
                f"special_case_value called on unrecognized {node.bl_idname=}"
            )

        value_expr = represent_default_value(val, simple=True)

    # set value
    if node.bl_idname in [Nodes.Value, Nodes.RGB]:
        code += f"{varname}.outputs[0].default_value = {value_expr}\n"
    elif node.bl_idname == Nodes.Vector:
        code += f"{varname}.vector = {value_expr}\n"
    elif node.bl_idname == Nodes.InputColor:
        code += f"{varname}.color = {value_expr}\n"
    elif node.bl_idname == Nodes.Integer:
        code += f"{varname}.integer = {value_expr}\n"
    else:
        raise ValueError(f"special_case_value called on unrecognized {node.bl_idname=}")

    return code


def get_connected_link(node_tree, input_socket):
    links = [l for l in node_tree.links if l.to_socket == input_socket]
    return None if len(links) == 0 else links


def create_attrs_dict(node_tree, node):
    """
    Create a dict to be passed into the attrs=... kwarg of NodeWrangler.new_node

    IE, the dict should represent all the properties of `node` that need to be set
    but are NOT part of node.inputs - things like setting what operation a math node
    does, or how a mix node should mix its inputs.
    """

    attr_names = node_attrs_available(node)

    for a in COMMON_ATTR_NAMES:
        if hasattr(node, a) and a not in attr_names:
            raise ValueError(
                f"{node.bl_idname=} has attr {repr(a)} but it is not listed in node_info.NODE_ATTRS_AVAILABLE, please add it to avoid incorrect behavior"
            )

    # Check that the dict is correct / doesnt contain typos
    for a in attr_names:
        if not hasattr(node, a):
            nodetype_expr = get_nodetype_expression(node)
            raise ValueError(
                f"attrs_available[{nodetype_expr} is incorrect, real node {node} did not have an attribute '{a}' - please contact the developer"
            )

    # Filter out the attrs which havent been changed from their default values -
    #    clearly we dont need to set these ones manually, so we can save code verbosity
    attr_names = [
        a for a in attr_names if has_default_value_changed(node_tree, node, a)
    ]

    return {
        repr(k): represent_default_value(getattr(node, k), simple=True)
        for k in attr_names
    }


def process_single_input(node_tree, node, input_socket, input_name, memo):
    """
    Process a single input socket and return either:
    - None if the input should be skipped
    - (input_expression, code, new_targets) tuple if the input should be included
    """
    if not input_socket.enabled:
        return None

    links = get_connected_link(node_tree, input_socket)
    if links is None:
        if not hasattr(input_socket, "default_value"):
            return None
        if not has_default_value_changed(node_tree, node, input_socket):
            return None
        input_expression, targets = represent_default_value(
            input_socket.default_value, simple=False
        )
        return (input_expression, "", targets)

    # Process all valid links to this input
    all_expressions = []
    all_code = []
    all_targets = {}

    for link in links:
        if not link.from_socket.enabled or not link.to_socket.enabled:
            logger.warning(
                f"Transpiler encountered {'from' if not link.from_socket.enabled else 'to'} disabled socket "
                f"{link.from_socket if not link.from_socket.enabled else link.to_socket}, ignoring it"
            )
            continue

        input_varname, input_code, targets = create_node(
            node_tree, link.from_node, memo
        )
        all_code.append(input_code)
        all_targets.update(targets)

        if len(link.from_node.outputs) == 1:
            input_expression = input_varname
        else:
            socket_name = link.from_socket.name
            input_expression = f'{input_varname}.outputs["{socket_name}"]'

            # Catch shared socket output names
            if (
                link.from_node.outputs[socket_name].identifier
                != link.from_socket.identifier
            ):
                from_idx = [
                    i
                    for i, o in enumerate(link.from_node.outputs)
                    if o.identifier == link.from_socket.identifier
                ][0]
                input_expression = f"{input_varname}.outputs[{from_idx}]"

        all_expressions.append(input_expression)

    if not all_expressions:
        return None

    return (
        all_expressions[0] if len(all_expressions) == 1 else all_expressions,
        "".join(all_code),
        all_targets,
    )


def combine_input_results(node, processed_inputs):
    """
    Combine the results of processing multiple inputs into the final inputs_dict and code.
    processed_inputs is a list of (idx, name, result) tuples where result is either None or (expr, code, targets)
    """
    inputs_dict = {}
    all_code = []
    all_targets = {}

    def update_inputs(i, k, v):
        is_input_name_unique = [socket.name for socket in node.inputs].count(k) == 1
        k = repr(k) if is_input_name_unique else i
        if k not in inputs_dict:
            inputs_dict[k] = v
        else:
            if not isinstance(inputs_dict[k], list):
                inputs_dict[k] = [inputs_dict[k]]
            inputs_dict[k].append(v)

    for i, input_name, result in processed_inputs:
        if result is None:
            continue
        input_expression, code, targets = result
        update_inputs(i, input_name, input_expression)
        if code:
            all_code.append(code)
        all_targets.update(targets)

    return inputs_dict, "".join(all_code), all_targets


def create_inputs_dict(node_tree, node, memo):
    """
    Process all inputs of a node and return a dict mapping input names to their values,
    along with any generated code and new transpile targets.
    """
    processed = []
    for i, (input_name, input_socket) in enumerate(node.inputs.items()):
        result = process_single_input(node_tree, node, input_socket, input_name, memo)
        processed.append((i, input_name, result))

    return combine_input_results(node, processed)


def represent_list(inputs, spacing=" "):
    inputs = [repr_iter_val(x) for x in inputs]
    return "[" + f",{spacing}".join(inputs) + "]"


def represent_tuple(inputs, spacing=" "):
    inputs = [repr_iter_val(x) for x in inputs]
    for x in inputs:
        assert isinstance(x, str), x
    return "(" + f",{spacing}".join(inputs) + ")"


def represent_dict(inputs_dict, spacing=" "):
    vals = f",{spacing}".join(
        f"{k}: {repr_iter_val(v)}" for k, v in inputs_dict.items()
    )
    return "{" + vals + "}"


def get_varname(node, taken):
    """
    Choose a sensible python variable name to represent `node`,
    notably one which isnt in the list of already used variable names `taken`
    """

    if node.label:
        name = node.label.split("~")[
            0
        ].strip()  # remove any allowed postprocessor flags
        name = snake_case(name)
    elif hasattr(node, "operation"):
        # name the math nodes after their operations, for readability
        name = snake_case(node.operation.lower())
    elif node.bl_idname == "GeometryNodeGroup":
        name = snake_case(node.node_tree.name.lower())
    else:
        # for all other nodes, use the node.name, which should be unique to this node
        name, *rest = node.name.split(".")
        name = name.lower().replace(" ", "_")

        if len(rest) > 0:
            assert len(rest) == 1
            name += "_" + str(int(rest[0]))

    name = re.sub("[^0-9a-zA-Z_]+", "_", name)
    name = re.sub("_+", "_", name)
    name = name.strip("_")

    if keyword.iskeyword(name):
        name = "op_" + name

    if name in taken:
        i = 1
        while f"{name}_{i}" in taken:
            i += 1
        name = f"{name}_{i}"

    return name


def get_nodetype_expression(node):
    """
    Produce a python expression to be passed into the node_type input of
    NodeWrangler.new_node.
    IE, return either the node.bl_idname, or an alias for that name if one exists
    """

    id = node.bl_idname

    lookup = {getattr(Nodes, k): k for k in dir(Nodes) if not k.startswith("__")}

    if id in lookup:
        return f"Nodes.{lookup[id]}"
    elif id.endswith("NodeGroup"):
        return repr(node.node_tree.name)
    else:
        node_name = node.name.split(".")[0].replace(" ", "")
        logger.warning(
            f'Please add an alias for "{id}" in nodes.node_info.Nodes.'
            f"\n\t Suggestion: {node_name} = {repr(id)}"
        )
        return repr(id)


def create_node(node_tree, node, memo):
    if node.name in memo:
        return memo[node.name], "", {}

    idname = node.bl_idname
    if idname in SINGLETON_NODES:
        for n in memo:
            if node_tree.nodes[n].bl_idname == idname:
                return memo[n], "", {}

    code = ""
    new_transpile_targets = {}

    new_node_args = []

    if node.bl_idname.endswith("NodeGroup"):
        # node group will be transpiled to a function, then the funcname will be mapped to the nodegroup name by a decorator
        funcname = get_func_name(node)
        new_transpile_targets[funcname] = node
        nodetype_expr = f"{funcname}().name"
    else:
        nodetype_expr = get_nodetype_expression(node)
    new_node_args.append(nodetype_expr)

    # Add code to connect up any input nodes
    inputs_dict, inputs_code, targets = create_inputs_dict(node_tree, node, memo)
    new_transpile_targets.update(targets)
    code += inputs_code
    if len(inputs_dict) > 0:
        new_node_args.append(f"input_kwargs={represent_dict(inputs_dict)}")

    if node.label:
        new_node_args.append(f"label={repr(node.label)}")

    # Special case: input node
    if node.bl_idname == Nodes.GroupInput:
        all_inps = []
        for inp in ng_inputs(node_tree).values():
            repr_val, targets = (
                represent_default_value(inp.default_value, simple=False)
                if hasattr(inp, "default_value")
                else (None, {})
            )
            new_transpile_targets.update(targets)
            all_inps.append(
                f"({repr(inp.bl_socket_idname)}, {repr(inp.name)}, {repr_val})"
            )

        args = represent_list(all_inps, spacing="\n" + 2 * indent_string)
        new_node_args.append(f"expose_input={args}")

    # Add code to set the correct 'attrs', ie set the math operations
    attrs_dict = create_attrs_dict(node_tree, node)
    if len(attrs_dict) > 0:
        new_node_args.append(f"attrs={represent_dict(attrs_dict)}")

    # Compose the final nw.new_node() function call
    varname = get_varname(node, taken=list(memo.values()))
    if sum(len(x) for x in new_node_args) > LINE_LEN:
        arg_sep = ",\n" + indent_string
    else:
        arg_sep = ", "
    new_node_args_str = arg_sep.join(new_node_args)
    code += f"{varname} = nw.new_node({new_node_args_str})\n"

    # Handle various special case nodes that dont behave like the others
    if node.bl_idname == Nodes.ColorRamp:
        code += special_case_colorramp(node, varname)
    elif node.bl_idname in (Nodes.FloatCurve, Nodes.RGBCurve, Nodes.VectorCurve):
        code += special_case_curve(node, varname)
    elif node.bl_idname in VALUE_NODES:
        code += special_case_value(node, varname)

    code += "\n"

    memo[node.name] = varname
    return varname, code, new_transpile_targets


def get_node_tree(target):
    """
    Blender stores the node tree as a either 'node_group' or 'node_tree' depending on what the target is
    """

    if hasattr(target, "bl_idname") and target.bl_idname.endswith("NodeGroup"):
        return target.node_tree
    elif isinstance(target, bpy.types.NodesModifier):
        return target.node_group
    elif isinstance(target, (bpy.types.Material, bpy.types.World, bpy.types.Scene)):
        return target.node_tree
    else:
        raise ValueError(
            f"Couldnt infer node tree from {target=}, {type(target)=}, please contact the developer"
        )


def write_function_body(target):
    """
    Construct a python function body which will produce the node_tree of the `target`
    """

    output_node_id = OUTPUT_NODE_IDS[type(target)]
    node_tree = get_node_tree(target)

    try:
        output_node = next(n for n in node_tree.nodes if n.bl_idname == output_node_id)
    except StopIteration:
        logging.info([n.bl_idname for n in node_tree.nodes])
        raise ValueError(
            f"Couldnt find expected {output_node_id=} for node tree type {node_tree.bl_idname=}"
        )

    memo = {}
    final_varname, code, new_transpile_targets = create_node(
        node_tree, output_node, memo
    )
    return code, new_transpile_targets


def snake_case(name):
    name = name.replace(" ", "_").replace(".", "_")
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()  # Camel to snake
    return name


def get_func_name(target):
    """
    Decide a python function name which will produce the `target`
    """

    node_tree = get_node_tree(target)

    if hasattr(target, "bl_idname") and target.bl_idname.endswith("NodeGroup"):
        # Nodegroup wrapper nodes dont have good names, use the name of the nodegroup itself
        category = "nodegroup"
        name = snake_case(target.node_tree.name)
    else:
        category = snake_case(node_tree.bl_idname).split("_")[0]
        name = snake_case(target.name)

    # attempt to make names nicer when transpiling already transpiled code
    if name.startswith(category):
        name = name[len(category) :]
    nogc = "(no_gc)"
    if name.endswith(nogc):
        name = name[: -len(nogc)]

    finalname = f"{category}_{name}"
    finalname = finalname.replace("__", "_").strip("_")

    return finalname


def transpile(orig_targets, module_dependencies=[]):
    code = ""

    # initialize all targets as un-processed
    orig_names = [get_func_name(t) for t in orig_targets]
    targets = OrderedDict((n, (t, False)) for n, t in zip(orig_names, orig_targets))

    available_dependencies = {}
    for module_name in module_dependencies:
        module = importlib.import_module(module_name)
        available_dependencies.update(
            {k: [module_name, False] for k in dir(module) if k.startswith("nodegroup_")}
        )
    logging.info(f"{available_dependencies.keys()=}")

    while any(not v[1] for v in targets.values()):
        funcname, (target, _) = next((k, v) for k, v in targets.items() if not v[1])

        if funcname in orig_names:
            logging.info(
                f"Transpiling initial target {orig_targets.index(target)} {repr(target)} as {funcname}()"
            )
        else:
            logging.info(f"Transpiling dependency {repr(target)} as {funcname}()")

        # create function definition
        new_code = ""
        if (
            hasattr(target, "bl_idname")
            and target.bl_idname.endswith("NodeGroup")
            or funcname == "geometry_nodes"
        ):
            new_code += f"@node_utils.to_nodegroup({repr(funcname)}, singleton=False, type={repr(get_node_tree(target).bl_idname)})\n"
        new_code += f"def {funcname}(nw: NodeWrangler):\n"

        new_code += (
            indent(f"# Code generated using version {VERSION} of the node_transpiler")
            + "\n\n"
        )

        # prepend new function to running code body
        function_body, new_targets = write_function_body(target)
        new_code += indent(function_body)
        code = new_code + "\n\n" + code

        targets[funcname] = (target, True)  # mark as finished
        for k, v in new_targets.items():
            if k in available_dependencies:
                logger.info(
                    f"Using {k} from dependency module {available_dependencies[k][0]} - assuming the definition is unchanged"
                )
                available_dependencies[k][1] = True  # remember to add it to imports
                continue  # dont actually generate code for it
            if k not in targets:
                targets[k] = (v, False)  # mark as needing to be transpiled

    return code, orig_names, available_dependencies


def clean_and_capitalize(input_string):
    """
    Upper cases the first letter of the string and uppercases
    the character after '.', '-', or '_' while removing these
    special characters from the string.
    """
    # Uppercase the character after ., -, or _
    cleaned_string = re.sub(
        r"[._-](\w)", lambda match: match.group(1).upper(), input_string
    )
    # Remove all special characters except alphanumerics
    cleaned_string = re.sub(r"[^A-Za-z0-9\s]", "", cleaned_string)
    # Capitalize the first character
    if cleaned_string:
        result = cleaned_string[0].upper() + cleaned_string[1:]
    else:
        result = cleaned_string  # Handle empty string case
    return result


def add_asset_to_file(file_path, asset_name, class_name, import_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    mapping = runpy.run_path(file_path)
    obj_class_map = mapping.get("OBJECT_CLASS_MAP", {})
    if asset_name in obj_class_map:
        logging.warning(
            f"The asset name '{asset_name}' already exists in OBJECT_CLASS_MAP."
        )
        return

    new_import = f"from {import_path}.{asset_name} import {class_name}\n"
    new_dict_entry = f'    "{asset_name}": {class_name},\n'

    updated_lines = []
    import_added = False
    dict_entry_added = False
    inside_object_map = False

    for line in lines:
        # Handle import insertion
        if "# add newly transpiled assets here" in line and not import_added:
            updated_lines.append(new_import)
            import_added = True

        updated_lines.append(line)

        # Handle dictionary insertion
        if "OBJECT_CLASS_MAP" in line:
            inside_object_map = True

        if (
            inside_object_map
            and "# add newly transpiled assets here" in line
            and not dict_entry_added
        ):
            updated_lines.insert(
                len(updated_lines) - 1, new_dict_entry
            )  # Insert before the comment
            dict_entry_added = True
            inside_object_map = False  # Reset after adding

    # Write the updated content back
    with open(file_path, "w") as file:
        file.writelines(updated_lines)


def transpile_object_to_sim_class(
    obj,
    module_dependencies=[],
    sim_blueprint=None,
    output_name=None,
    add_to_catalog=True,
):
    targets = []
    targets += [mod for mod in obj.modifiers if mod.type == "NODES"]
    logging.info(
        f"Found {len(targets)} initial transpile targets for object {repr(obj)}"
    )

    func_code, funcnames, dependencies_used = transpile(targets, module_dependencies)

    code = prefix(dependencies_used) + "\n\n"
    code += func_code + "\n\n"
    # code += postfix(funcnames, targets)
    # code += "\n\n"
    output_name = obj.name if output_name is None else output_name
    class_name = clean_and_capitalize(output_name) + "Factory"
    code += f"""
class {class_name}(AssetFactory):

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
        self.sim_blueprint = {f"blueprint_path_completion('{sim_blueprint}')"}

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        return {'{}'}

    def create_asset(self,
                     asset_params=None,
                     **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=True,
            node_group={funcnames[0]}(),
            ng_inputs=self.sample_parameters()
        )

        return obj
    """

    if add_to_catalog:
        mapping_path = Path(mapping.__file__)
        add_asset_to_file(
            file_path=mapping_path,
            asset_name=output_name,
            class_name=class_name,
            import_path="infinigen.assets.sim_objects",
        )

    logging.info("")  # newline once done for ease of reading the logs
    return code


def transpile_object(obj, module_dependencies=[]):
    targets = []
    targets += [mod for mod in obj.modifiers if mod.type == "NODES"]
    targets += [slot.material for slot in obj.material_slots if slot.material.use_nodes]
    logging.info(
        f"Found {len(targets)} initial transpile targets for object {repr(obj)}"
    )

    func_code, funcnames, dependencies_used = transpile(targets, module_dependencies)

    code = prefix(dependencies_used) + "\n\n"
    code += func_code + "\n\n"
    code += postfix(funcnames, targets)

    logging.info("")  # newline once done for ease of reading the logs
    return code


def transpile_world(module_dependencies=[], compositing=True, worldshader=True):
    targets = []
    if compositing and bpy.context.scene.use_nodes:
        targets.append(bpy.context.scene)
    if worldshader and bpy.context.scene.world.use_nodes:
        targets.append(bpy.context.scene.world)

    funccode, funcnames, dependencies_used = transpile(targets, module_dependencies)

    code = prefix(dependencies_used) + "\n\n" + funccode

    return code
