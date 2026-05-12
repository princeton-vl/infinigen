"""Serialize a ComputeGraph to a human-readable JSON format."""

import dataclasses
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import procfunc as pf
from procfunc import compute_graph as cg

logger = logging.getLogger(__name__)


def _serialize_value(value: Any) -> Any:
    """Convert a non-Node constant to a JSON-serializable value."""

    if isinstance(value, np.random.Generator):
        return {"$type": "rng"}
    elif isinstance(value, type):
        return {"$type": "type", "name": value.__qualname__}
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.dtype):
        return {"$type": "dtype", "name": str(value)}
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, (pf.Color, pf.Vector, pf.Euler, pf.Quaternion, pf.Matrix)):
        return {
            "$type": type(value).__name__,
            "value": [round(x, 6) for x in value],
        }
    elif isinstance(value, Enum):
        return {"$enum": f"{type(value).__name__}.{value.name}"}
    elif isinstance(value, Path):
        return {"$type": "Path", "value": str(value)}
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: _serialize_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
        return {"$type": type(value).__name__, **fields}
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, (bool, int, float, str, type(None))):
        return value
    else:
        return {"$repr": repr(value)}


def _serialize_arg(arg: Any, node_ids: dict[int, str]) -> Any:
    """Serialize an arg that may be a Node reference, constant, or nested structure."""

    if isinstance(arg, cg.Node):
        node_id = node_ids.get(id(arg))
        if node_id is None:
            logger.warning(f"Node {arg} not found in node_ids")
            return {"$ref": "???"}
        return {"$ref": node_id}
    elif isinstance(arg, dict):
        return {k: _serialize_arg(v, node_ids) for k, v in arg.items()}
    elif isinstance(arg, (list, tuple)):
        return [_serialize_arg(v, node_ids) for v in arg]
    else:
        return _serialize_value(arg)


def _func_name(node: cg.FunctionCallNode) -> str:
    return f"{node.func.__module__}.{node.func.__qualname__}"


def _serialize_node(
    node: cg.Node,
    node_id: str,
    node_ids: dict[int, str],
) -> dict:
    args = [_serialize_arg(a, node_ids) for a in node.args]
    kwargs = {k: _serialize_arg(v, node_ids) for k, v in node.kwargs.items()}

    if isinstance(node, cg.FunctionCallNode):
        return {
            "id": node_id,
            "type": "FunctionCall",
            "func": _func_name(node),
            "args": args,
            "kwargs": kwargs,
        }
    elif isinstance(node, cg.MethodCallNode):
        return {
            "id": node_id,
            "type": "MethodCall",
            "method": node.method_name,
            "args": args,
            "kwargs": kwargs,
        }
    elif isinstance(node, cg.GetAttributeNode):
        return {
            "id": node_id,
            "type": "GetAttribute",
            "attribute": node.attribute_name,
            "args": args,
            "kwargs": kwargs,
        }
    elif isinstance(node, cg.SubgraphCallNode):
        return {
            "id": node_id,
            "type": "SubgraphCall",
            "subgraph": node.subgraph.name,
            "args": args,
            "kwargs": kwargs,
        }
    elif isinstance(node, cg.ProceduralNode):
        return {
            "id": node_id,
            "type": "Procedural",
            "node_type": node.node_type,
            "attrs": {k: _serialize_value(v) for k, v in node.attrs.items()},
            "args": args,
            "kwargs": kwargs,
        }
    elif isinstance(node, cg.MutatedArgumentNode):
        return {
            "id": node_id,
            "type": "MutatedArgument",
            "args": args,
            "kwargs": kwargs,
        }
    elif isinstance(node, cg.ConstantNode):
        return {
            "id": node_id,
            "type": "Constant",
            "value": _serialize_value(node.value),
        }
    elif isinstance(node, cg.InputPlaceholderNode):
        return {
            "id": node_id,
            "type": "InputPlaceholder",
            "name": node.input_name,
            "default": _serialize_value(node.default_value),
        }
    else:
        return {
            "id": node_id,
            "type": type(node).__name__,
            "args": args,
            "kwargs": kwargs,
        }


def _serialize_graph(
    graph: cg.ComputeGraph,
    node_ids: dict[int, str],
) -> dict:
    nodes = []
    for node in cg.traverse_depth_first(graph):
        nid = node_ids[id(node)]
        nodes.append(_serialize_node(node, nid, node_ids))

    outputs = _serialize_arg(graph.outputs.obj(), node_ids)

    return {
        "name": graph.name,
        "nodes": nodes,
        "outputs": outputs,
    }


def to_json(
    graph: cg.ComputeGraph,
    indent: int = 2,
) -> str:
    # Collect all graphs (top-level + nested subgraphs)
    all_graphs = list(cg.traverse_nested_graphs(graph))

    # Assign node IDs across all graphs
    node_ids: dict[int, str] = {}
    counter = 0
    for g in all_graphs:
        for node in cg.traverse_depth_first(g):
            if id(node) not in node_ids:
                node_ids[id(node)] = f"n{counter}"
                counter += 1

    graphs_data = [_serialize_graph(g, node_ids) for g in all_graphs]

    if len(graphs_data) == 1:
        result = graphs_data[0]
    else:
        result = {"graphs": graphs_data}

    return json.dumps(result, indent=indent)
