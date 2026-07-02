# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Jack Nugent

"""Compute-graph extraction and stat summaries

Representation after setup:

- choice_probs[choice_id] gives branch probabilities for one discrete choice
- choice_branches[choice_id] gives the branch ids for that choice
- branch_vars[branch_id] gives the variables activated by that branch
- var_kind[var_id] is "choice" or "continuous"

This code handles:
- extracting this model from a compute graph
- topologically reindexing variables so the solver can use bitmasks
- calling either the plain recursion or the memoized DP
"""

from __future__ import annotations

import logging
from graphlib import TopologicalSorter

import procfunc.compute_graph as cg
from procfunc.control import choice
from procfunc.tracer import TraceLevel
from procfunc.transforms.distribution import as_distribution
from procfunc.util.pytree import PyTree

from infinigen2.util.codestats.compute import (
    cyclomatic_complexity,
    solve_dp,
    solve_tree,
)

logger = logging.getLogger(__name__)


def _iter_node_children(node: cg.Node):
    for child in PyTree((node.args, node.kwargs)).children:
        if isinstance(child, cg.Node):
            yield child


def _is_choice_node(node: cg.Node) -> bool:
    if not isinstance(node, cg.FunctionCallNode):
        return False
    return getattr(node.func, "__wrapped__", node.func) is choice


def _is_continuous_node(node: cg.Node) -> bool:
    return as_distribution(node) is not None


def _normalize_choice_probabilities(
    raw_options: list[tuple[object, float]],
) -> list[float]:
    if len(raw_options) == 0:
        raise ValueError("choice node has no options")

    weights: list[float] = []
    for _result, weight in raw_options:
        weight = float(weight)
        if weight < 0.0:
            raise ValueError(f"choice option had negative weight: {weight}")
        weights.append(weight)

    total = sum(weights)
    if total <= 0.0:
        raise ValueError(f"choice option weights must sum to > 0, got {weights}")

    return [weight / total for weight in weights]


def _scan_option_tree(
    root: object,
    reachable_ids: set[int],
    choice_ids: set[int],
    continuous_ids: set[int],
) -> set[int]:
    """Return the variables directly activated by one choice option."""
    activated_vars: set[int] = set()
    visited: set[int] = set()

    def walk(node: cg.Node):
        node_id = id(node)
        if node_id in visited or node_id not in reachable_ids:
            return
        visited.add(node_id)

        if node_id in choice_ids:
            activated_vars.add(node_id)
            return

        if node_id in continuous_ids:
            activated_vars.add(node_id)

        for child in _iter_node_children(node):
            walk(child)

    for child in PyTree(root).children:
        if isinstance(child, cg.Node):
            walk(child)

    return activated_vars


def build_model_from_compute_graph(graph: cg.ComputeGraph) -> dict:
    """Extract the compact branch-first model from a traced compute graph."""
    reachable_nodes: dict[int, cg.Node] = {}
    for subgraph in cg.traverse_nested_graphs(graph):
        for node in cg.traverse_depth_first(subgraph):
            reachable_nodes[id(node)] = node

    reachable_ids = set(reachable_nodes)
    choice_ids = {
        node_id for node_id, node in reachable_nodes.items() if _is_choice_node(node)
    }
    continuous_ids = {
        node_id
        for node_id, node in reachable_nodes.items()
        if _is_continuous_node(node)
    }
    all_var_ids = choice_ids | continuous_ids

    choice_probs: dict[int, list[float]] = {}
    choice_branches: dict[int, list[int]] = {}
    branch_vars: dict[int, list[int]] = {}
    var_kind = {var_id: "choice" for var_id in choice_ids}
    var_kind.update({var_id: "continuous" for var_id in continuous_ids})

    activated_by_branch: set[int] = set()
    next_branch_id = 0

    for choice_id in sorted(choice_ids):
        node = reachable_nodes[choice_id]
        assert isinstance(node, cg.FunctionCallNode)

        raw_options = list(node.kwargs.get("choice_options", []))
        probabilities = _normalize_choice_probabilities(raw_options)
        branch_ids: list[int] = []

        for option_result, _weight in raw_options:
            activated_vars = _scan_option_tree(
                root=option_result,
                reachable_ids=reachable_ids,
                choice_ids=choice_ids,
                continuous_ids=continuous_ids,
            )
            activated_vars.discard(choice_id)
            activated_by_branch.update(activated_vars)

            branch_id = next_branch_id
            next_branch_id += 1
            branch_ids.append(branch_id)
            branch_vars[branch_id] = sorted(activated_vars)

        choice_probs[choice_id] = probabilities
        choice_branches[choice_id] = branch_ids

    root_vars = sorted(all_var_ids - activated_by_branch)

    return {
        "root_vars": root_vars,
        "var_kind": var_kind,
        "choice_probs": choice_probs,
        "choice_branches": choice_branches,
        "branch_vars": branch_vars,
        "n_discrete_choices": len(choice_ids),
        "n_continuous_params": len(continuous_ids),
    }


def _topological_var_order(model: dict) -> list[int]:
    sorter = TopologicalSorter()

    for var_id in model["var_kind"]:
        sorter.add(var_id)

    for choice_id, branch_ids in model["choice_branches"].items():
        for branch_id in branch_ids:
            for child_var_id in model["branch_vars"][branch_id]:
                sorter.add(child_var_id, choice_id)

    return list(sorter.static_order())


def build_indexed_model(model: dict) -> dict:
    """Reindex variables into topological order and convert branches to masks."""
    var_order = _topological_var_order(model)
    var_index = {var_id: index for index, var_id in enumerate(var_order)}

    root_mask = 0
    for var_id in model["root_vars"]:
        root_mask |= 1 << var_index[var_id]

    indexed_var_kind = [model["var_kind"][var_id] for var_id in var_order]
    indexed_choice_probs: dict[int, list[float]] = {}
    indexed_choice_branches: dict[int, list[int]] = {}
    indexed_branch_vars: dict[int, int] = {}

    for branch_id, child_var_ids in model["branch_vars"].items():
        mask = 0
        for child_var_id in child_var_ids:
            mask |= 1 << var_index[child_var_id]
        indexed_branch_vars[branch_id] = mask

    for choice_id, probabilities in model["choice_probs"].items():
        indexed_choice_id = var_index[choice_id]
        indexed_choice_probs[indexed_choice_id] = list(probabilities)
        indexed_choice_branches[indexed_choice_id] = list(
            model["choice_branches"][choice_id]
        )

    return {
        "root_mask": root_mask,
        "var_kind": indexed_var_kind,
        "choice_probs": indexed_choice_probs,
        "choice_branches": indexed_choice_branches,
        "branch_vars": indexed_branch_vars,
        "n_discrete_choices": model["n_discrete_choices"],
        "n_continuous_params": model["n_continuous_params"],
    }


def compute_stats(
    graph: cg.ComputeGraph,
    method: str = "dp",
    continuous_bits: float = 3.0,
    trace_level: TraceLevel | None = None,
) -> dict:
    """Extract the compact model and run either the tree or DP solver."""
    model = build_model_from_compute_graph(graph)
    indexed_model = build_indexed_model(model)

    if method == "dp":
        stats = solve_dp(indexed_model)
    elif method == "tree":
        stats = solve_tree(indexed_model)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'dp' or 'tree'")

    has_discrete = trace_level is None or trace_level >= TraceLevel.RANDOM_CONTROL
    if not has_discrete:
        logger.warning(
            "trace_level=%s is below RANDOM_CONTROL; "
            "discrete choice stats are not meaningful and will be omitted",
            trace_level.name,
        )

    if has_discrete:
        stats["discrete_choices"] = model["n_discrete_choices"]
        stats["cyclomatic_complexity"] = cyclomatic_complexity(model["choice_probs"])

    stats["continuous_params"] = model["n_continuous_params"]
    stats["continuous_bits"] = float(continuous_bits)
    stats["entropy_with_continuous_bits"] = (
        stats.get("entropy_discrete_bits", 0.0)
        + stats["continuous_bits"] * stats["continuous_count_mean"]
    )
    return stats
