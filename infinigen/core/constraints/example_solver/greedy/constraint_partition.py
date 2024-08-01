# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
import operator
import typing
from functools import partial

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r

logger = logging.getLogger(__name__)

OPS_COMMUTATIVE = {operator.add, operator.and_, operator.mul, operator.or_}

OPS_UNIT_VALUE = {
    operator.add: 0,
    operator.mul: 1,
    operator.pow: 0,
    operator.truediv: 0,
}


def _get_op_unit_value(node: cl.BoolExpression | cl.ScalarExpression):
    match node:
        case cl.BoolOperatorExpression(func, _):
            return True
        case cl.ScalarOperatorExpression(func, _) if func in OPS_UNIT_VALUE:
            return OPS_UNIT_VALUE[func]
        case _:
            raise ValueError(f"Found no unit value for  {node.__class__} {node.func}")


def _partition_dict(terms: dict[str, cl.Node], recurse: typing.Callable):
    new_terms = {}
    for k, v in terms.items():
        part, relevant = recurse(v)
        if not relevant:
            continue
        new_terms[k] = part
    return new_terms


def _update_item_nodes(
    node: cl.Node, from_varname: str, to_varname: str, to_objs: cl.ObjectSetExpression
):
    for child in node.traverse():
        if not isinstance(child, cl.item):
            continue
        if child.name != from_varname:
            continue
        child.name = to_varname
        child.member_of = to_objs

    return node


def _filter_gather_constraint(
    node: cl.ForAll | cl.SumOver | cl.MeanOver,
    recurse: typing.Callable,
    filter_dom: r.Domain,
    var_assignments: dict[t.Variable, r.Domain],
) -> tuple[cl.Node, bool]:
    objs, var, pred = node.objs, node.var, node.pred
    var = t.Variable(var)

    objs_part, objs_rel = recurse(objs)

    obj_dom = r.constraint_domain(objs)
    obj_dom = r.substitute_all(obj_dom, var_assignments)

    var_assignments = copy.deepcopy(var_assignments) or {}
    for varname, dom in var_assignments.items():
        assert isinstance(varname, t.Variable)
        var_assignments[varname] = r.domain_tag_substitute(dom, var, obj_dom)
    var_assignments[var] = obj_dom

    pred_part, pred_rel = recurse(pred, var_assignments=var_assignments)

    for pred_child in pred_part.traverse():
        if not isinstance(pred_child, r.FilterByDomain):
            continue
        subst_filter, matched = r.domain_tag_substitute(
            pred_child.filter, var, obj_dom, return_match=True
        )

        pred_child.filter = subst_filter

    res = copy.copy(node)
    res.objs = objs_part
    res.var = var
    res.pred = pred_part

    relevant = pred_rel
    return res, relevant


def _filter_object_set(
    node: cl.ObjectSetExpression,
    recurse: typing.Callable,
    filter_dom: r.Domain,
    var_assignments: dict[t.Variable, r.Domain],
) -> tuple[cl.Node, bool]:
    new_consnode = copy.deepcopy(node)

    dom = r.constraint_domain(node)
    dom_subst = r.substitute_all(dom, var_assignments)

    if not r.domain_finalized(dom_subst, check_anyrel=False, check_variable=True):
        raise ValueError(
            "Domain not finalized, unable to check against filter. "
            "Check for any undefined variables? should be impossible. "
            f"{dom=}."
        )

    relevant = dom_subst.intersects(filter_dom, require_satisfies_right=True)
    if (
        relevant
        and not dom_subst.satisfies(
            filter_dom
        )  # no need to filter something that is already strict enough
    ):
        finalized = r.domain_finalized(
            filter_dom, check_anyrel=False, check_variable=True
        )
        assert finalized, filter_dom
        new_consnode = r.FilterByDomain(new_consnode, filter_dom)

    return new_consnode, relevant


def _filter_operator(
    node: cl.BoolOperatorExpression | cl.ScalarOperatorExpression,
    recurse: typing.Callable,
    filter_dom: r.Domain,
    var_assignments: dict[t.Variable, r.Domain],
) -> tuple[cl.Node, bool]:
    operands, func = node.operands, node.func

    op_results = [recurse(o) for o in operands]
    relevant_ops = [node for node, rel in op_results if rel]

    match relevant_ops, func:
        case ([], _):
            return cl.constant(_get_op_unit_value(node)), False
        case ([op], f) if f in OPS_COMMUTATIVE:
            return op, True
        case (new_operands, f) if (
            len(new_operands) == len(operands) or f in OPS_COMMUTATIVE
        ):
            return node.__class__(f, new_operands), True
        case _:
            res = node.__class__(func, [o[0] for o in op_results])
            any_relevant = any(o[1] for o in op_results)
            return res, any_relevant


def _filter_node_cases(
    node: cl.Node,
    recurse: typing.Callable,
    filter_dom: r.Domain,
    var_assignments: dict[t.Variable, r.Domain],
) -> tuple[cl.Node, bool]:
    match node:
        case cl.Problem(cons, score_terms):
            prob = cl.Problem(
                _partition_dict(cons, recurse), _partition_dict(score_terms, recurse)
            )
            relevant = len(prob.constraints) > 0 or len(prob.score_terms) > 0
            return prob, relevant
        case cl.ForAll() | cl.SumOver() | cl.MeanOver():
            return _filter_gather_constraint(node, recurse, filter_dom, var_assignments)
        case cl.BoolOperatorExpression() | cl.ScalarOperatorExpression():
            return _filter_operator(node, recurse, filter_dom, var_assignments)
        case cl.ObjectSetExpression():
            return _filter_object_set(node, recurse, filter_dom, var_assignments)
        case _:
            result_relevant = False
            result_consnode = copy.deepcopy(node)

            for name, child in node.children():
                res, relevant = recurse(child)
                if not hasattr(node, name):
                    raise ValueError(
                        f"Node {node.__class__} has child with {name=} but no attribute {name} to set"
                    )
                setattr(result_consnode, name, res)
                result_relevant = result_relevant or relevant

            return result_consnode, result_relevant


def _check_partition_correctness(
    node: cl.ObjectSetExpression,
    filter_dom: r.Domain,
    var_assignments: dict[t.Variable, r.Domain],
):
    res_dom = r.constraint_domain(node)
    res_dom = r.substitute_all(res_dom, var_assignments)

    if not r.domain_finalized(res_dom, check_anyrel=False, check_variable=True):
        raise ValueError(
            f"While doing {_check_partition_correctness.__name__} for {node=} {filter_dom=}, "
            f"got {res_dom=} is not finalized, {var_assignments.keys()=}"
        )

    if not res_dom.satisfies(filter_dom):
        raise ValueError(f"{res_dom=} does not satisfy {filter_dom=}")


def filter_constraints(
    node: cl.Node,
    filter_dom: r.Domain,
    var_assignments: dict[str, r.Domain] = None,
    check_correctness=True,
) -> tuple[cl.Node, bool]:
    """Return a constraint graph representing the component of `node` that is relevant for
    to a particular greedy filter domain.

    Parameters
    ----------
    node : cl.Node
        The constraint program to partition
    filter_dom : Domain
        The domain which determines whether a constraint is relevant
    var_assignments : Domain
        Domains to substitute for any t.Variable(name: str) in the constraint program, typically used for recursive calls.

    Returns
    -------
    partitioned: cl.Node
        The partitioned constraint program
    relevant: bool
        Was any part of the constraint program relevant?

    """

    assert isinstance(node, cl.Node), node

    if var_assignments is None:
        var_assignments = {}

    recurse = partial(
        filter_constraints, filter_dom=filter_dom, var_assignments=var_assignments
    )

    logger.debug(
        f"{filter_constraints.__name__} for {node.__class__.__name__}, {var_assignments.keys()=}"
    )
    new_node, relevant = _filter_node_cases(node, recurse, filter_dom, var_assignments)

    if relevant and check_correctness and isinstance(new_node, cl.ObjectSetExpression):
        _check_partition_correctness(new_node, filter_dom, var_assignments)

    logger.debug(
        f"Partitioned {node.__class__.__name__} to {new_node.__class__.__name__}"
    )

    return new_node, relevant
