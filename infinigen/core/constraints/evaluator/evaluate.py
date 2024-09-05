# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
import operator
from dataclasses import dataclass

import pandas as pd

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator import eval_memo, node_impl
from infinigen.core.constraints.example_solver.state_def import State

logger = logging.getLogger(__name__)

SPECIAL_CASE_NODES = [
    cl.ForAll,
    cl.SumOver,
    cl.MeanOver,
    cl.item,
    cl.Problem,
    cl.scene,
    cl.debugprint,
]

gather_funcs = {
    cl.ForAll: all,
    cl.SumOver: sum,
    cl.MeanOver: lambda vs: (sum(vs) / len(vs)) if len(vs) else 0,
}


def _compute_node_val(node: cl.Node, state: State, memo: dict):
    match node:
        case cl.scene():
            return set(k for k, v in state.objs.items() if v.active)
        case (
            cl.ForAll(objs, var, pred)
            | cl.SumOver(objs, var, pred)
            | cl.MeanOver(objs, var, pred)
        ):
            assert isinstance(var, str)

            loop_over_objs = evaluate_node(objs, state, memo)

            results = []
            for o in loop_over_objs:
                memo_sub = copy.copy(memo)
                memo_sub[var] = {o}
                results.append(evaluate_node(pred, state, memo=memo_sub))

            # slogger.debug(f"{node.__class__.__name__} had {len(results)=}")

            return gather_funcs[node.__class__](results)
        case cl.item():
            raise ValueError(
                f"_compute_node_val encountered undefined variable {node}. {memo.keys()}"
            )
        case cl.Node() if node.__class__ in node_impl.node_impls:
            impl_func = node_impl.node_impls.get(node.__class__)
            child_vals = {
                name: evaluate_node(c, state, memo) for name, c in node.children()
            }
            kwargs = {}
            if hasattr(node, "others_tags"):
                kwargs["others_tags"] = getattr(node, "others_tags")
            return impl_func(node, state, child_vals, **kwargs)
        case cl.Problem():
            raise TypeError(
                f"evaluate_node is invalid for {node}, please use evaluate_problem"
            )
        case cl.debugprint(val, msg):
            res = evaluate_node(val, state, memo)
            var_assignments = [
                v
                for k, v in memo.items()
                if isinstance(k, str) and k.startswith("var_")
            ]
            print(f"cl.debugprint {msg}: {res} {var_assignments}")
            return res
        case _:
            raise NotImplementedError(
                f"Couldnt compute value for {type(node)}, please add it to "
                f"{node_impl.node_impls.keys()=} or add a specialcase"
            )


def relevant(node: cl.Node, filter: r.Domain | None) -> bool:
    if filter is None:
        # TODO filter can be None in room graph
        # raise ValueError()
        return True

    if not isinstance(node, cl.Node):
        raise TypeError(f"{node=}")

    match node:
        case cl.ObjectSetExpression():
            d = r.constraint_domain(node, finalize_variables=True)
            if not r.domain_finalized(d):
                raise RuntimeError(f"{relevant.__name__} encountered unfinalized {d=}")
            res = d.intersects(filter, require_satisfies_right=True)
            logger.debug(f"{relevant.__name__} got {res=} for {d=}\n {filter=}")
            return res
        case _:
            return any(relevant(c, filter) for _, c in node.children())


def _viol_count_binop_integer(
    node: cl.BoolOperatorExpression, lhs: int, rhs: int
) -> int:
    match node.func:
        case operator.ge:  # lhs >= rhs
            err = rhs - lhs
        case operator.le:  # lhs <= rhs
            err = lhs - rhs
        case operator.gt:  # lhs > rhs
            err = rhs - lhs + 1
        case operator.lt:  # lhs < rhs
            err = lhs - rhs + 1
        case _:
            raise ValueError(f"Unhandled {node.func=}")

    return max(err, 0)


def _viol_count_binop(node: cl.BoolOperatorExpression, lhs, rhs) -> float:
    if isinstance(lhs, int) and isinstance(rhs, int):
        return _viol_count_binop_integer(node, lhs, rhs)
    else:
        satisfied = node.func(lhs, rhs)
        return 1 if not satisfied else 0


def viol_count(node: cl.Node, state: State, memo: dict, filter: r.Domain = None):
    match node:
        case cl.BoolOperatorExpression(operator.and_, cons) | cl.Problem(cons):
            res = sum(viol_count(o, state, memo, filter) for o in cons)
        case cl.in_range(val, low, high):
            val_res = evaluate_node(val, state, memo)

            if val_res < low:
                res = low - val_res
            elif val_res > high:
                res = val_res - high
            else:
                res = 0

            if not relevant(val, filter):
                res = 0

        case cl.BoolOperatorExpression(operator.eq, [lhs, rhs]):
            res = abs(evaluate_node(lhs, state, memo) - evaluate_node(rhs, state, memo))
            if not relevant(lhs, filter) and not relevant(rhs, filter):
                res = 0
        case cl.ForAll(objs, var, pred):
            assert isinstance(var, str)
            viol = 0
            for o in evaluate_node(objs, state, memo):
                memo_sub = copy.copy(memo)
                memo_sub[var] = {o}
                viol += viol_count(pred, state, memo_sub, filter)
            res = viol
        case cl.BoolOperatorExpression(
            operator.ge | operator.le | operator.gt | operator.lt, [lhs, rhs]
        ):
            either_relevant = relevant(lhs, filter) or relevant(rhs, filter)
            if either_relevant:
                l_res = evaluate_node(lhs, state, memo)
                r_res = evaluate_node(rhs, state, memo)
                res = _viol_count_binop(node, l_res, r_res)
            else:
                res = 0
        case cl.constant(val) if isinstance(val, bool):
            res = 0 if val else 1
        case cl.BoolOperatorExpression(operator.or_, [lhs, rhs]):
            res = min(viol_count(rhs, state, memo), viol_count(lhs, state, memo))
        case cl.BoolOperatorExpression(operator.not_, [lhs]):
            lhs_res = evaluate_node(lhs, state, memo)
            res = 1 if lhs_res is True else 0
        case cl.Node():
            return evaluate_node(node, state, memo)
        case _:
            raise NotImplementedError(
                f"{node.__class__.__name__}(...) is not supported for hard constraints. Please use an alternative. Full node was {node}"
            )

    return res


@dataclass
class ConstraintsViolated:
    constraints: list[cl.Node]

    def __bool__(self):
        return False


def evaluate_node(node: cl.Node, state: State, memo=None):
    k = eval_memo.memo_key(node)

    if memo is None:
        memo = {}
    elif k in memo:
        return memo[k]
    val = _compute_node_val(node, state, memo)

    memo[k] = val
    # logger.debug("Evaluated %s to %s", node.__class__, val)

    return val


@dataclass
class EvalResult:
    loss_vals: dict[str, float]
    violations: dict[str, bool]

    def loss(self):
        return sum(v for v in self.loss_vals.values())

    def viol_count(self):
        return sum(x for x in self.violations.values())

    def to_df(self) -> pd.DataFrame:
        keys = set(self.loss_vals.keys()).union(self.violations.keys())
        return pd.DataFrame.from_dict(
            {
                k: dict(loss=self.loss_vals.get(k), viol_count=self.violations.get(k))
                for k in keys
            }
        )

    def __iter__(self):
        yield self.loss()
        yield self.viol_count()


def evaluate_problem(
    problem: cl.Problem,
    state: State,
    filter: r.Domain = None,
    memo=None,
    enable_loss=True,
    enable_violated=True,
):
    logger.debug(
        f"Evaluating problem {len(problem.constraints)=} {len(problem.score_terms)=}"
    )

    if memo is None:
        memo = {}

    scores = {}
    if enable_loss:
        for name, score_node in problem.score_terms.items():
            logger.debug(f"Evaluating score for {name=}")
            scores[name] = evaluate_node(score_node, state, memo)
            logger.debug(f"Evaluator got score {scores[name]} for {name=}")

    violated = {}
    if enable_violated:
        for name, node in problem.constraints.items():
            logger.debug(f"Evaluating constraint {name=}")
            violated[name] = viol_count(node, state, memo, filter=filter)

            if violated[name]:
                logger.debug(f"Evaluator found {violated[name]} violations for {name=}")

    return EvalResult(loss_vals=scores, violations=violated)
