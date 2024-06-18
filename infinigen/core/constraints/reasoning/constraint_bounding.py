# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: 
# - Alexander Raistrick: primary author
# - David Yan: bounding for inequalities / expressions

import operator
import typing
import copy
import dataclasses
from functools import partial
import logging

import numpy as np

from infinigen.core.constraints import constraint_language as cl

from .domain import Domain
from .constraint_domain import constraint_domain
from .domain_substitute import  domain_tag_substitute
    
from .constraint_constancy import is_constant

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Bound:

    domain: Domain = None
    low: int = None
    high: int = None

    _init_ops: typing.ClassVar = {
        operator.eq,
        operator.le,
        operator.ge,
        operator.lt,
        operator.gt,
    }

    @classmethod
    def from_comparison(cls, opfunc, lhs, rhs):

        lhs = lhs() if is_constant(lhs) else None
        rhs = rhs() if is_constant(rhs) else None

        if lhs is None == rhs is None:
            raise ValueError(f'Attempted to create bound with neither side constant {lhs=} {rhs=}')
        right_const = rhs is not None
        val = rhs if right_const else lhs
        
        match (opfunc, right_const):
            case operator.eq, _:
                return cls(low=val, high=val)
            case (operator.le, False) | (operator.ge, True):
                return cls(low=val)
            case (operator.le, True) | (operator.ge, False):
                return cls(high=val)
            case (operator.lt, True):
                return cls(high=val - 1)
            case (operator.gt, False):
                return cls(high=val - 1)
            case (operator.lt, False):
                return cls(low=val + 1)
            case (operator.gt, True):
                return cls(low=val + 1)
            case _:
                raise ValueError(f'Unhandled case {opfunc=}, {right_const=}')

    def map(self, func, lhs=None, rhs=None):
        
        if lhs is None == rhs is None:
            raise ValueError(f'Expected exactly one of {lhs=} {rhs=} to be provided')

        if lhs is not None:
            return Bound(
                low=func(lhs, self.low), 
                high=func(lhs, self.high)
            )
        else:
            return Bound(
                low=func(self.low, rhs), 
                high=func(self.high, rhs)
            )

int_inverse_op = {
    operator.add: operator.sub,
    operator.mul: operator.floordiv,
}
int_inverse_op.update({v: k for k, v in int_inverse_op.items()})

def _expression_map_bound_binop(
    node: cl.ScalarOperatorExpression, 
    bound: Bound
) -> list[Bound]:

    lhs, rhs = node.operands
    inv_func = int_inverse_op.get(node.func)
    if inv_func is None:
        return []
    
    consts = is_constant(lhs), is_constant(rhs)
    match consts:
        case (False, False):
            return []
        case (True, False):
            return expression_map_bound(rhs, bound.map(inv_func, lhs=lhs()))
        case (False, True):
            return expression_map_bound(lhs, bound.map(inv_func, rhs=rhs()))
        case (True, True): # both const, nothing to bound
            return []
        case _:
            raise ValueError("Impossible")

        case cl.ScalarOperatorExpression(f, (lhs, rhs)) if f in int_inverse_op.keys() or f in int_inverse_op:
def expression_map_bound(node: cl.Node, bound: Bound) -> list[Bound]:

    match node:
        case cl.ScalarOperatorExpression(f, (lhs, rhs)) if f in int_inverse_op.keys():
            return _expression_map_bound_binop(node, bound)
        case cl.count(objs):
            return expression_map_bound(objs, bound)
        case cl.ObjectSetExpression() as objs:
            bound = Bound(
                domain=constraint_domain(objs),
                low=bound.low,
                high=bound.high,
            )
            return [bound]
        case _:
            # distance & other hard constraints do not produce quantity-bounds
            return []


def constraint_bounds(
    node: cl.Node,
    state=None
) -> list[Bound]:

    recurse = partial(constraint_bounds, state=state)

    match node:
        case cl.Problem(cons):
            return sum((recurse(c) for c in cons.values()), [])
        case cl.BoolOperatorExpression(operator.and_, cons):
            return sum((recurse(c) for c in cons), [])
        case cl.in_range(val, low, high):
            low = update_var(low, state)
            high = update_var(high, state)
            bound = Bound(low=low, high=high)
            return expression_map_bound(val, bound)
        case cl.BoolOperatorExpression(f, (lhs, rhs)) if f in Bound._init_ops:
            lhs, rhs = update_var(lhs, state), update_var(rhs, state)

            if not is_constant(lhs) and not is_constant(rhs):
                logger.debug(f'Encountered {cl.BoolOperatorExpression.__name__} {f} with non-constant lhs and rhs. Producing no bound.')
                return []

            bound = Bound.from_comparison(node.func, lhs, rhs)
            expr = rhs if is_constant(lhs) else lhs
            return expression_map_bound(expr, bound)
        case cl.ForAll(objs, varname, pred):
            o_domain = constraint_domain(objs)
            bounds = recurse(pred)
            for b in bounds:
                # TODO INCORRECT. Doesnt force EVERY object in o_domain to satify the bound
                b.domain = domain_tag_substitute(b.domain, t.Variable(varname), o_domain)
            return bounds
        case unmatched:
            assert isinstance(unmatched, cl.Expression), unmatched
            return []
    