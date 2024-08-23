# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import functools
import operator
import typing

import numpy as np

from .types import Node, nodedataclass

OPERATOR_ASSOCIATIVE = [operator.add, operator.mul, operator.and_, max, min]


@nodedataclass()
class Expression(Node):
    @classmethod
    def register_postfix_func(cls, expr_cls):
        @functools.wraps(expr_cls)
        def postfix_instantiator(self, *args, **kwargs):
            return expr_cls(self, *args, **kwargs)

        setattr(cls, expr_cls.__name__, postfix_instantiator)
        return expr_cls


@nodedataclass()
class ArithmethicExpression(Expression):
    pass


@nodedataclass()
class ScalarExpression(ArithmethicExpression):
    def minimize(self, *, weight: float):
        return self * constant(weight)

    def maximize(self, *, weight: float):
        return self * constant(-weight)

    def multiply(self, other):
        return ScalarOperatorExpression(operator.mul, [self, other])

    __mul__ = multiply

    def abs(self):
        return ScalarOperatorExpression(operator.abs, [self])

    __abs__ = abs

    def add(self, other):
        return ScalarOperatorExpression(operator.add, [self, other])

    __add__ = add

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def sub(self, other):
        return ScalarOperatorExpression(operator.sub, [self, other])

    __sub__ = sub

    def div(self, other):
        return ScalarOperatorExpression(operator.truediv, [self, other])

    __truediv__ = div

    def safediv(self, other):
        def safediv_impl(a, b):
            if b == 0:
                return 0 if a == 0 else 1
            return a / b

        return ScalarOperatorExpression(safediv_impl, [self, other])

    def pow(self, other):
        return ScalarOperatorExpression(operator.pow, [self, other])

    __pow__ = pow

    def equals(self, other):
        return BoolOperatorExpression(operator.eq, [self, other])

    __eq__ = equals

    def __ge__(self, other):
        return BoolOperatorExpression(operator.ge, [self, other])

    def __gt__(self, other):
        return BoolOperatorExpression(operator.gt, [self, other])

    def __le__(self, other):
        return BoolOperatorExpression(operator.le, [self, other])

    def __lt__(self, other):
        return BoolOperatorExpression(operator.lt, [self, other])

    def __ne__(self, other):
        return BoolOperatorExpression(operator.ne, [self, other])

    def __neg__(self):
        return self * constant(-1)

    def clamp_min(self, other):
        return max_expr(self, other)

    def clamp_max(self, other):
        return min_expr(self, other)

    def clip(self, a_min=-np.inf, a_max=np.inf):
        return ScalarOperatorExpression(
            (lambda x, y, z: min(max(x, y), z)), [self, a_min, a_max]
        )

    def log(self):
        return ScalarOperatorExpression(np.log, [self])


def max_expr(*args):
    return ScalarOperatorExpression(max, args)


def min_expr(*args):
    return ScalarOperatorExpression(min, args)


@nodedataclass()
class BoolExpression(ArithmethicExpression):
    def __mul__(self, other):
        return BoolOperatorExpression(operator.and_, [self, other])

    def __add__(self, other):
        return BoolOperatorExpression(operator.or_, [self, other])

    def __invert__(self):
        return BoolOperatorExpression(operator.not_, [self])


@nodedataclass()
class constant(ScalarExpression):
    value: int

    def __post_init__(self):
        assert isinstance(self.value, (bool | float | int))

    def __call__(self):
        return self.value


def _preprocess_operands(operands):
    def cast_to_node(x):
        match x:
            case Node():
                return x
            case x if isinstance(x, (bool | float | int)):
                return constant(x)
            case _:
                raise ValueError(f"Unsupported operand type {type(x)=} {x=}")

    return [cast_to_node(x) for x in operands]


def _collapse_associative(self, operands):
    if self.func not in OPERATOR_ASSOCIATIVE:
        return operands

    new_operands = []
    for op in operands:
        if isinstance(op, self.__class__) and op.func == self.func:
            new_operands.extend(op.operands)
        else:
            new_operands.append(op)
    return new_operands


@nodedataclass()
class BoolOperatorExpression(BoolExpression):
    func: typing.Callable
    operands: list[Expression]

    def __post_init__(self):
        self.operands = _preprocess_operands(self.operands)
        self.operands = _collapse_associative(self, self.operands)

    def children(self):
        for i, v in enumerate(self.operands):
            yield f"operands[{i}]", v

    def __call__(self) -> typing.Any:
        return self.func(*[x() for x in self.operands])


@nodedataclass()
class ScalarOperatorExpression(ScalarExpression):
    func: typing.Callable
    operands: list[Expression]

    def __post_init__(self):
        self.operands = _preprocess_operands(self.operands)
        self.operands = _collapse_associative(self, self.operands)
        assert self.func not in [operator.and_, operator.or_]

    def children(self):
        for i, v in enumerate(self.operands):
            yield f"operands[{i}]", v

    def __call__(self) -> typing.Any:
        return self.func(*[x() for x in self.operands])


@ScalarExpression.register_postfix_func
@nodedataclass()
class hinge(ScalarExpression):
    val: ScalarExpression
    low: float
    high: float


@Expression.register_postfix_func
@nodedataclass()
class debugprint(ScalarExpression, BoolExpression):
    val: Expression
    msg: str
