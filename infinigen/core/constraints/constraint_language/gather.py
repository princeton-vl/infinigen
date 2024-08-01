# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import typing

from .expression import BoolExpression, ScalarExpression, nodedataclass
from .geometry import ObjectSetExpression


@nodedataclass()
class item(ObjectSetExpression):
    name: str
    member_of: ObjectSetExpression

    def __repr__(self):
        return f"item({self.name})"

    def children(self):
        # member_of is metadata, should not be treated as a child
        return []


@nodedataclass()
class ForAll(BoolExpression):
    objs: ObjectSetExpression
    var: str
    pred: BoolExpression


@ObjectSetExpression.register_postfix_func
def all(
    objs: ObjectSetExpression, pred: typing.Callable[[item], BoolExpression]
) -> BoolExpression:
    var = "var_all_" + str(id(pred))
    return ForAll(objs, var, pred(item(var, objs)))


@nodedataclass()
class SumOver(ScalarExpression):
    objs: ObjectSetExpression
    var: str
    pred: ScalarExpression


@ObjectSetExpression.register_postfix_func
def sum(objs: ObjectSetExpression, pred: typing.Callable[[item], ScalarExpression]):
    var = "var_sum_" + str(id(pred))
    return SumOver(objs, var, pred(item(var, objs)))


@nodedataclass()
class MeanOver(ScalarExpression):
    objs: ObjectSetExpression
    var: str
    pred: ScalarExpression


@ObjectSetExpression.register_postfix_func
def mean(objs: ObjectSetExpression, pred: typing.Callable[[item], ScalarExpression]):
    var = "var_mean_" + str(id(pred))
    return MeanOver(objs, var, pred(item(var, objs)))
