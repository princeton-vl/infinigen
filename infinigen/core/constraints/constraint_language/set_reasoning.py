# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import typing
from dataclasses import dataclass, field

from infinigen.core.constraints import usage_lookup
from .relations import Relation, AnyRelation
from .expression import Expression, BoolExpression, ScalarExpression, nodedataclass

@nodedataclass()
class ObjectSetExpression(Expression):
    
    def __getitem__(self, key):
        return tagged(self, key)

@nodedataclass()
class scene(ObjectSetExpression):
    pass

@ObjectSetExpression.register_postfix_func
@nodedataclass()
class tagged(ObjectSetExpression):
    objs: ObjectSetExpression

    def __post_init__(self):
        self.tags = t.to_tag_set(self.tags, fac_context=usage_lookup._factory_lookup)

    
@ObjectSetExpression.register_postfix_func
def excludes(objs, tags):
    
    # syntactic helper - assume people wont construct obvious contradictions
    if isinstance(objs, tagged):
        tags = tags.difference(objs.tags)

    
@ObjectSetExpression.register_postfix_func
@nodedataclass()
class related_to(ObjectSetExpression):
    child: ObjectSetExpression
    parent: ObjectSetExpression
    relation: Relation = field(default_factory=AnyRelation)

    def __post_init__(self):
        if not isinstance(self.child, ObjectSetExpression):
            raise TypeError(f'related_to got {self.child=}, must be an ObjectSetExpression')
        if not isinstance(self.parent, ObjectSetExpression):
            raise TypeError(f'related_to got {self.parent=}, must be an ObjectSetExpression')
        if not isinstance(self.relation, Relation):
            raise TypeError(f'related_to got {self.relation=}, must be a Relation')

@ObjectSetExpression.register_postfix_func
@nodedataclass()
class count(ScalarExpression):
    objs: ObjectSetExpression

    def __post_init__(self):
        if not isinstance(self.objs, ObjectSetExpression):
            raise TypeError(f'count got {self.objs=}, must be an ObjectSetExpression')


@ScalarExpression.register_postfix_func
@nodedataclass()
class in_range(BoolExpression):
    val: ScalarExpression
    low: float
    high: float

    def __post_init__(self):
        if not isinstance(self.val, ScalarExpression):
            raise TypeError(f'in_range got {self.val=}, must be a ScalarExpression')
        if not isinstance(self.low, (int, float)):
            raise TypeError(f'in_range got {self.low=}, must be a number')
        if not isinstance(self.high, (int, float)):
            raise TypeError(f'in_range got {self.high=}, must be a number')

