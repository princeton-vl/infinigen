import typing
from dataclasses import dataclass, field

import numpy as np

from .expression import Expression, BoolExpression, ScalarExpression, nodedataclass
from .set_reasoning import ObjectSetExpression
class center_stable_surface_dist(ScalarExpression):
    normal: np.array = field(default=np.array([1, 0, 0]))
    dist: float = 1.0
    def __post_init__(self):
        if isinstance(self.normal, (list, tuple)):
            self.normal = np.array(self.normal)
        assert isinstance(self.normal, np.ndarray)

@ObjectSetExpression.register_postfix_func
@nodedataclass()
class distance(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression
    others_tags: set = field(default_factory=set)

    def __post_init__(self):
        assert isinstance(self.objs, ObjectSetExpression)
        assert isinstance(self.others, ObjectSetExpression)
        assert isinstance(self.others_tags, set)

@nodedataclass()
class min_distance_internal(ScalarExpression):
    objs: ObjectSetExpression
    
@nodedataclass()
    objs: ObjectSetExpression
    others: ObjectSetExpression
@nodedataclass()
    objs: ObjectSetExpression
    others: ObjectSetExpression
    def __post_init__(self):
        if self.others_tags is None:
            self.others_tags = set()
        assert isinstance(self.others_tags, set), type(self.others_tags)

@nodedataclass()
    objs: ObjectSetExpression
    others: ObjectSetExpression
@nodedataclass()
    objs: ObjectSetExpression
    others: ObjectSetExpression
@nodedataclass()
    objs: ObjectSetExpression

@ObjectSetExpression.register_postfix_func
@nodedataclass()
class volume(ScalarExpression):
    objs: ObjectSetExpression
