import numpy as np

from infinigen.core.constraints.constraint_language import ScalarExpression, ObjectSetExpression, BoolExpression
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.constraints.constraint_language.types import nodedataclass
from infinigen.core.tags import Tag


@nodedataclass()
class shortest_path(ScalarExpression):
    objs: ObjectSetExpression
    reduce_fn: str = 'sum'


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class direct_access(ScalarExpression):
    objs: ObjectSetExpression
    angle: float = np.pi / 3


@nodedataclass()
class typical_area(ScalarExpression):
    objs: ObjectSetExpression
    types: dict[Tag, float]
    reduce_fn: str = 'sum'


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class aspect_ratio(ScalarExpression):
    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class convexity(ScalarExpression):
    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class n_verts(ScalarExpression):
    objs: ObjectSetExpression


@nodedataclass()
class exterior_corner(ScalarExpression):
    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class grid_line_count(ScalarExpression):
    objs: ObjectSetExpression
    constants: RoomConstants
    direction: str = None


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class narrowness(ScalarExpression):
    objs: ObjectSetExpression
    constants: RoomConstants
    thresh: float


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class intersection(ScalarExpression):
    objs: ObjectSetExpression
    objs_: ObjectSetExpression


@nodedataclass()
class graph_coherent(BoolExpression):
    constants: RoomConstants


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class same_level(ObjectSetExpression):
    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class area(ScalarExpression):
    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class shared_length(ScalarExpression):
    objs: ObjectSetExpression
    objs_: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class shared_n_verts(ScalarExpression):
    objs: ObjectSetExpression
    objs_: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class length(ScalarExpression):
    objs: ObjectSetExpression


@ScalarExpression.register_postfix_func
@nodedataclass()
class rand(ScalarExpression):
    count: ScalarExpression
    type: str
    args: float | list[float]
