# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei

from infinigen.core.constraints.constraint_language import (
    BoolExpression,
    ObjectSetExpression,
    ScalarExpression,
)
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.constraints.constraint_language.types import nodedataclass


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class access_angle(ScalarExpression):
    """Computes the angle between the vector from the root to the room and the vector from the room's neighbour to the room"""

    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class aspect_ratio(ScalarExpression):
    """Computes the aspect ratio of a room contour, always >1"""

    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class convexity(ScalarExpression):
    """Computes the ratio of the area of bounding box of the room contour to the room"""

    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class n_verts(ScalarExpression):
    """Computes the number of vertices of the room contour"""

    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class grid_line_count(ScalarExpression):
    """Computes the number of unique x/y grid lines objs occupies"""

    objs: ObjectSetExpression
    constants: RoomConstants
    direction: str = None


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class narrowness(ScalarExpression):
    """Computes the circumference difference after the room contour is eroded and buffered by thresh. Would be non-zero if the countour has a narrow end"""

    objs: ObjectSetExpression
    constants: RoomConstants
    thresh: float


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class intersection(ScalarExpression):
    """Computes the intersection area between two sets of room contours"""

    objs: ObjectSetExpression
    objs_: ObjectSetExpression


@nodedataclass()
class graph_coherent(BoolExpression):
    """Computes if the state is coherent with the room graph in terms of adjacency"""

    constants: RoomConstants


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class same_level(ObjectSetExpression):
    """Computes the set of room contours on the same level/floor"""

    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class area(ScalarExpression):
    """Computes the area of the room contour"""

    objs: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class shared_length(ScalarExpression):
    """Computes the shared length between two sets of room contours"""

    objs: ObjectSetExpression
    objs_: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class shared_n_verts(ScalarExpression):
    """Computes the number of shared vertices of two sets of room contours"""

    objs: ObjectSetExpression
    objs_: ObjectSetExpression


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class length(ScalarExpression):
    """Computes the circumference of a room contour"""

    objs: ObjectSetExpression


@ScalarExpression.register_postfix_func
@nodedataclass()
class rand(ScalarExpression):
    """
    Computes the NLL of an integer sample given the distribution type and arguments
    type can be of 'bool'('bern') / 'categorical'('cat')
    """

    count: ScalarExpression
    type: str
    args: float | list[float]
