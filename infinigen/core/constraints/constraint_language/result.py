# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


from .expression import BoolExpression, ScalarExpression, nodedataclass
from .types import Node


@nodedataclass()
class Problem(Node):
    constraints: dict[str, BoolExpression]
    score_terms: dict[str, ScalarExpression]

    def __post_init__(self):
        if isinstance(self.constraints, list):
            self.constraints = {i: c for i, c in enumerate(self.constraints)}
        if isinstance(self.score_terms, list):
            self.score_terms = {i: s for i, s in enumerate(self.score_terms)}

    def children(self):
        for i, v in enumerate(self.constraints.values()):
            yield f"constraints[{i}]", v
        for i, v in enumerate(self.score_terms.values()):
            yield f"score_terms[{i}]", v
