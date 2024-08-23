# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick
import re
from dataclasses import field

from infinigen.core.util.math import int_hash

from .constants import RoomConstants
from .expression import BoolExpression, ScalarExpression, nodedataclass
from .types import Node


@nodedataclass()
class Problem(Node):
    constraints: dict[str, BoolExpression]
    score_terms: dict[str, ScalarExpression]
    constants: RoomConstants = field(default_factory=RoomConstants)

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

    def filter(self, name):
        constraints = {k: v for k, v in self.constraints.items() if re.match(name, k)}
        score_terms = {k: v for k, v in self.score_terms.items() if re.match(name, k)}
        return Problem(constraints, score_terms, self.constants)

    def __hash__(self):
        return int_hash(re.sub("[0-9]", "", re.sub(r"<[^)]*>", "", str(self))))
