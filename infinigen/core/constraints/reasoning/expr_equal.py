# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import dataclasses

from infinigen.core.constraints.constraint_language.types import Node


@dataclasses.dataclass
class FalseEqualityResult:
    n1: Node
    n2: Node
    reason: str

    def __repr__(self) -> str:
        # default dataclass repr is too long
        c1 = self.n1.__class__.__name__
        c2 = self.n2.__class__.__name__
        return f"{self.__class__.__name__}({c1}, {c2}, {repr(self.reason)})"

    def __bool__(self):
        return False


def expr_equal(n1: Node, n2: Node, name: str = None) -> bool | FalseEqualityResult:
    """An equality comparison operator for constraint Node expressions

    Using the default Node == Node is unsafe since Nodes override ==
    in order to return another expression
    """

    if not dataclasses.is_dataclass(n1) or not dataclasses.is_dataclass(n2):
        raise ValueError(
            f"expr_equal {name=} called with non-dataclass {n1.__class__=} {n2.__class__=}."
            " Expected all Node types to be dataclasses"
        )

    if name is None:
        name = n1.__class__.__name__

    if type(n1) is not type(n2):
        return FalseEqualityResult(
            n1,
            n2,
            f"Unequal types for {name}: {type(n1).__name__} != {type(n2).__name__}",
        )

    n1_child_keys = [k for k, _ in n1.children()]
    n2_child_keys = [k for k, _ in n1.children()]
    n1_children = [v for _, v in n1.children()]
    n2_children = [v for _, v in n1.children()]

    if n1_child_keys != n2_child_keys:
        return FalseEqualityResult(
            n1, n2, f"Unequal child keys for {name}: {n1_children}!={n2_children}"
        )

    for f in dataclasses.fields(n1):
        v1 = getattr(n1, f.name)
        v2 = getattr(n2, f.name)
        if isinstance(v1, Node):
            res = expr_equal(v1, v2, name=f"{name}.{f.name}")
            if not res:
                return res
        elif v1 != v2:
            return FalseEqualityResult(
                n1, n2, f"Unequal attr {repr(f.name)}, {v1} != {v2}"
            )

    return True
