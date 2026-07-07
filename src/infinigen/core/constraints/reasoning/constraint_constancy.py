# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


from infinigen.core.constraints import constraint_language as cl


def is_constant(node: cl.Node):
    match node:
        case cl.constant():
            return True
        case cl.BoolOperatorExpression(_, vs) | cl.ScalarOperatorExpression(_, vs):
            return all(is_constant(x) for x in vs)
        case _:
            return False
