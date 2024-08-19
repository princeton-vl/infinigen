# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from infinigen.core.constraints import constraint_language as cl
from infinigen_examples.constraints import home as ex


def test_residential():
    cons = ex.home_furniture_constraints()

    assert isinstance(cons, cl.Node)
    assert isinstance(repr(cons), str)


def test_operators_simple():
    val = cl.constant(value=1)
    assert hasattr(val, "__add__")

    comp = cl.constant(1) + cl.constant(2)
    assert isinstance(comp, cl.Expression)
    val = comp()
    assert val == 3, val

    comp = cl.constant(1) < cl.constant(2)
    assert isinstance(comp, cl.Expression)
    assert comp() is True


def test_operators_cast():
    comp = cl.constant(1) + 2
    assert isinstance(comp, cl.ScalarOperatorExpression)
    assert comp() == 3


def test_associative_construction():
    comp = cl.constant(1) + cl.constant(2) + cl.constant(3)
    assert len(list(comp.traverse())) == 4  # 1 for additions, 3 for constants
