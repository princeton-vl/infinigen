# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from infinigen.core import tags as t


def test_implies():
    assert t.implies(set(), set())
    assert t.implies({t.Subpart.Wall}, {t.Subpart.Wall})
    assert t.implies({t.Subpart.Wall}, set())

    assert t.implies({t.Semantics.Room, t.Variable("room")}, {t.Semantics.Room})
