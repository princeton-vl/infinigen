# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from infinigen.core import tags as t


def test_tagset_operations():
    example = {t.Subpart.Side, -t.Subpart.Bottom}
    assert t.implies(example, example)
    assert not t.contradiction(example)

    superset_pos = {t.Subpart.Side}
    assert not t.implies(superset_pos, example)
    assert t.implies(example, superset_pos)

    superset_neg = {-t.Subpart.Bottom}
    assert t.implies(example, superset_neg)
    assert not t.implies(superset_neg, example)

    subset_pos = {t.Subpart.Side, t.Subpart.Front, -t.Subpart.Bottom}
    assert not t.implies(example, subset_pos)
    assert t.implies(subset_pos, example)

    subset_neg = [t.Subpart.Side, -t.Subpart.Bottom, -t.Subpart.Top]
    assert not t.implies(example, subset_neg)
    assert t.implies(subset_neg, example)

    intersect_pos = {t.Subpart.Side, t.Subpart.Front}
    assert not t.implies(example, intersect_pos)
    assert not t.implies(intersect_pos, example)
    assert not t.contradiction(example.union(intersect_pos))

    intersect_neg = {t.Subpart.Side, -t.Subpart.Back}
    assert not t.implies(example, intersect_neg)
    assert not t.implies(intersect_neg, example)
    assert not t.contradiction(example.union(intersect_neg))

    assert not t.implies(
        {t.Subpart.Top, -t.Subpart.Bottom}, {t.Subpart.Top, t.Subpart.Bottom}
    )
