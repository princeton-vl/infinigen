# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from collections import defaultdict

from infinigen.core import tags as t

_factory_lookup: dict[type, set[t.Tag]] = None
_tag_lookup: dict[t.Tag, set[type]] = None


def initialize_from_dict(d):
    global _factory_lookup, _tag_lookup
    _factory_lookup = defaultdict(set)
    _tag_lookup = defaultdict(set)

    for tag, fac_list in d.items():
        _tag_lookup[tag] = set()
        for fac in fac_list:
            _factory_lookup[fac].add(tag)
            _tag_lookup[tag].add(fac)


def usages_of_factory(fac) -> set[t.Tag]:
    return _factory_lookup[fac].union({t.FromGenerator(fac)})


def factories_for_usage(tags: set[t.Tag]):
    if not isinstance(tags, set):
        tags = [tags]
    else:
        tags = list(tags)

    res = _tag_lookup[tags[0]]
    for tag in tags[1:]:
        res.intersection_update(_tag_lookup[tag])
    return res


def all_usage_tags():
    return _tag_lookup.keys()


def all_factories():
    return _factory_lookup.keys()


def has_usage(fac, tag):
    assert fac in _factory_lookup.keys(), fac
    assert tag in _tag_lookup.keys(), tag
    return tag in _factory_lookup[fac]
