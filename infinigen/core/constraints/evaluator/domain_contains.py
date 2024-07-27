# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import state_def

logger = logging.getLogger(__name__)


def domain_contains(dom: r.Domain, state: state_def.State, obj: state_def.ObjectState):
    assert isinstance(dom, r.Domain), dom
    assert isinstance(obj, state_def.ObjectState), obj

    if not t.satisfies(obj.tags, dom.tags):
        # logger.debug(f"domain_contains failed, {obj} does not satisfy {obj.tags}")
        return False

    for rel, dom in dom.relations:
        if isinstance(rel, cl.NegatedRelation):
            if any(
                relstate.relation.intersects(rel.rel)
                and domain_contains(dom, state, state.objs[relstate.target_name])
                for relstate in obj.relations
            ):
                # logger.debug(f"domain_contains failed, {obj} satisfies negative {rel} {dom}")
                return False
        else:
            if not any(
                relstate.relation.intersects(rel)
                and domain_contains(dom, state, state.objs[relstate.target_name])
                for relstate in obj.relations
            ):
                # logger.debug(f"domain_contains failed, {obj} does not satisfy {rel} {dom}")
                return False

    return True


def objkeys_in_dom(dom: r.Domain, curr: state_def.State):
    return [
        k for k, o in curr.objs.items() if domain_contains(dom, curr, o) and o.active
    ]
