# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from dataclasses import dataclass, field
import itertools
import typing
import pdb
import copy

import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

from infinigen.assets.creatures.util.tree import Tree
from infinigen.assets.creatures.util.creature_util import interp_dict

from infinigen.core.util.math import lerp


@dataclass
class IKParams:
    name: str
    chain_parts: int = 1 # how many parts up the hierarchy can this IK affect
    chain_length: int = None
    rotation_weight: float = 0
    mode: str = 'iksolve'  # iksolve, pin
    target_size: float = 0.2


@dataclass
class Joint:
    rest: np.array  # (3) float, degrees
    pose: np.array = None
    bounds: np.array = None  # (2x3) float, degrees
    stiffness: np.array = None  # (3) float
    stretch: float = 0

    def __post_init__(self):
        self.rest = np.array(self.rest, dtype=float)
        assert self.rest.shape == (3,), self.rest

        if self.pose is not None:
            self.pose = np.array(self.pose, dtype=float)
            assert self.pose.shape == (3,), self.pose

        if self.bounds is not None:
            self.bounds = np.array(self.bounds, dtype=float)


@dataclass
class Attachment:
    coord: np.array
    joint: Joint = None
    bridge: str = None
    side: int = 1
    rotation_basis: str = 'global'
    bridge_rad: float = 0.0
    smooth_rad: float = 0.0

    def __post_init__(self):
        self.coord = np.array(self.coord, dtype=float)


@dataclass
class CreatureNode:
    part_factory: typing.Any  # PartFactory
    att: Attachment


@dataclass
class CreatureGenome:
    parts: Tree  # of CreatureNode
    postprocess_params: dict[dict] = field(default_factory=dict)


def compute_child_matching(a: list[Tree], b: list[Tree]):
    def match_cost(a: Tree, b: Tree):
        diff = b.item.att.coord - a.item.att.coord
        diff[1] = min(diff[1], 1 - diff[1])
        return np.linalg.norm(diff)

    cost_matrix = np.array([match_cost(ac, bc) for ac, bc in itertools.product(a, b)])
    cost_matrix = cost_matrix.reshape(len(a), len(b))
    cost_matrix = csr_matrix(cost_matrix)

    perm = maximum_bipartite_matching(-cost_matrix, perm_type='column')

    res = []
    for ai, bi in enumerate(perm):
        res.append((a[ai] if ai != -1 else None, b[bi] if bi != -1 else None))

    return res


def lerp_any(a, b, t):
    def cast(x):
        if isinstance(x, (tuple, list)):
            return np.array(x)
        return x

    res = lerp(cast(a), cast(b), t)

    if isinstance(a, int) or isinstance(b, int):
        res = int(res)

    return res


def interp_attachment(a: Attachment, b: Attachment, t: float):
    if a is None or b is None:
        return None

    s = b if t > 0.5 else a

    joint = Joint(rest=lerp(a.joint.rest, b.joint.rest, t), bounds=s.joint.bounds)

    att = Attachment(coord=lerp(a.coord, b.coord, t), joint=joint, bridge=s.bridge, side=s.side)

    return att


def interp_creature_node(a: CreatureNode, b: CreatureNode, t):
    s = b if t > 0.5 else a  # which of a,b should we take non-interpolatable things from
    fac = copy.copy(s.part_factory)
    fac.params = interp_dict(a.part_factory.params, b.part_factory.params, t, keys='switch', lerp=lerp_any)

    #att = interp_attachment(a.att, b.att, t)
    att = a.att # TODO: Enable attachment interp later, debug symmetry

    return CreatureNode(part_factory=fac, att=att)


def interp_part_tree(a: Tree, b: Tree, t: float):
    new_children = []
    for ac, bc in compute_child_matching(a.children, b.children):
        if ac is None:
            if t < 0.5:
                continue
            else:
                new_children.append(bc)
        elif bc is None:
            if t < 0.5:
                new_children.append(ac)
            else:
                continue
        else:
            new_children.append(interp_part_tree(ac, bc, t))
    return Tree(item=interp_creature_node(a.item, b.item, t), children=new_children)


def interp_genome(a: CreatureGenome, b: CreatureGenome, t: float) -> CreatureGenome:
    assert 0 <= t and t <= 1

    if t == 0:
        return a
    elif t == 1:
        return b
    
    #postprocess = interp_dict(a.postprocess_params, b.postprocess_params, t, recurse=True, keys='switch')
    #TODO a.postprocess_params
    postprocess = a.postprocess_params

    return CreatureGenome(parts=interp_part_tree(a.parts, b.parts, t),
                          postprocess_params=postprocess)


################
# Syntactic sugar to make defining trees of part params less verbose
################

def part(fac):
    return Tree(CreatureNode(fac, None))


def attach(child: Tree, parent: Tree, coord=None, joint=None, bridge=None, side=1, rotation_basis='global',
           bridge_rad=.0, smooth_rad=.0):
    assert child.item.att is None
    if coord is None:
        coord = np.array([0, 0, 0])
    if joint is None:
        joint = Joint((0, 0, 0))
    child.item.att = Attachment(coord, joint, bridge, side, rotation_basis, bridge_rad, smooth_rad)
    parent.children.append(child)
    return parent
