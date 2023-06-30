from dataclasses import dataclass, field
import itertools
import typing
import pdb
import copy

import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

from assets.creatures.util.tree import Tree
from assets.creatures.creature_util import interp_dict

from util.math import lerp

@dataclass
class IKParams:
    name: str
    chain_parts: int = 1 # how many parts up the hierarchy can this IK affect
    chain_length: int = None
    rotation_weight: float = 0
    target_size: float = 0.2

@dataclass
class Joint:
    pose: np.array = None

    def __post_init__(self):
        assert self.rest.shape == (3,), self.rest

        if self.pose is not None:
            self.pose = np.array(self.pose, dtype=np.float)
            assert self.pose.shape == (3,), self.pose
        if self.bounds is not None:
            self.bounds = np.array(self.bounds, dtype=np.float)

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
        self.coord = np.array(self.coord, dtype=np.float)

@dataclass
class CreatureNode:
    att: Attachment

@dataclass
class CreatureGenome:
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

    return res


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



    return att


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

def interp_genome(a: CreatureGenome, b: CreatureGenome, t: float) -> CreatureGenome:
    assert 0 <= t and t <= 1

    if t == 0:
        return a
    elif t == 1:
        return b
    
    #postprocess = interp_dict(a.postprocess_params, b.postprocess_params, t, recurse=True, keys='switch')
    #TODO a.postprocess_params
    postprocess = a.postprocess_params

                          postprocess_params=postprocess)


# Syntactic sugar to make defining trees of part params less verbose
################

def part(fac):
    return Tree(CreatureNode(fac, None))

    assert child.item.att is None
    parent.children.append(child)
