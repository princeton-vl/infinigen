# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from dataclasses import dataclass, field
import typing
import itertools

@dataclass
class Tree:
    item: typing.Any
    children: list = field(default_factory=list)

def iter_nodes(t: Tree, postorder=False):
    if not postorder:
        yield t 
    for c in t.children:
        yield from iter_nodes(c, postorder=postorder)
    if postorder:
        yield t

def iter_items(t: Tree, postorder=False):
    for n in iter_nodes(t, postorder=postorder):
        yield n.item

Tree.__iter__ = iter_items

inorder = iter_items

def iter_parent_child(t: Tree, parent=None, postorder=False):
    if not postorder:
        yield None if parent is None else parent.item, t.item
    for c in t.children:
        yield from iter_parent_child(c, parent=t, postorder=postorder)
    if postorder:
        yield None if parent is None else parent.item, t.item
        
def fold(t: Tree, func):
    child_res = [fold(func, node=child) for child in t.children]
    return func(t.item, child_res)

def map(t: Tree, func) -> Tree:
    return Tree(item=func(t.item), children=[map(c, func) for c in t.children])

def map_parent_child(t, func, parent_node=None, parent_res=None, **opts) -> Tree:
    arg = (t, parent_node) if opts.get('include_parent_node', False) else t
    res = func(arg, parent_res)
    return Tree(res, children=[map_parent_child(c, func, parent_node=t, parent_res=res, **opts) for c in t.children])

def tzip(*trees):
    return Tree(tuple(t.item for t in trees), 
        children=[tzip(*children) for children in zip(*[t.children for t in trees])])

def to_node_parent(t):
    nodes = list(iter_items(t))
    parents = {}

    index = lambda x: next(i for i, v in enumerate(nodes) if v is x)

    for parent, child in iter_parent_child(t):
        if parent is None:
            continue
        parents[index(child)] = index(parent)

    return nodes, parents


