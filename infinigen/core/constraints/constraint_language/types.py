# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from dataclasses import dataclass

nodedataclass_kwargs = dict(eq=False, order=False)


def _nodeclass_bool_throw(self):
    raise RuntimeError(
        f"Attempted to convert {self.__class__} to bool, "
        f"truth value of {self} is ambiguous. Constraint language must use  * instead of `and`, etc since python bool ops are not overridable"
    )


def nodedataclass(frozen=False):
    def decorator(cls):
        ddec = dataclass(eq=False, order=False, frozen=frozen)
        cls = ddec(cls)
        cls.__bool__ = _nodeclass_bool_throw
        return cls

    return decorator


@nodedataclass()
class Node:
    def children(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Node):
                yield k, v

    def traverse(self, inorder=True):
        if inorder:
            yield self
        for _, c in self.children():
            yield from c.traverse(inorder=inorder)
        if not inorder:
            yield self

    def size(self):
        return len(list(self.traverse()))
