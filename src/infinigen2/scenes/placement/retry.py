# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import Any, Callable, TypeVar

import procfunc as pf

__all__ = [
    "repeat_attempts",
]

T = TypeVar("T")


def repeat_attempts(
    func: Callable[..., T | None],
    rng: pf.RNG,
    attempts: int = 5,
    *args: Any,
    **kwargs: Any,
) -> T | None:
    if attempts < 1:
        raise ValueError(f"repeat_attempts needs attempts >= 1, got {attempts}")
    rngs = rng.spawn(attempts)
    for i in range(attempts):
        res = func(rngs[i], *args, **kwargs)
        if res is not None:
            return res
    if child := kwargs.get("child"):
        if isinstance(child, pf.MeshObject):
            child.item().name = child.item().name + f"_FAILED_{i}"
    return None
