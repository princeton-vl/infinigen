# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import importlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_warned = set()


@lru_cache(maxsize=None)
def optional_import(module, attr=None, hint=None):
    """Import `module` (or its `attr`), or return None and log `hint` once so the caller
    can fall back to the default; `hint` should say how to build the kernel."""
    try:
        mod = importlib.import_module(module)
    except Exception as e:
        # Not just ImportError: a wrapper that loads its shared object at import
        # time raises RuntimeError when it was never built (as
        # infinigen.perf.assets.objects.monocot._pinecone.core does).
        if module not in _warned:
            _warned.add(module)
            logger.warning(
                "optional native kernel %s unavailable (%s); using the default "
                "implementation.%s",
                module,
                e,
                f" {hint}" if hint else "",
            )
        return None
    if attr is None:
        return mod
    value = getattr(mod, attr, None)
    if value is None and module not in _warned:
        _warned.add(module)
        logger.warning(
            "native kernel %s has no attribute %s; using the default implementation.",
            module,
            attr,
        )
    return value


def available(module, attr=None, hint=None):
    return optional_import(module, attr, hint) is not None
