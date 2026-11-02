# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Route gated definitions to their `infinigen.perf` counterparts when selected.

With nothing selected the gated definition runs as written. An optimized method
runs with `self` an instance of the default class, so it can only use what that
class has; `optimizable_class` substitutes the class itself instead.
"""

import functools
import inspect
import logging
from importlib import import_module

logger = logging.getLogger(__name__)

# Set by the tests that compare the two implementations against each other.
_verify_signatures = True

# Every gate's resolver, so that a new selection can reset them.
_resolvers = []


def reset() -> None:
    """Clear every gate's cached resolution after the selection changes."""
    for resolve in _resolvers:
        resolve.cache_clear()


def _selected(name: str) -> bool:
    from infinigen import _perf_selection

    return _perf_selection.selected(name)


@functools.lru_cache(maxsize=1)
def _perf_present() -> bool:
    """Whether this tree ships `infinigen.perf`."""
    from importlib.util import find_spec

    try:
        return find_spec("infinigen.perf") is not None
    except (ImportError, ValueError):
        return False


def _counterpart(module: str, qualname: str):
    """Definition named `qualname` in `module`, or None if the module is missing."""
    try:
        target = import_module(module)
    except ImportError as e:
        # A perf module that exists but fails to import is a bug, not a
        # missing optional package, so say which one and why.
        if _perf_present():
            logger.warning("infinigen.perf: %s could not be imported: %s", module, e)
        return None
    found = target
    for part in qualname.split("."):
        found = getattr(found, part, None)
        if found is None:
            # Running the default here would look like the optimization
            # working, so fail instead.
            raise AttributeError(
                f"{module} has no {qualname}; the gate for it would silently "
                f"run the default implementation"
            )
    if hasattr(found, "resolve") and hasattr(found, "default"):
        # the gated default itself (inherited or imported): dispatching to it
        # would recurse
        raise TypeError(
            f"{module}.{qualname} is the gated default, not a counterpart; "
            f"define the counterpart or remove the gate"
        )
    return found


def _opaque(signature) -> bool:
    """True for a `(*args, **kwargs)` signature, which accepts anything."""
    kinds = [p.kind for p in signature.parameters.values()]
    return kinds == [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]


def _check_signature(default, optimized, qualname: str) -> None:
    """Raise if the counterpart's signature differs from the default's."""
    if not _verify_signatures:
        return
    try:
        want = inspect.signature(default)
        got = inspect.signature(optimized)
    except (TypeError, ValueError):  # builtins, C extensions
        return
    if _opaque(want) or _opaque(got):
        # a wrapper without functools.wraps (node_utils.to_nodegroup's init_fn)
        # has nothing to compare
        return
    if want != got:
        raise TypeError(
            f"{qualname}: the optimized implementation takes {got}, "
            f"but the default takes {want}"
        )


def optimizable(name: str, module: str, qualname: str | None = None):
    """Call the same-named definition in `module` when package `name` is selected.

    Put `@gin.configurable` above the gate so its bindings apply first. Pass
    `qualname` when a decorator below the gate hides the real name, as
    `node_utils.to_nodegroup` does.
    """

    def decorator(fn):
        target = qualname or fn.__qualname__
        if "<locals>" in target:
            raise TypeError(
                f"cannot tell which counterpart {target!r} should dispatch to: "
                f"a decorator beneath the gate hides the name. Pass it as "
                f"qualname=."
            )

        @functools.lru_cache(maxsize=1)
        def resolve():
            if not _selected(name):
                return None
            optimized = _counterpart(module, target)
            if optimized is None:
                return None
            _check_signature(fn, optimized, target)
            return optimized

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            optimized = resolve()
            if optimized is None:
                return fn(*args, **kwargs)
            return optimized(*args, **kwargs)

        _resolvers.append(resolve)
        wrapper.resolve = resolve
        wrapper.default = fn
        return wrapper

    return decorator


def optimizable_class(name: str, module: str):
    """Construct the same-named subclass from `module` when `name` is selected.

    The counterpart must subclass this class. Subclasses defined elsewhere are
    constructed as themselves.
    """

    def decorator(cls):
        @functools.lru_cache(maxsize=1)
        def resolve():
            if not _selected(name):
                return cls
            optimized = _counterpart(module, cls.__name__)
            if optimized is None:
                return cls
            if not issubclass(optimized, cls):
                raise TypeError(
                    f"{cls.__name__}: the optimized class must subclass the "
                    f"default one so that instances remain usable everywhere"
                )
            return optimized

        _resolvers.append(resolve)
        original_new = cls.__new__

        def __new__(actual_cls, *args, **kwargs):
            if actual_cls is cls:
                actual_cls = resolve()
            if original_new is object.__new__:
                return object.__new__(actual_cls)
            return original_new(actual_cls, *args, **kwargs)

        cls.__new__ = staticmethod(__new__)
        cls.resolve = staticmethod(resolve)
        return cls

    return decorator
