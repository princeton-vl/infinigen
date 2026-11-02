# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""No optimized package passes a parameter only another package's copy of a `core/`
module accepts, checked from the source so it covers packages no test exercises."""

import ast
import pathlib
import warnings

import pytest

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "infinigen"
PERF = SRC / "perf"

pytestmark = pytest.mark.skipif(
    not PERF.is_dir(), reason="no infinigen.perf in this tree"
)


def _parse(path):
    try:
        with warnings.catch_warnings():
            # stray escapes in docstrings would warn on every parse
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return None


def _params(fn):
    a = fn.args
    return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}


def _functions(path):
    tree = _parse(path)
    if tree is None:
        return {}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, _params(node))
    return out


def _default_counterpart(rel):
    for cand in (SRC / "core" / rel, SRC / "assets" / "objects" / rel, SRC / rel):
        if cand.exists():
            return cand
    return None


def _widened_shared_params():
    """{parameter: owning package} for parameters a perf copy of a `core/` module
    adds. Only `core/` is shared, since asset packages are selected as a whole."""
    out = {}
    for p in sorted(PERF.rglob("*.py")):
        rel = p.relative_to(PERF)
        default = SRC / "core" / rel
        if not default.exists():
            continue
        theirs, ours = _functions(default), _functions(p)
        for name, params in ours.items():
            for extra in params - theirs.get(name, params):
                out[extra] = rel.parts[0]
    return out


def _passes_keyword(path, keyword):
    """Whether `path` passes `keyword` to any call, directly or via a dict. Matching
    the name also catches wrappers that forward `**kwargs`."""
    tree = _parse(path)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(
            kw.arg == keyword for kw in node.keywords
        ):
            return True
        # kwargs assembled first, then splatted into the call
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == keyword
        ):
            return True
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == keyword for k in node.keys
        ):
            return True
    return False


# (using package, parameter) -> why the use is guarded. The syntax cannot tell
# a guarded use from an unguarded one, so these are checked by hand.
REVIEWED = {
    ("trees", "nearest_point_attributes"): (
        "passed only when the optimized placement is selected, checked at "
        "the call site; without it the remesh interpolates the skinning "
        "attributes, which is what the default implementation does"
    ),
}


def test_no_optimized_package_requires_another():
    offences = []
    for keyword, owner in sorted(_widened_shared_params().items()):
        for p in sorted(PERF.rglob("*.py")):
            caller = p.relative_to(PERF).parts[0]
            if caller == owner:
                continue  # one package, selected as a whole
            if not _passes_keyword(p, keyword):
                continue
            if (caller, keyword) in REVIEWED:
                continue
            offences.append(
                f"perf/{p.relative_to(PERF)} (package {caller!r}) passes "
                f"{keyword!r}, which only the optimized {owner!r} package "
                f"accepts"
            )
    assert not offences, (
        "optimized packages must work with every other flag off:\n  "
        + "\n  ".join(offences)
    )
