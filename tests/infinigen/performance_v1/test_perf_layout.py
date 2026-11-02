# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""The layout infinigen/perf/README.md describes."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "infinigen"
PERF = SRC / "perf"


def _definitions(path):
    """{name: normalized AST} of top-level and class-level definitions,
    without decorators and docstrings."""
    out = {}

    def norm(node):
        node = ast.parse(ast.unparse(node)).body[0]
        node.decorator_list = []
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            node.body = body[1:] or [ast.Pass()]
        return ast.dump(node)

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = norm(node)
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        out[f"{node.name}.{item.name}"] = norm(item)
    return out


def test_every_perf_package_has_an_init():
    missing = [
        str(d.relative_to(PERF))
        for d in {p.parent for p in PERF.rglob("*.py")}
        if "__pycache__" not in d.parts and not (d / "__init__.py").exists()
    ]
    assert not missing, missing


def _bindings(path):
    """Where each module-level name comes from: the module it is imported
    from, or this module for its own definitions."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            names = (
                [node.name]
                if not isinstance(node, ast.Assign)
                else [t.id for t in node.targets if isinstance(t, ast.Name)]
            )
            for n in names:
                out[n] = f"{path}:{n}"
    return out


def _names_used(path, qualname):
    src = path.read_text()
    tree = ast.parse(src)
    node = tree
    for part in qualname.split("."):
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part
        )
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_no_counterpart_repeats_its_default():
    """A counterpart may equal its default only when a name it uses is rebound."""
    repeated = []
    for path in PERF.rglob("*.py"):
        default = SRC / path.relative_to(PERF)
        if path.name == "__init__.py" or not default.exists():
            continue
        mine, theirs = _definitions(path), _definitions(default)
        here, there = _bindings(path), _bindings(default)
        for name, node in mine.items():
            if theirs.get(name) != node:
                continue
            rebound = [
                n
                for n in _names_used(path, name)
                if n in here
                and here[n] != there.get(n)
                and not here[n].startswith(str(path))
            ]
            if not rebound:
                repeated.append(f"{path.relative_to(PERF)}::{name}")
    assert not repeated, repeated
