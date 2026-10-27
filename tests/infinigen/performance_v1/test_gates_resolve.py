# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Every gate resolves to a counterpart of its own once everything is selected, so a
missing, inherited or imported one fails here before an asset first calls it."""

import importlib
from pathlib import Path

import infinigen
from infinigen import _optimized

SRC = Path(__file__).resolve().parents[3] / "src"


def _gated_modules():
    for path in sorted((SRC / "infinigen").rglob("*.py")):
        if "perf" in path.relative_to(SRC / "infinigen").parts[:1]:
            continue
        text = path.read_text(errors="ignore")
        if "@optimizable" in text:
            yield ".".join(path.relative_to(SRC).with_suffix("").parts)


def test_every_gate_resolves_to_a_counterpart():
    modules = list(_gated_modules())
    assert modules
    for name in modules:
        importlib.import_module(name)
    infinigen.use_optimized(True)
    problems = []
    try:
        for resolve in list(_optimized._resolvers):
            resolve.cache_clear()
            try:
                resolve()
            except (AttributeError, TypeError) as e:
                problems.append(str(e))
    finally:
        # gates cache their resolution; clear it so later tests that patch a
        # counterpart see the patch
        for resolve in _optimized._resolvers:
            resolve.cache_clear()
    assert not problems, problems
