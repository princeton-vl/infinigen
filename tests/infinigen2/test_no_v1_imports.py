# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import ast
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent


def _v1_import(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "infinigen" or alias.name.startswith("infinigen."):
                return alias.name
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        if node.module == "infinigen" or node.module.startswith("infinigen."):
            return node.module
    return None


@pytest.mark.parametrize(
    "path",
    sorted(_TESTS_DIR.rglob("*.py")),
    ids=lambda p: str(p.relative_to(_TESTS_DIR)),
)
def test_no_v1_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [name for node in ast.walk(tree) if (name := _v1_import(node))]
    assert not offenders, (
        f"{path.relative_to(_TESTS_DIR)} imports v1 package(s) {offenders}; "
        f"tests/infinigen2 must depend only on infinigen2 (v1 needs the "
        f"'infinigen1' extra, absent in the v2 CI env)."
    )
