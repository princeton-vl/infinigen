# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import importlib
import pkgutil

import pytest

import infinigen2

# Third-party top-level packages that are legitimately optional in the lean/CI
# env; a missing one is a skip. A missing infinigen2 submodule is always a bug.
OPTIONAL_TOPLEVEL = {
    "infinigen_gpl",
    "skimage",
    "geomdl",
    "landlab",
    "terrain",
    "bnurbs",
    "mujoco",
    "coacd",
    "usd",
    "pxr",
    "wandb",
    "flask",
    "jinja2",
}


def _all_module_names() -> list[str]:
    names = ["infinigen2"]
    for info in pkgutil.walk_packages(infinigen2.__path__, prefix="infinigen2."):
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _all_module_names())
def test_module_imports(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        missing = e.name or ""
        top_level = missing.split(".")[0]
        if top_level == "infinigen2":
            raise
        if top_level not in OPTIONAL_TOPLEVEL:
            raise
        pytest.skip(f"optional dependency not installed: {missing}")
