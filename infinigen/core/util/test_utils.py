# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import importlib
from pathlib import Path

import gin

from infinigen.core import init, surface


def setup_gin(configs_folders, configs, overrides=None):
    gin.clear_config()
    init.apply_gin_configs(
        config_folders=configs_folders,
        configs=configs,
        overrides=overrides,
        skip_unknown=True,
        finalize_config=False,
    )
    surface.registry.initialize_from_gin()
    gin.unlock_config()


def import_item(name):
    *path_parts, name = name.split(".")
    with gin.unlock_config():
        try:
            return importlib.import_module("." + name, ".".join(path_parts))
        except ModuleNotFoundError:
            mod = importlib.import_module(".".join(path_parts))
            return getattr(mod, name)


def load_txt_list(path: Path, skip_sharp=True):
    path = Path(path)
    pathabs = path.absolute()

    if not pathabs.exists():
        raise FileNotFoundError(f"{path=} resolved to {pathabs=} which does not exist")

    res = pathabs.read_text().splitlines()

    res = [
        f.strip()
        for f in res
        if (not f.startswith("#") or not skip_sharp) and len(f) > 0
    ]

    res = [f[: f.index("#")] if "#" in f else f for f in res]

    res = [r for r in res if len(r.strip()) > 0]

    return res
