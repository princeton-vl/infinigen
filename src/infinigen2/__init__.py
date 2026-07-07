# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
from pathlib import Path

import bpy
from pandas import read_json as _read_json

from .util.import_utils import module_path

__version__ = "2.0.0a1"

GENERATORS_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
GENERATORS_MANIFEST = _read_json(GENERATORS_MANIFEST_PATH)


def clear_default_scene():
    for name in ["Camera", "Light", "Cube"]:
        if o := bpy.data.objects.get(name):
            bpy.data.objects.remove(o)
    if c := bpy.data.collections.get("Collection"):
        bpy.data.collections.remove(c)


# clear_default_scene()


__all__ = [
    "context",
    "module_path",
    "__version__",
    "GENERATORS_MANIFEST",
    "GENERATORS_MANIFEST_PATH",
]
