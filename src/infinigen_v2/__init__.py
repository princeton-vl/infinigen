# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import logging

import bpy

from .util.import_utils import module_path

__version__ = "2.0.1-alpha"


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
]
