# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Iterable

import bpy
import procfunc as pf

__all__ = [
    "cleanup_except",
    "delete_object",
]

logger = logging.getLogger(__name__)


def delete_object(obj: bpy.types.Object) -> None:
    data = getattr(obj, "data", None)
    item_type = getattr(obj, "type", None)
    name = obj.name
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is None or not hasattr(data, "users") or data.users != 0:
        logger.debug(f"Deleting {name} ({item_type}) but NOT deleting its data")
        return
    else:
        logger.debug(f"Deleting {name} ({item_type}) and associated data")
    if item_type == "MESH":
        bpy.data.meshes.remove(data)
    elif item_type == "LIGHT":
        bpy.data.lights.remove(data)


def cleanup_except(keep: Iterable[pf.Object]) -> list[str]:
    """Delete every bpy.data.objects entry not in `keep`.

    Placement helpers (e.g. repeat_attempts) leave failed-placement objects
    in the scene. They are excluded from the returned `all_objects` lists but
    still live in the blend and would otherwise be rendered. Pass the union of
    objects/cameras/lights you want kept and this removes the rest.
    """
    valid = {o.item() for o in keep}
    cleaned = []
    for asset in list(bpy.data.objects):
        if asset in valid:
            continue
        cleaned.append(asset.name)
        asset.name = asset.name + "_CLEANED"
        delete_object(asset)
    logger.info(f"Cleaned {len(cleaned)} stray objects from scene")
    return cleaned
