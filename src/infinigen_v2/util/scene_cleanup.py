import logging
from typing import Iterable

import bpy
import procfunc as pf

from infinigen_v2.generators.scenes.placement_utils import delete_object

logger = logging.getLogger(__name__)


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
