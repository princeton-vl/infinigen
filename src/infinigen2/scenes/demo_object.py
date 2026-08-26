# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import numpy as np
import procfunc as pf
from procfunc import types as t

from infinigen2.cameras import camera_with_distance_framing_objects
from infinigen2.lighting import sky_lighting
from infinigen2.scenes.demo_util import (
    DevSceneResult,
    demo_cube,
    grid_plane,
    scale_reference,
)

__all__ = [
    "object_demo",
]

logger = logging.getLogger(__name__)


@pf.tracer.grammar
def object_demo(
    rng: pf.RNG,
    obj: pf.MeshObject | None = None,
    all_objects: list[pf.MeshObject] | None = None,
    camera: pf.CameraObject | None = None,
    environment: pf.World | None = None,
) -> DevSceneResult:
    if obj is not None:
        assert all_objects is None, "object_demo takes obj or all_objects, not both"
        all_objects = [obj]
    if all_objects is None:
        logger.warning("No object provided; using a default object.")
        all_objects = [demo_cube()]

    bounds = [pf.ops.attr.bbox_min_max(o, global_coords=True) for o in all_objects]
    bbox_min = np.min([lo for lo, _ in bounds], axis=0)
    bbox_max = np.max([hi for _, hi in bounds], axis=0)
    for o in all_objects:
        o.item().location.z -= bbox_min[-1]
    bbox_max[-1] -= bbox_min[-1]
    bbox_min[-1] = 0.0

    if camera is None:
        camera = camera_with_distance_framing_objects(
            all_objects, t.Vector((1, 1, 0.4)), margin_pct=0.1, use_bbox=True
        )

    if environment is None:
        environment = sky_lighting.nishita_sky(
            sun_rotation_deg=200,
            sun_elevation_deg=30,
        ).environment
    background = grid_plane()

    ref_rad = 0.3
    pos = (bbox_min[0] - ref_rad - 0.1, 0, 0)
    scale_ref = scale_reference(location=pos, radius=ref_rad)

    return DevSceneResult(
        environment=environment,
        all_objects=[*all_objects, background, scale_ref],
        cameras=[camera],
    )
