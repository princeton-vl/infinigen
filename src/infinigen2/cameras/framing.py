# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import numpy as np
import procfunc as pf

from infinigen2.util import camera_projection

__all__ = [
    "camera_with_distance_framing_object",
]


@pf.tracer.generator
def camera_with_distance_framing_object(
    target_object: pf.MeshObject,
    direction: pf.Vector,
    center_location: pf.Vector | None = None,
    camera: pf.CameraObject | None = None,
    margin_pct: float = 0.05,
    use_bbox: bool = False,
):
    """
    Args:
        margin_pct: What percent of the image width/height should remain empty around the object
        query_distance: distance to use for an initial projection of the target object.
    """

    if camera is None:
        camera = pf.ops.primitives.perspective_camera()
        camera_projection.adjust_camera_sensor(camera)

    resolution = camera_projection.bpy_resolution()

    bbox = pf.ops.attr.bbox_min_max(target_object, global_coords=True)
    if center_location is None:
        center_location = (bbox[0] + bbox[1]) / 2

    query_distance = max(bbox[1] - bbox[0]) * 2
    cam_rotation = (-direction).to_track_quat("-Z", "Y").to_euler()
    pf.ops.object.set_transform(
        camera,
        location=center_location + direction * query_distance,
        rotation_euler=cam_rotation,
    )

    if use_bbox:
        obj_points = pf.ops.attr.bbox_corners(target_object, global_coords=True)
    else:
        obj_points = pf.ops.attr.vertex_positions(target_object, global_coords=True)

    projected = camera_projection.project_points(camera, obj_points)
    u, v, _ = projected.T

    half_w, half_h = resolution[0] / 2, resolution[1] / 2
    right_extent = (u.max() - half_w) / half_w
    left_extent = (half_w - u.min()) / half_w
    bottom_extent = (v.max() - half_h) / half_h
    top_extent = (half_h - v.min()) / half_h

    max_extent = max(right_extent, left_extent, bottom_extent, top_extent)
    new_distance = query_distance * max_extent / (1 - margin_pct)
    assert not np.isnan(new_distance), new_distance
    assert not np.isinf(new_distance), new_distance

    new_point = center_location + direction * new_distance
    pf.ops.object.set_transform(camera, location=new_point, rotation_euler=cam_rotation)

    return camera
