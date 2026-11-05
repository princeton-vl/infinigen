# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import json
import logging
from collections import Counter
from pathlib import Path

import bpy
import imageio.v3 as iio
import numpy as np
import procfunc as pf
from mathutils import Vector

from infinigen2.exporters.util.format import ExportType

__all__ = ["check_backfacing"]

logger = logging.getLogger(__name__)

HIGHLIGHT_OPACITY = 0.45


def _camera_rays(camera: bpy.types.Object, resolution: tuple[int, int]) -> np.ndarray:
    width, height = resolution
    matrix = np.array(camera.matrix_world)
    frame = np.array([list(v) for v in camera.data.view_frame(scene=bpy.context.scene)])
    top_right, _, bottom_left, top_left = frame @ matrix[:3, :3].T + matrix[:3, 3]
    u = ((np.arange(width) + 0.5) / width)[None, :, None]
    v = ((np.arange(height) + 0.5) / height)[:, None, None]
    dirs = top_left + u * (top_right - top_left) + v * (bottom_left - top_left)
    dirs -= matrix[:3, 3]
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def _highlight(rgb_path: Path | None, bad: np.ndarray) -> np.ndarray:
    if rgb_path is None:
        base = np.full((*bad.shape, 3), 64.0)
    else:
        base = iio.imread(rgb_path)[..., :3].astype(np.float64)
    red = np.array([255.0, 0.0, 0.0])
    base[bad] = (1 - HIGHLIGHT_OPACITY) * base[bad] + HIGHLIGHT_OPACITY * red
    return base.astype(np.uint8)


def _rgb_paths(
    exports: list[dict[ExportType, list[Path]]], camera_name: str
) -> list[Path]:
    for export_type in (ExportType.IMAGE_DENOISED, ExportType.IMAGE):
        paths = [Path(p) for e in exports for p in e.get(export_type, [])]
        paths = sorted(p for p in paths if camera_name in p.parts)
        if paths:
            return paths
    return []


def _frame_report(
    camera: bpy.types.Object,
    resolution: tuple[int, int],
    frame: int,
    rgb_path: Path | None,
    camera_folder: Path,
) -> dict:
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    dirs = _camera_rays(camera, resolution)
    origin = camera.matrix_world.translation
    hit = np.zeros(dirs.shape[:2], dtype=bool)
    bad = np.zeros(dirs.shape[:2], dtype=bool)
    bad_by_object = Counter()
    for pixel in np.ndindex(*dirs.shape[:2]):
        direction = Vector(dirs[pixel])
        hit[pixel], _, normal, _, obj, _ = bpy.context.scene.ray_cast(
            depsgraph, origin, direction
        )
        if hit[pixel] and normal.dot(direction) > 0:
            bad[pixel] = True
            bad_by_object[obj.original.name] += 1

    entry = {
        "frame": frame,
        "bad_pixels": int(bad.sum()),
        "hit_pixels": int(hit.sum()),
        "bad_by_object": dict(bad_by_object),
        "image": None,
    }
    if entry["bad_pixels"] == 0:
        return entry
    path = camera_folder / f"backfacing_{frame:04d}.png"
    iio.imwrite(path, _highlight(rgb_path, bad))
    entry["image"] = str(path)
    logger.warning(
        f"{entry['bad_pixels']} backfacing pixels at {frame=}: "
        f"{entry['bad_by_object']}, see {path}"
    )
    return entry


@pf.tracer.primitive
def check_backfacing(
    camera: pf.CameraObject,
    output_folder: Path,
    exports: list[dict[ExportType, list[Path]]] | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
    resolution: tuple[int, int] = (1280, 720),
) -> dict[ExportType, list[Path]]:
    cam = camera.item()
    camera_folder = Path(output_folder) / cam.name
    camera_folder.mkdir(parents=True, exist_ok=True)
    rgb_paths = _rgb_paths(exports or [], cam.name)

    frames = []
    for i, frame in enumerate(range(frame_start, frame_end + 1)):
        rgb = rgb_paths[i] if i < len(rgb_paths) else None
        frames.append(_frame_report(cam, resolution, frame, rgb, camera_folder))

    report = json.dumps({"frames": frames}, indent=2)
    (camera_folder / "backfacing.json").write_text(report)
    vis_paths = [Path(f["image"]) for f in frames if f["image"] is not None]
    return {ExportType.VISUALIZATIONS: vis_paths}
