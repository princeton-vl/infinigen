#!/usr/bin/env python
"""Render a linear pan of a livingroom in several passes: clay-flat (no
displacement), clay (real displacement), rgb (full materials), plus ground-truth
depth/normal/object/flow."""

# ruff: noqa: I001, E402

import argparse
import logging
import os
import sys
from pathlib import Path

import bpy
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import procfunc as pf

from infinigen_v2.exporters.render_cycles import (
    render_cycles,
    render_cycles_ground_truth,
)
from infinigen_v2.exporters.render_cycles_clay import render_cycles_clay
from infinigen_v2.exporters.util.blender_render import DisplacementMode
from infinigen_v2.exporters.util.format import ExportType, RenderPass
from infinigen_v2.generators.cameras import monocular
from infinigen_v2.generators.scenes.room import room, room_shape
from infinigen_v2.util.render_metadata import time_step, write_render_metadata
from infinigen_v2.util.scene_cleanup import cleanup_except

logger = logging.getLogger(__name__)

AO_DISTANCE = 2.0


def main():
    parser = argparse.ArgumentParser(description="Render a clay orbit pan with GT")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/clay_pan_video"))
    parser.add_argument("--frames", type=int, nargs=2, default=(0, 35))
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--samples", type=int, default=256, help="clay/AO passes")
    parser.add_argument(
        "--rgb_samples", type=int, default=2048, help="full-material rgb pass"
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--skip_gt", action="store_true")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    frame_start, frame_end = args.frames
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    pf.ops.object.clear_scene()
    bpy.context.scene.render.resolution_x = args.resolution[0]
    bpy.context.scene.render.resolution_y = args.resolution[1]
    bpy.context.scene.render.fps = args.fps
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end

    logger.info("Building scene with seed %s", hex(seed))
    rng = np.random.default_rng(seed)
    rngs = rng.spawn(3)

    dimensions = room_shape.room_dimensions_distribution(rngs[0])

    times = {}

    with time_step(times, "livingroom"):
        living = room.livingroom_with_smallobj_distribution(
            rng=rngs[1], dimensions=dimensions, frame_start=0, frame_end=0
        )
    objects = list(living.all_objects)

    with time_step(times, "linear_pan_camera"):
        cameras = monocular.linear_pan_camera_distribution(
            rng=rngs[2],
            objects=objects,
            colliders=living.colliders,
            dimensions=dimensions,
            frame_start=frame_start,
            frame_end=frame_end,
        )
    camera = cameras[0]

    cleanup_except(objects + list(cameras) + list(living.lights))

    # Short AO-pass distance so only tight crevices / fine displacement darken; the
    # AO pass is shown directly as the clay image to surface small geometry.
    world = bpy.context.scene.world
    if world is not None and world.light_settings is not None:
        world.light_settings.distance = AO_DISTANCE

    render_kwargs = dict(
        objects=objects,
        camera=camera,
        output_folder=output,
        frame_start=frame_start,
        frame_end=frame_end,
        resolution=tuple(args.resolution),
        min_samples=32,
        film_exposure=2.0,
    )

    # ao-flat: occlusion of the undisplaced mesh; ao-disp: occlusion of the displaced
    # mesh (fine surface detail appears). The swipe goes ao-flat -> ao-disp -> rgb.
    clay_flat_passes = [
        RenderPass(ExportType.IMAGE, Path("%c/clay-flat-%f.png"), np.dtype(np.uint8)),
        RenderPass(
            ExportType.AMBIENT_OCCLUSION,
            Path("%c/ao-flat-%f.png"),
            np.dtype(np.uint8),
        ),
    ]
    clay_passes = [
        RenderPass(ExportType.IMAGE, Path("%c/clay-%f.png"), np.dtype(np.uint8)),
        RenderPass(
            ExportType.AMBIENT_OCCLUSION,
            Path("%c/ao-disp-%f.png"),
            np.dtype(np.uint8),
        ),
    ]
    rgb_passes = [
        RenderPass(ExportType.IMAGE, Path("%c/rgb-%f.png"), np.dtype(np.uint8)),
        RenderPass(ExportType.CAMERA, Path("%c/camera.npz"), np.dtype(np.float32)),
    ]
    gt_passes = [
        RenderPass(ExportType.DEPTH, Path("%c/depth-%f.npy"), np.dtype(np.float32)),
        RenderPass(
            ExportType.SURFACE_NORMAL,
            Path("%c/surface-normal-%f.npy"),
            np.dtype(np.float32),
        ),
        RenderPass(
            ExportType.OBJECT_INDEX, Path("%c/object-%f.npy"), np.dtype(np.uint32)
        ),
        RenderPass(
            ExportType.OPTICAL_FLOW,
            Path("%c/optical-flow-%f.npy"),
            np.dtype(np.float32),
        ),
    ]

    exports_list = []

    logger.info("Rendering clay_flat")
    with time_step(times, "clay_flat"):
        clay_flat_exports = render_cycles_clay(
            render_passes=clay_flat_passes,
            **render_kwargs,
            displacement_mode=DisplacementMode.NONE,
            fill_light=True,
            max_samples=args.samples,
        )
    exports_list.append(clay_flat_exports)

    logger.info("Rendering clay")
    with time_step(times, "clay"):
        clay_exports = render_cycles_clay(
            render_passes=clay_passes,
            **render_kwargs,
            displacement_mode=DisplacementMode.DISPLACEMENT_AND_BUMP,
            fill_light=True,
            max_samples=args.samples,
        )
    exports_list.append(clay_exports)

    logger.info("Rendering rgb")
    with time_step(times, "rgb"):
        rgb_exports = render_cycles(
            render_passes=rgb_passes,
            **render_kwargs,
            max_samples=args.rgb_samples,
        )
    exports_list.append(rgb_exports)

    if not args.skip_gt:
        gt_passes_kwargs = dict(render_passes=gt_passes, max_samples=args.samples)
        logger.info("Rendering gt")
        with time_step(times, "gt"):
            gt_exports = render_cycles_ground_truth(**render_kwargs, **gt_passes_kwargs)
        exports_list.append(gt_exports)

    all_exports: dict[ExportType, list[Path]] = {}
    for exports in exports_list:
        for k, v in exports.items():
            all_exports.setdefault(k, []).extend(v)

    write_render_metadata(
        output=output,
        seed=seed,
        times=times,
        exports=all_exports,
        build_keys={"livingroom", "linear_pan_camera"},
        render_keys={"clay_flat", "clay", "rgb", "gt"},
        n_frames=frame_end - frame_start + 1,
    )

    for paths in all_exports.values():
        for p in paths:
            print(p)


if __name__ == "__main__":
    main()
    sys.exit(0)
