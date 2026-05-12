#!/usr/bin/env python
"""Render a stereo video with ground truth for the left camera."""

# ruff: noqa: I001, E402

import argparse
import json
import logging
import os
import sys
import time
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
from infinigen_v2.exporters.util.format import ExportType, RenderPass
from infinigen_v2.generators.animations.random_walk import RandomWalkSampler, walk_loop
from infinigen_v2.generators.cameras import stereo
from infinigen_v2.generators.scenes import floating_objects
from infinigen_v2.generators.scenes.room import room, room_shape
from infinigen_v2.util.hardware_info import get_hardware_info
from infinigen_v2.util.scene_cleanup import cleanup_except

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Render stereo video with GT")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/stereo_video"))
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    render_kwargs = dict(
        output_folder=output,
        frame_start=0,
        frame_end=23,
        resolution=(1280, 720),
        min_samples=32,
        max_samples=512,
        film_exposure=2.0,
    )

    pf.ops.object.clear_scene()
    bpy.context.scene.render.resolution_x = render_kwargs["resolution"][0]
    bpy.context.scene.render.resolution_y = render_kwargs["resolution"][1]
    bpy.context.scene.render.fps = 8
    bpy.context.scene.frame_start = render_kwargs["frame_start"]
    bpy.context.scene.frame_end = render_kwargs["frame_end"]

    # Build scene
    logger.info("Building scene with seed %s", hex(seed))
    rng = np.random.default_rng(seed)
    rngs = rng.spawn(6)

    dimensions = room_shape.room_dimensions_distribution(rngs[0])
    room_bbox = (np.zeros(3), np.array(dimensions))

    generator_times = {}

    t = time.perf_counter()
    living = room.livingroom_distribution(
        rng=rngs[1],
        dimensions=dimensions,
        frame_start=render_kwargs["frame_start"],
        frame_end=render_kwargs["frame_end"],
    )
    objects = list(living.all_objects)
    generator_times["livingroom"] = time.perf_counter() - t

    t = time.perf_counter()
    floating = floating_objects.floating_objects_distribution(
        rng=rngs[2],
        colliders=living.colliders,
        bbox=room_bbox,
        volume_density=0.15625,
    )
    objects += floating.all_objects
    generator_times["floating_objects"] = time.perf_counter() - t

    # Animate a random subset of floating objects with biased random walks
    fly_objs = floating.all_objects
    bbox_min, bbox_max = room_bbox
    bbox_center = (bbox_min + bbox_max) / 2
    n_frames = max(render_kwargs["frame_end"] - render_kwargs["frame_start"], 1)
    obj_rng = rngs[3]
    obj_rngs = obj_rng.spawn(len(fly_objs))
    for i, obj in enumerate(fly_objs):
        if obj_rngs[i].random() > 0.5:
            continue
        r = obj_rngs[i]
        target_loc = r.uniform(bbox_min, bbox_max)
        loc_bias = (target_loc - bbox_center) / n_frames
        sampler = RandomWalkSampler(
            bbox=room_bbox,
            speed_mps_range=(0.3, 1.5),
            loc_step_range=(0.2, 1.0),
            rot_std_deg=(5.0, 5.0, 10.0),
            roll_range_deg=(-180.0, 180.0),
            pitch_range_deg=(0.0, 180.0),
            loc_bias=loc_bias,
        )
        walk_loop(
            rng=r,
            obj=obj,
            sampler=sampler,
            accept_fn=lambda: True,
            frame_start=render_kwargs["frame_start"],
            frame_end=render_kwargs["frame_end"],
            failure_mode="warn",
        )

    t = time.perf_counter()
    light_rng = rngs[5]
    light_result = floating_objects.floating_lights_distribution(
        rng=light_rng,
        colliders=floating.colliders,
        bbox=room_bbox,
        max_lights=3,
    )
    lights = light_result.all_objects
    light_walk_rngs = light_rng.spawn(max(len(lights), 1))
    for i, light in enumerate(lights):
        r = light_walk_rngs[i]
        target_loc = r.uniform(bbox_min, bbox_max)
        loc_bias = (target_loc - bbox_center) / n_frames
        sampler = RandomWalkSampler(
            bbox=room_bbox,
            speed_mps_range=(0.5, 2.0),
            loc_step_range=(0.3, 1.5),
            rot_std_deg=(0.0, 0.0, 0.0),
            roll_range_deg=(0.0, 0.0),
            pitch_range_deg=(0.0, 0.0),
            loc_bias=loc_bias,
        )
        walk_loop(
            rng=r,
            obj=light,
            sampler=sampler,
            accept_fn=lambda: True,
            frame_start=render_kwargs["frame_start"],
            frame_end=render_kwargs["frame_end"],
            failure_mode="warn",
        )
    generator_times["floating_lights"] = time.perf_counter() - t

    cam_rng = rngs[4]
    cam_rngs = cam_rng.spawn(2)

    cam_bbox_margin = 0.5
    cam_bbox = (bbox_min + cam_bbox_margin, bbox_max - cam_bbox_margin)

    cam_target = cam_rngs[0].uniform(*cam_bbox)
    loc_bias = (cam_target - bbox_center) / n_frames

    t = time.perf_counter()
    cameras = stereo.stereo_random_walk_camera(
        rng=cam_rngs[1],
        colliders=floating.colliders,
        objects=objects,
        frame_start=render_kwargs["frame_start"],
        frame_end=render_kwargs["frame_end"],
        bbox=cam_bbox,
        rot_std_deg=(7.5, 7.5, 15.0),
        loc_bias=loc_bias,
        speed_mps_range=(1.5, 2.25),
    )
    generator_times["stereo_camera"] = time.perf_counter() - t
    camera_left, camera_right = cameras
    render_kwargs["objects"] = objects

    cleanup_except(objects + list(cameras) + list(living.lights) + list(lights))

    rgb_passes = [
        RenderPass(ExportType.IMAGE, Path("%c/%f.png"), np.dtype(np.uint8)),
        RenderPass(ExportType.CAMERA, Path("%c/camera.npz"), np.dtype(np.float32)),
    ]

    left_render_data_passes = [
        RenderPass(
            ExportType.MATERIAL_INDEX,
            Path("%c/material-index_%f.npy"),
            np.dtype(np.uint32),
        ),
        RenderPass(
            ExportType.DIFFUSE_COLOR,
            Path("%c/diffuse-color_%f.png"),
            np.dtype(np.uint8),
        ),
        RenderPass(
            ExportType.ENVIRONMENT, Path("%c/environment_%f.png"), np.dtype(np.uint8)
        ),
    ]

    gt_passes = [
        RenderPass(ExportType.DEPTH, Path("%c/depth_%f.npy"), np.dtype(np.float32)),
        RenderPass(
            ExportType.SURFACE_NORMAL,
            Path("%c/surface-normal_%f.npy"),
            np.dtype(np.float32),
        ),
        RenderPass(
            ExportType.OBJECT_INDEX, Path("%c/object_%f.npy"), np.dtype(np.uint32)
        ),
        RenderPass(
            ExportType.OPTICAL_FLOW,
            Path("%c/optical-flow_%f.npy"),
            np.dtype(np.float32),
        ),
    ]

    logger.info("Rendering left camera (rgb)")
    t = time.perf_counter()
    left_exports = render_cycles(
        camera=camera_left,
        render_passes=rgb_passes + left_render_data_passes,
        **render_kwargs,
    )
    generator_times["render_left_rgb"] = time.perf_counter() - t

    logger.info("Rendering right camera (rgb)")
    t = time.perf_counter()
    right_exports = render_cycles(
        camera=camera_right,
        render_passes=rgb_passes,
        **render_kwargs,
    )
    generator_times["render_right_rgb"] = time.perf_counter() - t

    logger.info("Rendering left camera (ground truth)")
    t = time.perf_counter()
    left_gt_exports = render_cycles_ground_truth(
        camera=camera_left,
        render_passes=gt_passes,
        **render_kwargs,
    )
    generator_times["render_left_gt"] = time.perf_counter() - t

    right_gt_passes = [
        RenderPass(ExportType.DEPTH, Path("%c/depth_%f.npy"), np.dtype(np.float32)),
    ]

    logger.info("Rendering right camera (ground truth)")
    t = time.perf_counter()
    right_gt_exports = render_cycles_ground_truth(
        camera=camera_right,
        render_passes=right_gt_passes,
        **render_kwargs,
    )
    generator_times["render_right_gt"] = time.perf_counter() - t

    all_exports: dict[ExportType, list[Path]] = {}
    for exports in [left_exports, right_exports, left_gt_exports, right_gt_exports]:
        for k, v in exports.items():
            all_exports.setdefault(k, []).extend(v)

    metadata = {
        "seed": hex(seed),
        "hardware": get_hardware_info(),
        "generator_times": generator_times,
        "exports": {str(k): [str(p) for p in v] for k, v in all_exports.items()},
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for paths in all_exports.values():
        for p in paths:
            print(p)


if __name__ == "__main__":
    main()
    sys.exit(0)
