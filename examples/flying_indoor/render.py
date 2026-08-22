#!/usr/bin/env python
"""Render a stereo video with ground truth for the left camera."""

# ruff: noqa: I001, E402

import argparse
import logging
import os
from functools import partial
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
from procfunc.util.teardown import skip_teardown_on_exit

from infinigen2.exporters.object_data import save_object_data
from infinigen2.exporters.render_cycles import (
    render_cycles,
    render_cycles_ground_truth,
)
from infinigen2.exporters.util.blender_render import configure_object_index_table
from infinigen2.exporters.util.format import ExportType, RenderPass
from infinigen2.animations.random_walk import random_walk
from infinigen2.cameras import (
    attach_stereo_right,
    sample_baseline,
    stereo_accept_pred,
)
from infinigen2.cameras.util import pose_and_filter
from infinigen2.scenes import floating_objects
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.room import room, room_shape
from infinigen2.util.errors import RejectedScene
from infinigen2.util.polycount import estimated_eval_tricount
from infinigen2.util.render_metadata import time_step, write_render_metadata
from infinigen2.util.scene_cleanup import cleanup_except

logger = logging.getLogger(__name__)


def _parse_seed(value: str) -> int:
    return int(value, 0)


def _place_and_walk_camera(
    r: pf.RNG,
    camera: pf.CameraObject,
    cam_bbox: tuple[np.ndarray, np.ndarray],
    dimensions,
    colliders,
    accept_pred,
    frame_start: int,
    frame_end: int,
) -> pf.CameraObject | None:
    """Pose the camera in the low-x half of the room, aimed into it rather than
    at the near x=0 wall, then walk it through the scene."""
    lo, hi = cam_bbox
    x_hi = min(hi[0], float(dimensions[0]) / 2)

    def pose_rand(rr: pf.RNG) -> tuple:
        x = pf.random.uniform(rr, lo[0], x_hi)
        y = pf.random.uniform(rr, lo[1], hi[1])
        z = pf.random.clip_gaussian(rr, 1.5, 0.4, lo[2], hi[2])
        pitch = pf.random.clip_gaussian(rr, np.pi / 2, 0.3, np.pi / 4, 3 * np.pi / 4)
        roll = pf.random.clip_gaussian(rr, 0.0, 0.05, -0.2, 0.2)
        yaw = -np.pi / 2 + pf.random.clip_gaussian(
            rr, 0.0, np.deg2rad(30), -np.pi / 2, np.pi / 2
        )
        return (x, y, z), (pitch, roll, yaw)

    if pose_and_filter(r, camera, pose_rand, colliders, accept_pred) is None:
        return None
    return random_walk(
        r,
        camera,
        cam_bbox,
        frame_start=frame_start,
        frame_end=frame_end,
        accept_fn=lambda: accept_pred(camera, colliders),
        failure_mode="return",
        speed_mps_range=(1.5, 2.25),
        rot_std_deg=(11.25, 11.25, 15.0),
        height_range=(0.5, 2.2),
    )


def build_scene(
    seed: int,
    trajectory_seed: int | None = None,
    frame_start: int = 0,
    frame_end: int = 23,
    resolution: tuple[int, int] = (1280, 720),
):
    """Build the furnished room, floating objects/lights and the biased stereo
    camera. Returns (objects, render_lights, cameras, times)."""
    if trajectory_seed is None:
        trajectory_seed = seed
    pf.ops.object.clear_scene()
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.fps = 8
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end

    logger.info("Building scene %s trajectory %s", hex(seed), hex(trajectory_seed))
    rng = np.random.default_rng(seed)
    rngs = rng.spawn(6)

    dimensions = room_shape.room_dimensions_rand(rngs[0])
    room_bbox = (np.zeros(3), np.array(dimensions))

    times = {}

    with time_step(times, "room"):
        living = room.room_rand(
            rng=rngs[1],
            dimensions=dimensions,
            frame_start=frame_start,
            frame_end=frame_end,
        )
    objects = list(living.all_objects)

    with time_step(times, "floating_objects"):
        floating = floating_objects.floating_objects_rand(
            rng=rngs[2],
            colliders=living.colliders,
            bbox=room_bbox,
            volume_density=0.15625,
        )
    objects += floating.all_objects

    with time_step(times, "floating_lights"):
        light_result = floating_objects.floating_lights_rand(
            rng=rngs[5],
            colliders=floating.colliders,
            bbox=room_bbox,
        )
    lights = light_result.all_objects

    # Animate the floating objects and lights with a shared random walk
    bbox_min, bbox_max = room_bbox
    walkers = list(floating.all_objects) + list(lights)
    walk_rngs = rngs[3].spawn(len(walkers))
    for r, obj in zip(walk_rngs, walkers):
        random_walk(
            r,
            obj,
            room_bbox,
            frame_start=frame_start,
            frame_end=frame_end,
            failure_mode="warn",
            speed_mps_range=(0.3, 1.5),
            loc_step_range=(0.2, 1.0),
            rot_std_deg=(5.0, 5.0, 10.0),
            roll_range_deg=(-180.0, 180.0),
            pitch_range_deg=(0.0, 180.0),
        )

    cam_rng = rngs[4]
    cam_rngs = cam_rng.spawn(2)

    cam_bbox_margin = 0.5
    cam_bbox = (bbox_min + cam_bbox_margin, bbox_max - cam_bbox_margin)

    with time_step(times, "stereo_camera"):
        # the rig is a property of the scene; trajectory_seed varies only the path
        baseline = sample_baseline(cam_rngs[1])
        accept_pred = stereo_accept_pred(baseline)
        camera_left = pf.ops.primitives.perspective_camera(focal_length_mm=15)
        place = partial(
            _place_and_walk_camera,
            camera=camera_left,
            cam_bbox=cam_bbox,
            dimensions=dimensions,
            colliders=floating.colliders,
            accept_pred=accept_pred,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        traj_rng = np.random.default_rng(trajectory_seed)
        if repeat_attempts(place, traj_rng, attempts=40) is None:
            raise RejectedScene("Could not place random-walk camera")
        cameras = attach_stereo_right(camera_left, baseline)
    render_lights = list(living.lights) + list(lights)

    cleanup_except(objects + render_lights + list(cameras))
    return objects, render_lights, cameras, times


def main():
    parser = argparse.ArgumentParser(description="Render stereo video with GT")
    parser.add_argument("--seed", type=_parse_seed, default=None)
    parser.add_argument(
        "--trajectory_seed",
        type=_parse_seed,
        default=None,
        help="Seed for the camera path only; defaults to --seed. Vary it while "
        "holding --seed fixed to render several trajectories through one scene.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/flying_indoor"))
    parser.add_argument(
        "--camera_idx",
        type=int,
        required=True,
        choices=[0, 1],
        help="Render one camera of the stereo pair (0=left, 1=right) so the two "
        "cameras can be sharded across tasks. Both cameras of a seed write into the "
        "same output dir, so per-camera shards pack together into one scene.",
    )
    parser.add_argument("--save_blend", type=Path, default=None)
    parser.add_argument(
        "--max_render_tris",
        type=int,
        default=32_000_000,
        help="Reject the scene if its render-level triangle count exceeds this; such "
        "scenes thrash RAM and time out rather than fail cleanly. 0 disables.",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    traj_seed = args.trajectory_seed if args.trajectory_seed is not None else seed
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    frame_start, frame_end = 0, 23
    frames = (frame_start, frame_end)
    resolution = (1280, 720)

    objects, render_lights, cameras, times = build_scene(
        seed, traj_seed, frame_start, frame_end, resolution
    )
    camera_left, camera_right = cameras

    if args.save_blend is not None:
        pf.ops.file.save_blend(output_path=args.save_blend)
        return

    render_tris = sum(estimated_eval_tricount(obj) for obj in objects)
    logger.info("Render-level triangles: %d", render_tris)
    if args.max_render_tris and render_tris > args.max_render_tris:
        raise RejectedScene(
            f"{render_tris} render-level triangles exceeds bound {args.max_render_tris}"
        )

    render_kwargs = dict(
        output_folder=output,
        frame_start=frame_start,
        frame_end=frame_end,
        resolution=resolution,
        min_samples=32,
        max_samples=512,
        film_exposure=2.0,
        objects=objects,
        lights=render_lights,
    )

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

    right_gt_passes = [
        RenderPass(ExportType.DEPTH, Path("%c/depth_%f.npy"), np.dtype(np.float32)),
    ]

    collected = []
    render_keys = set()

    # Scene-scoped data stays inside this shard, so independent array tasks never share it.
    with time_step(times, "object_data"):
        # the right shard renders no object-index pass to assign pass_index for us
        configure_object_index_table()
        object_data = save_object_data(objects, output, frames[0], frames[1])
        collected.append(object_data)
    render_keys.add("object_data")

    # --camera_idx selects one camera of the pair for sharding.
    if args.camera_idx == 0:
        logger.info("Rendering left camera (rgb)")
        with time_step(times, "render_left_rgb"):
            left_rgb = render_cycles(
                camera=camera_left,
                render_passes=rgb_passes + left_render_data_passes,
                **render_kwargs,
            )
        logger.info("Rendering left camera (ground truth)")
        with time_step(times, "render_left_gt"):
            left_gt = render_cycles_ground_truth(
                camera=camera_left, render_passes=gt_passes, **render_kwargs
            )
        collected += [left_rgb, left_gt]
        render_keys |= {"render_left_rgb", "render_left_gt"}
    elif args.camera_idx == 1:
        logger.info("Rendering right camera (rgb)")
        with time_step(times, "render_right_rgb"):
            right_rgb = render_cycles(
                camera=camera_right, render_passes=rgb_passes, **render_kwargs
            )
        logger.info("Rendering right camera (ground truth)")
        with time_step(times, "render_right_gt"):
            right_gt = render_cycles_ground_truth(
                camera=camera_right, render_passes=right_gt_passes, **render_kwargs
            )
        collected += [right_rgb, right_gt]
        render_keys |= {"render_right_rgb", "render_right_gt"}
    else:
        raise ValueError(f"Invalid camera_idx: {args.camera_idx}")

    all_exports: dict[ExportType, list[Path]] = {}
    for exports in collected:
        for k, v in exports.items():
            all_exports.setdefault(k, []).extend(v)

    n_frames = frame_end - frame_start + 1
    build_keys = {"room", "floating_objects", "stereo_camera"}
    write_render_metadata(
        output=output,
        seed=seed,
        times=times,
        exports=all_exports,
        build_keys=build_keys,
        render_keys=render_keys,
        n_frames=n_frames,
        trajectory_seed=traj_seed,
    )

    for paths in all_exports.values():
        for p in paths:
            print(p)


if __name__ == "__main__":
    with skip_teardown_on_exit():
        main()
