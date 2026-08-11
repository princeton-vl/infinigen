#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Render a camera-trajectory generator over a frame range with the workbench
engine and encode the frames into a single H.264 MP4 for the integration
viewer.

Invoked per camera generator by launch.sh. Renders
``{scene} {camera} render_workbench`` into ``--output`` and then encodes the
per-frame PNGs in each camera subfolder into one ``image_<camera>.mp4`` so the
viewer shows the trajectory as a single looping clip instead of dozens of stills.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

FRAME_GLOB = "[0-9][0-9][0-9][0-9].png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs=2, default=[0, 47])
    parser.add_argument("--resolution", "-r", type=int, nargs=2, default=[640, 360])
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def infinigen_bin() -> str:
    candidate = Path(sys.executable).parent / "infinigen2"
    return str(candidate) if candidate.exists() else "infinigen2"


def render_command(args: argparse.Namespace) -> list[str]:
    if os.environ.get("INFINIGEN_COVERAGE"):
        cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--rcfile=pyproject.toml",
            "-m",
            "infinigen2",
        ]
    else:
        cmd = [infinigen_bin()]
    return [
        *cmd,
        args.scene,
        args.camera,
        "render_workbench",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
        "--passes",
        "rgb",
        "--frames",
        str(args.frames[0]),
        str(args.frames[1]),
        "-r",
        str(args.resolution[0]),
        str(args.resolution[1]),
        "--loglevel",
        "WARNING",
    ]


def render_frames(args: argparse.Namespace) -> int:
    cmd = render_command(args)
    print(f"[trajectory_video] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def frame_folders(output: Path) -> list[Path]:
    folders = {p.parent for p in output.rglob(FRAME_GLOB)}
    return sorted(folders)


def stack_mp4(folder: Path, mp4_path: Path, fps: int) -> None:
    frame_paths = sorted(folder.glob(FRAME_GLOB))
    first_frame = int(frame_paths[0].stem)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(max(fps, 1)),
        "-start_number",
        str(first_frame),
        "-i",
        str(folder / "%04d.png"),
        "-frames:v",
        str(len(frame_paths)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        # yuv420p needs even dimensions; round each down to the nearest even pixel
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)
    print(
        f"[trajectory_video] wrote {mp4_path} ({len(frame_paths)} frames)", flush=True
    )


def main() -> int:
    args = parse_args()
    rc = render_frames(args)

    folders = frame_folders(args.output)
    if not folders:
        print(
            "[trajectory_video] no frames rendered; nothing to encode", file=sys.stderr
        )
        return rc or 1

    for folder in folders:
        mp4_path = args.output / f"image_{folder.name}.mp4"
        stack_mp4(folder, mp4_path, args.fps)
        if not args.keep_frames:
            shutil.rmtree(folder)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
