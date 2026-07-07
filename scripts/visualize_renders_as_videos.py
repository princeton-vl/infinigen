#!/usr/bin/env -S uv run --no-sync python
# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
import json
import logging
import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import cvdpack
import cvdpack.pack_frames as cvd_pack_frames
import numpy as np

from infinigen2.exporters.util.format import ExportType
from infinigen2.exporters.visualize_gt import visualize_any_frametype

logger = logging.getLogger(__name__)

try:
    import submitit
except ImportError:
    submitit = None

_SCRIPT_DIR = Path(__file__).parent
with open(_SCRIPT_DIR / "cvdpack_infinigen2.json") as _f:
    CVDPACK_CONFIG: dict = json.load(_f)["data_types"]

GT_TYPE_TO_EXPORT_TYPE: dict[str, ExportType | None] = {
    # None = passthrough: PNG frames from MKV are the visualization directly
    "rgb": None,
    "diffuse-color": None,
    "environment": None,
    # GT types requiring unpack + visualization
    "depth": ExportType.DEPTH,
    "optical-flow": ExportType.OPTICAL_FLOW,
    "surface-normal": ExportType.SURFACE_NORMAL,
    "semantic-segmentation": ExportType.OBJECT_INDEX,
    "material-segmentation": ExportType.MATERIAL_INDEX,
}

# optical-flow is packed as 2 channels, padded to 3 in PNG; trim on unpack
_UNPACK_CHANNELS_LAST: dict[str, int] = {
    "optical-flow": 2,
}


def _gt_type_display_name(gt_type: str) -> str:
    return " ".join(w.capitalize() for w in gt_type.replace("-", " ").split())


def draw_text_overlay(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad = 0.6, 1, 5
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        img,
        (0, 0),
        (pad + text_w + pad, pad + text_h + baseline + pad),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        img,
        text,
        (pad, pad + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img


def _find_mkv_channels(
    scene_folder: Path, gt_types: list[str], cameras: list[str] | None
) -> list[tuple[str, str]]:
    """Return [(gt_type, camera), ...] in gt_types order, then sorted cameras."""
    result = []
    for gt_type in gt_types:
        for mkv in sorted(scene_folder.glob(f"{gt_type}-*.mkv")):
            camera = mkv.stem[len(gt_type) + 1 :]
            if cameras is None or camera in cameras:
                result.append((gt_type, camera))
    return result


def _write_concat_list(vis_paths: list[Path], concat_path: Path, fps: int) -> None:
    with open(concat_path, "w") as f:
        for p in vis_paths:
            f.write(f"file '{p.resolve()}'\n")
            f.write(f"duration {1 / fps}\n")


def _vf_filter(resize_pct: float | None) -> str:
    parts = []
    if resize_pct is not None:
        parts.append(f"scale=iw*{resize_pct}:-2")
    parts.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    return ",".join(parts)


def _stitch_to_video(
    vis_paths: list[Path], output_path: Path, fps: int, resize_pct: float | None
) -> None:
    concat_list = output_path.parent / (output_path.stem + "_concat.txt")
    _write_concat_list(vis_paths, concat_list, fps)
    try:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-vf",
                _vf_filter(resize_pct),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
        )
    finally:
        concat_list.unlink(missing_ok=True)


def _xstack_layout(cols: int, rows: int, n: int) -> str:
    positions = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = "+".join(f"w{k}" for k in range(col)) if col > 0 else "0"
        y = "+".join(["h0"] * row) if row > 0 else "0"
        positions.append(f"{x}_{y}")
    return "|".join(positions)


def _assemble_grid(
    channels: list[tuple[str, str, list[Path]]],
    cols: int,
    rows: int,
    output_path: Path,
    fps: int,
    resize_pct: float | None,
    tmp: Path,
) -> None:
    n_slots = cols * rows
    n_real = len(channels)

    # Determine tile size from first real frame
    first_frame = channels[0][2][0]
    img0 = cv2.imread(str(first_frame))
    h, w = img0.shape[:2]
    if resize_pct is not None:
        w = int(w * resize_pct)
        h = int(h * resize_pct)
    # Ensure even dims for libx264
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    # Build ffmpeg inputs
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    concat_paths = []
    for i in range(n_slots):
        if i < n_real:
            gt_type, camera, vis_paths = channels[i]
            concat_path = tmp / f"concat_{i}.txt"
            _write_concat_list(vis_paths, concat_path, fps)
            concat_paths.append(concat_path)
            cmd += ["-f", "concat", "-safe", "0", "-i", str(concat_path)]
        else:
            # Black filler for empty grid slots
            cmd += ["-f", "lavfi", "-i", f"color=black:size={w}x{h}:rate={fps}"]

    # Build filter_complex
    filter_parts = []
    stream_labels = []
    for i in range(n_slots):
        if resize_pct is not None:
            label = f"[sv{i}]"
            filter_parts.append(f"[{i}:v]scale={w}:{h}[sv{i}]")
            stream_labels.append(label)
        else:
            stream_labels.append(f"[{i}:v]")

    layout = _xstack_layout(cols, rows, n_slots)
    xstack_in = "".join(stream_labels)
    filter_parts.append(f"{xstack_in}xstack=inputs={n_slots}:layout={layout}[out]")

    cmd += ["-filter_complex", ";".join(filter_parts)]
    cmd += ["-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)]

    subprocess.check_call(cmd)


def _visualize_channel(
    gt_type: str,
    camera: str,
    scene_folder: Path,
    tmp: Path,
    overlay: bool,
) -> list[Path]:
    """Unpack one MKV channel to visualized PNGs. Returns sorted vis PNG paths."""
    mkv_path = scene_folder / f"{gt_type}-{camera}.mkv"
    export_type = GT_TYPE_TO_EXPORT_TYPE[gt_type]
    packing_config = CVDPACK_CONFIG.get(gt_type, {}).get("packing")

    chan_tmp = tmp / f"{gt_type}_{camera}"
    chan_tmp.mkdir()
    png_template = chan_tmp / "raw_{frame:06d}.png"
    ffmpeg_tmp = chan_tmp / "ffmpeg_tmp"

    cvdpack.unpack_video(
        input_video_path=mkv_path,
        output_frames_path_template=png_template,
        tmp_folder=ffmpeg_tmp,
    )

    if export_type is None:
        # Passthrough: PNG frames from the MKV are already the visualization
        vis_paths = sorted(chan_tmp.glob("raw_*.png"))
    else:
        npy_template = chan_tmp / "frame_{frame:06d}.npy"
        packer = cvd_pack_frames.get_channel_packer(packing_config)
        unpack_channels_last = _UNPACK_CHANNELS_LAST.get(gt_type)
        cvd_pack_frames.unpack_frameset(
            input_path_template=png_template,
            output_path_template=npy_template,
            packer=packer,
            unpack_channels_last=unpack_channels_last,
        )

        npy_paths = sorted(chan_tmp.glob("frame_*.npy"))
        if not npy_paths:
            logger.warning(f"No npy frames produced for {mkv_path}, skipping")
            return []

        vis_paths = visualize_any_frametype(export_type, npy_paths, chan_tmp)

    if not vis_paths:
        logger.warning(f"No vis frames produced for {mkv_path}, skipping")
        return []

    if overlay:
        label = f"{scene_folder.name} | {_gt_type_display_name(gt_type)} {camera}"
        for p in vis_paths:
            img = cv2.imread(str(p))
            img = draw_text_overlay(img, label)
            cv2.imwrite(str(p), img)

    return vis_paths


def visualize_scene(payload: dict) -> list[Path]:
    scene_folder = Path(payload["scene_folder"])
    fps = payload["fps"]
    overlay = payload["overlay"]
    gt_types = payload["gt_types"]
    resize_pct = payload["resize_pct"]
    grid_dim = payload["grid_dim"]  # (cols, rows) or None
    tmp_root = Path(payload["tmp_root"]) if payload["tmp_root"] else None
    output_folder = (
        Path(payload["output_folder"]) if payload.get("output_folder") else None
    )

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )

    def _out_dir() -> Path:
        if output_folder is not None:
            d = output_folder / scene_folder.name
            d.mkdir(parents=True, exist_ok=True)
            return d
        return scene_folder

    cameras = payload["cameras"]
    channels = _find_mkv_channels(scene_folder, gt_types, cameras)
    if not channels:
        logger.warning(f"No visualizable MKVs found in {scene_folder}")
        return []

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_str:
        tmp = Path(tmp_str)

        if grid_dim is not None:
            # Grid mode: visualize all channels, then assemble one grid MP4
            cols, rows = grid_dim
            out_mp4 = _out_dir() / "vis_grid.mp4"
            if out_mp4.exists():
                logger.info(f"Skipping {out_mp4} (already exists)")
                return [out_mp4]

            vis_channels = []
            for gt_type, camera in channels:
                vis_paths = _visualize_channel(
                    gt_type, camera, scene_folder, tmp, overlay
                )
                if vis_paths:
                    vis_channels.append((gt_type, camera, vis_paths))

            if not vis_channels:
                return []

            logger.info(f"Assembling {cols}x{rows} grid -> {out_mp4}")
            _assemble_grid(vis_channels, cols, rows, out_mp4, fps, resize_pct, tmp)
            logger.info(f"Wrote {out_mp4}")
            return [out_mp4]

        else:
            # Individual mode: one MP4 per (gt_type, camera)
            output_paths = []
            for gt_type, camera in channels:
                out_mp4 = _out_dir() / f"vis_{gt_type}-{camera}.mp4"
                if out_mp4.exists():
                    logger.info(f"Skipping {out_mp4} (already exists)")
                    output_paths.append(out_mp4)
                    continue

                logger.info(f"Visualizing {gt_type}-{camera}")
                vis_paths = _visualize_channel(
                    gt_type, camera, scene_folder, tmp, overlay
                )
                if not vis_paths:
                    continue

                _stitch_to_video(vis_paths, out_mp4, fps, resize_pct)
                logger.info(f"Wrote {out_mp4}")
                output_paths.append(out_mp4)

            return output_paths


def _mapfunc(
    f, payloads: list[dict], n_workers: int, slurm: bool, log_folder: Path
) -> list:
    if n_workers == 1:
        return [f(p) for p in payloads]
    if not slurm:
        with Pool(n_workers) as pool:
            return list(pool.imap(f, payloads))
    if submitit is None:
        raise RuntimeError("submitit not installed; cannot use --slurm")
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        name="visualize_renders",
        cpus_per_task=2,
        mem_gb=16,
        slurm_partition=os.environ.get("INFINIGEN_SLURMPARTITION"),
        slurm_array_parallelism=n_workers,
    )
    jobs = executor.map_array(f, payloads)
    return [j.result() for j in jobs]


def find_scene_folders(roots: list[Path], glob_pattern: str) -> list[Path]:
    folders = []
    for root in roots:
        if root.is_dir() and any(root.glob("*.mkv")):
            folders.append(root)
        else:
            for candidate in sorted(root.glob(glob_pattern)):
                if candidate.is_dir() and any(candidate.glob("*.mkv")):
                    folders.append(candidate)
    return folders


def _resolve_tmp_root(candidates: list[str] | None) -> Path | None:
    if not candidates:
        return None
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    logger.warning(
        f"None of --tmp_folder candidates exist: {candidates}, using system default"
    )
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize packed renders as videos")
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="Scene folder(s) or parent directory"
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern when inputs are parent dirs (default: '*')",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--overlay", action="store_true", help="Draw gt-type label overlay on frames"
    )
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        metavar="CAM",
        help="Camera names to include (default: all). E.g. --cameras CameraLeft",
    )
    parser.add_argument(
        "--gt_types",
        nargs="+",
        default=list(GT_TYPE_TO_EXPORT_TYPE.keys()),
        choices=list(GT_TYPE_TO_EXPORT_TYPE.keys()),
        metavar="GT_TYPE",
        help=f"GT types to visualize (default: all). Choices: {list(GT_TYPE_TO_EXPORT_TYPE.keys())}",
    )
    parser.add_argument(
        "--resize_pct",
        type=float,
        default=None,
        help="Scale output frames (e.g. 0.5 for half size)",
    )
    parser.add_argument(
        "--grid_dim",
        type=str,
        default=None,
        metavar="CxR",
        help="If set (e.g. '3x2'), output a single grid MP4 per scene instead of individual MP4s",
    )
    parser.add_argument(
        "--tmp_folder",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Candidate temp directories; first existing one is used",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Process only the first N scene folders (default: all)",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Write output MP4s to OUTPUT_FOLDER/<scene_name>/ instead of into each scene folder",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )

    grid_dim = None
    if args.grid_dim is not None:
        cols, rows = map(int, args.grid_dim.split("x"))
        grid_dim = (cols, rows)

    tmp_root = _resolve_tmp_root(args.tmp_folder)

    scene_folders = find_scene_folders(args.inputs, args.glob)
    if not scene_folders:
        logger.error("No scene folders with MKV files found")
        return

    logger.info(f"Found {len(scene_folders)} scene folder(s)")

    if args.max_videos is not None:
        scene_folders = scene_folders[: args.max_videos]
        logger.info(f"Limiting to {len(scene_folders)} scene folder(s) (--max_videos)")

    payloads = [
        {
            "scene_folder": str(f),
            "fps": args.fps,
            "overlay": args.overlay,
            "cameras": args.cameras,
            "gt_types": args.gt_types,
            "resize_pct": args.resize_pct,
            "grid_dim": grid_dim,
            "tmp_root": str(tmp_root) if tmp_root else None,
            "output_folder": str(args.output_folder) if args.output_folder else None,
        }
        for f in scene_folders
    ]

    log_folder = args.inputs[0] / "visualize_logs"
    results = _mapfunc(
        visualize_scene, payloads, args.n_workers, args.slurm, log_folder
    )
    for paths in results:
        for p in paths:
            print(p)


if __name__ == "__main__":
    main()
