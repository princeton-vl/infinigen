#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Jack Nugent

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch integration_v2 renders across GPUs and per-GPU parallel slots."
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default="",
        help='GPU selection: empty=all, "available"=memory heuristic, or csv like "0,1".',
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args forwarded to infinigen2.list and launch.sh.",
    )
    return parser.parse_args()


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return proc.stdout


def parse_csv_ids(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def all_gpu_ids() -> list[str]:
    out = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def available_gpu_ids() -> list[str]:
    used_max_mb = int(os.environ.get("GPU_MEM_USED_MAX_MB", "10000"))
    sleep_seconds = int(os.environ.get("GPU_WAIT_SLEEP_SECONDS", "300"))
    max_retries = int(os.environ.get("GPU_WAIT_MAX_RETRIES", "12"))

    retries = 0
    while True:
        out = run_capture(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        selected: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            gpu_id, used_mb = parts
            try:
                if int(used_mb) < used_max_mb:
                    selected.append(gpu_id)
            except ValueError:
                continue

        if selected:
            return selected

        retries += 1
        if max_retries > 0 and retries >= max_retries:
            raise RuntimeError(f"No available GPUs after {retries} retries")

        time.sleep(sleep_seconds)


def resolve_gpu_ids(gpus_arg: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found")

    token = gpus_arg.strip()
    if token == "":
        ids = all_gpu_ids()
    elif token == "available":
        ids = available_gpu_ids()
    else:
        ids = parse_csv_ids(token)

    if not ids:
        raise RuntimeError("No GPU ids selected")
    return ids


def list_category(category: str, extra_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "infinigen2.list",
        "--categories",
        category,
        "--missing_values",
        "drop",
        "--columns",
        "shortname",
    ]
    if extra_args:
        cmd.extend(extra_args)
    out = run_capture(cmd)
    return [line.strip() for line in out.splitlines() if line.strip()]


def shard_items(items: list[str], num_shards: int, shard_index: int, limit: int) -> str:
    if limit == 0:
        return ""
    if limit > 0:
        items = items[:limit]
    shard = [item for idx, item in enumerate(items) if idx % num_shards == shard_index]
    return "\n".join(shard)


def count_items(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def failed_render_names(output_path: Path) -> list[str]:
    events_dir = output_path / "render_index" / "events"
    if not events_dir.is_dir():
        return []

    failed: list[str] = []
    for event_path in events_dir.glob("*.json"):
        try:
            payload = json.loads(event_path.read_text())
        except Exception:
            continue
        if payload.get("returncode", 0) != 0:
            failed.append(payload.get("asset_dir") or event_path.stem)
    return failed


def render_runner(output_path: Path) -> str:
    python_bin = Path(".venv/bin/python")
    infinigen_bin = Path(".venv/bin/infinigen2")
    if not python_bin.exists():
        raise RuntimeError("Expected .venv/bin/python to exist")
    if not infinigen_bin.exists():
        raise RuntimeError("Expected .venv/bin/infinigen2 to exist")
    return (
        f"{python_bin} scripts/integration_v2/run_and_index.py "
        f"--index-root {output_path} -- {infinigen_bin}"
    )


def main() -> int:
    args = parse_args()

    if args.jobs_per_gpu <= 0:
        raise SystemExit("--jobs-per-gpu must be > 0")

    extra_args = list(args.args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    gpu_ids = resolve_gpu_ids(args.gpus)
    slot_gpus = [gpu_id for gpu_id in gpu_ids for _ in range(args.jobs_per_gpu)]

    # Limit semantics:
    #   -1: no limit
    #    0: disable category
    #   >0: use first N entries
    material_limit = int(os.environ.get("MATERIAL_LIMIT", "-1"))
    object_limit = int(os.environ.get("OBJECT_LIMIT", "-1"))
    scene_limit = int(os.environ.get("SCENE_LIMIT", "-1"))
    mask_limit = int(os.environ.get("MASK_LIMIT", "-1"))

    output_path = args.output_path
    slot_count = len(slot_gpus)
    materials_all = list_category("Material", extra_args)
    objects_all = list_category("Object", extra_args)
    scenes_all = list_category("Scene", extra_args)
    masks_all = list_category("Mask", extra_args)

    procs: list[tuple[int, str, subprocess.Popen[str]]] = []
    runner = render_runner(output_path)

    for slot_idx, gpu_id in enumerate(slot_gpus):
        materials = shard_items(materials_all, slot_count, slot_idx, material_limit)
        objects = shard_items(objects_all, slot_count, slot_idx, object_limit)
        scenes = shard_items(scenes_all, slot_count, slot_idx, scene_limit)
        masks = shard_items(masks_all, slot_count, slot_idx, mask_limit)

        print(
            f"slot={slot_idx}/{slot_count - 1} gpu={gpu_id} "
            f"materials={count_items(materials)} objects={count_items(objects)} "
            f"scenes={count_items(scenes)} masks={count_items(masks)}"
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["GPU"] = gpu_id
        env["RENDER_RUNNER"] = runner
        env["MATERIALS"] = materials
        env["OBJECTS"] = objects
        env["SCENES"] = scenes
        env["MASKS"] = masks

        cmd = ["scripts/integration_v2/launch.sh", str(output_path), "1", *extra_args]
        proc = subprocess.Popen(cmd, env=env, text=True)
        procs.append((slot_idx, gpu_id, proc))

    failed_slots: list[tuple[int, str, int]] = []
    for slot_idx, gpu_id, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed_slots.append((slot_idx, gpu_id, rc))

    if failed_slots:
        for slot_idx, gpu_id, rc in failed_slots:
            print(
                f"slot {slot_idx} (gpu {gpu_id}) failed with exit code {rc}",
                file=sys.stderr,
            )

    failed = failed_render_names(output_path)
    if failed:
        raise ValueError(f"{len(failed)} render(s) exited non-zero: {failed}")

    if failed_slots:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
