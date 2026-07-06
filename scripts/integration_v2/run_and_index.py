#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Jack Nugent

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided")
    args.cmd = cmd
    return args


def parse_output_dir(cmd: list[str]) -> Path | None:
    for i, token in enumerate(cmd[:-1]):
        if token == "--output":
            return Path(cmd[i + 1])
    return None


def parse_asset_fields(asset_dir: str) -> tuple[str, str, str]:
    name = Path(asset_dir).name
    if "-" not in name:
        return "unknown", name, name

    asset_type, rest = name.split("-", 1)
    if asset_type not in {
        "material",
        "object",
        "scene",
        "mask",
        "preset",
        "environment",
    }:
        return "unknown", name, name

    parts = rest.rsplit("-", 3)
    if len(parts) != 4:
        return asset_type, rest, rest

    generator, obj_name, renderer, variant = parts
    return asset_type, generator, f"{obj_name}-{renderer}-{variant}"


def collect_pngs(asset_output_dir: Path, index_root: Path) -> list[str]:
    if not asset_output_dir.exists():
        return []

    pngs = []
    for p in sorted(asset_output_dir.rglob("*.png")):
        rel_asset = p.relative_to(asset_output_dir)
        if "tmp_" in rel_asset.parts:
            continue
        try:
            rel = p.resolve().relative_to(index_root.resolve())
        except Exception:
            rel = p.resolve()
        pngs.append(rel.as_posix())
    return pngs


def main() -> int:
    args = parse_args()
    index_root = args.index_root.resolve()
    events_dir = index_root / "render_index" / "events"
    logs_dir = index_root / "render_index" / "logs"
    events_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    started = time.time()
    proc = subprocess.run(args.cmd, text=True, capture_output=True)
    ended = time.time()

    output_dir_arg = parse_output_dir(args.cmd)
    asset_output_dir = output_dir_arg.resolve() if output_dir_arg else None
    asset_dir_rel = ""
    if asset_output_dir is not None:
        try:
            asset_dir_rel = asset_output_dir.relative_to(index_root).as_posix()
        except Exception:
            asset_dir_rel = asset_output_dir.as_posix()

    pngs = collect_pngs(asset_output_dir, index_root) if asset_output_dir else []

    status = "success" if pngs else "no_outputs"

    asset_type, generator, variant_key = parse_asset_fields(asset_dir_rel)

    stderr_path = ""
    if proc.stderr:
        stderr_file = logs_dir / f"{event_id}.stderr.txt"
        stderr_file.write_text(proc.stderr)
        stderr_path = stderr_file.relative_to(index_root).as_posix()

    event = {
        "event_id": event_id,
        "timestamp_start": started,
        "timestamp_end": ended,
        "duration_sec": round(ended - started, 3),
        "status": status,
        "returncode": proc.returncode,
        "cmd": args.cmd,
        "asset_dir": asset_dir_rel,
        "asset_type": asset_type,
        "generator": generator,
        "variant_key": variant_key,
        "images": pngs,
        "stderr_path": stderr_path,
    }

    event_path = events_dir / f"{event_id}.json"
    event_path.write_text(json.dumps(event, ensure_ascii=True))

    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    # Do not gate success/failure on process exit code; downstream logic is image-based.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
