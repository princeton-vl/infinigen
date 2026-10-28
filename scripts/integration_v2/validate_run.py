#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

BACKFACING_MAX_FRACTION = 0.001
BACKFACING_EXCUSED = {
    "curtain_rand",
    "flower_rand",
    "leaf_ginkgo_rand",
    "leaf_maple_rand",
    "leaf_simple_rand",
    "obj_with_postprocessing",
    "plant_phyllo_ascending_rand",
    "plant_phyllo_basal_rand",
    "plant_pot_large_rand",
    "plant_pot_small_rand",
    "primitive_with_effect_rand",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the artifacts produced by an integration-render run."
    )
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--coverage", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def render_events(output_path: Path) -> list[dict]:
    events_dir = output_path / "render_index" / "events"
    if not events_dir.is_dir():
        return []
    return [load_json(path, {}) for path in sorted(events_dir.glob("*.json"))]


def validation_errors(events: list[dict], coverage: dict[str, list[str]]) -> list[str]:
    errors = []
    for event in events:
        name = event.get("asset_dir") or event.get("generator") or "unknown"
        if event.get("returncode", 0) != 0:
            errors.append(f"{name}: render command failed")
        elif not event.get("images"):
            errors.append(f"{name}: render wrote no images")

    generators = {
        event["generator"]
        for event in events
        if event.get("generator") and event.get("images")
    }
    for generator in sorted(generators):
        if not coverage.get(generator):
            errors.append(f"{generator}: no executed source recorded in coverage")
    return errors


def _backfacing_frame_error(asset_dir: str, frame: dict) -> str | None:
    fraction = frame["bad_pixels"] / max(frame["hit_pixels"], 1)
    if fraction <= BACKFACING_MAX_FRACTION:
        return None
    image = Path(frame["image"]).name if frame.get("image") else ""
    objects = frame["bad_by_object"]
    return f"{asset_dir}: {fraction:.1%} of pixels see a backface {objects} {image}"


def backfacing_errors(output_path: Path) -> list[str]:
    errors = []
    for report in sorted(output_path.glob("*/*/backfacing.json")):
        asset_dir = report.parent.parent.relative_to(output_path).as_posix()
        if asset_dir.split("-")[1] in BACKFACING_EXCUSED:
            continue
        for frame in load_json(report, {}).get("frames", []):
            errors.append(_backfacing_frame_error(asset_dir, frame))
    return [e for e in errors if e is not None]


def output_errors(output_path: Path) -> list[str]:
    gating_report = output_path / "gating_report.json"
    if not gating_report.is_file():
        return ["gating_report.json was not written"]
    if not isinstance(load_json(gating_report, None), dict):
        return ["gating_report.json is not an object"]
    return []


def main() -> int:
    args = parse_args()
    events = render_events(args.output_path)
    coverage = load_json(args.coverage, {})
    errors = output_errors(args.output_path)
    errors.extend(validation_errors(events, coverage))
    errors.extend(backfacing_errors(args.output_path))
    if errors:
        raise SystemExit("Invalid integration-render artifacts:\n" + "\n".join(errors))
    print(f"Validated {len(events)} render events and {len(coverage)} coverage entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
