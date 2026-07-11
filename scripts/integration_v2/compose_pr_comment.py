#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Compose the Integration Render PR comment body: viewer links, which assets
# were rerun vs skipped by the coverage gate (and why), and the pixel-diff
# summary vs the baseline run.

import argparse
import json
from pathlib import Path

MAX_TRIGGER_ROWS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-base", required=True)
    parser.add_argument("--rel-path", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--gating-report", type=Path, default=None)
    parser.add_argument("--pixel-summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def gating_section(report: dict) -> list[str]:
    if not report:
        return []
    if not report.get("enabled"):
        return ["**Gating:** disabled — all assets rendered.", ""]
    if report.get("mode") == "full":
        lines = [f"**Gating:** full render — {report.get('reason', 'unknown')}."]
        triggers = report.get("framework_triggers", [])
        if triggers:
            lines.append("Triggered by: " + ", ".join(f"`{t}`" for t in triggers))
        lines.append("")
        return lines

    categories = report.get("categories", {})
    counts = ", ".join(
        f"{name} {len(cat['kept'])}/{cat['total']}" for name, cat in categories.items()
    )
    lines = [f"**Gating:** rerendered {counts}.", ""]

    rows = []
    for name, cat in categories.items():
        for item, triggers in sorted(cat["kept"].items()):
            rows.append((name, item, triggers))
    if not rows:
        lines += ["No asset's recorded codepath intersects this diff.", ""]
        return lines

    lines += [
        "<details><summary>Why each asset was rerun</summary>",
        "",
        "| category | asset | changed files hit |",
        "|---|---|---|",
    ]
    for name, item, triggers in rows[:MAX_TRIGGER_ROWS]:
        shown = ", ".join(f"`{t}`" for t in triggers[:4])
        if len(triggers) > 4:
            shown += f" +{len(triggers) - 4} more"
        lines.append(f"| {name} | {item} | {shown} |")
    if len(rows) > MAX_TRIGGER_ROWS:
        lines.append(f"\n_…{len(rows) - MAX_TRIGGER_ROWS} more omitted_")
    lines += ["", "</details>", ""]
    return lines


def pixel_section(summary_path: Path | None) -> list[str]:
    if summary_path is None or not summary_path.is_file():
        return []
    text = summary_path.read_text().strip()
    return [
        "<details><summary>Pixel diff vs baseline</summary>",
        "",
        text,
        "",
        "</details>",
        "",
    ]


def main() -> int:
    args = parse_args()
    lines = [
        "Integration renders are ready:",
        f"{args.viewer_base}/?v={args.rel_path}",
        "",
        f"Compare vs {args.baseline}:",
        f"{args.viewer_base}/?v={args.baseline}&v={args.rel_path}",
        "",
        f"Path: {args.target_dir}",
        "",
    ]
    lines += gating_section(load_json(args.gating_report))
    lines += pixel_section(args.pixel_summary)
    args.out.write_text("\n".join(lines).strip() + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
