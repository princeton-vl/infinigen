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
MAX_CRASH_ROWS = 40
MAX_ERROR_CHARS = 160

# Lets the workflow update its own comment. Changing it orphans posted ones.
STATUS_MARKER = "<!-- integration-render-status -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["planned", "done"], default="done")
    parser.add_argument("--viewer-base", default="")
    parser.add_argument("--rel-path", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--target-dir", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--gating-report", type=Path, default=None)
    parser.add_argument("--pixel-summary", type=Path, default=None)
    parser.add_argument("--perf-summary", type=Path, default=None)
    parser.add_argument("--render-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "done":
        missing = [
            f"--{name.replace('_', '-')}"
            for name in ("viewer_base", "rel_path", "baseline", "target_dir")
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"--phase done requires: {', '.join(missing)}")
    return args


def load_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def gating_section(report: dict, planned: bool = False) -> list[str]:
    if not report:
        return []
    if not report.get("enabled"):
        tail = "all assets will render." if planned else "all assets rendered."
        return [f"**Gating:** disabled — {tail}", ""]
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
    verb = "will rerender" if planned else "rerendered"
    lines = [f"**Gating:** {verb} {counts}.", ""]

    rows = []
    for name, cat in categories.items():
        for item, triggers in sorted(cat["kept"].items()):
            rows.append((name, item, triggers))
    if not rows:
        lines += ["No asset's recorded codepath intersects this diff.", ""]
        return lines

    lines += [
        f"<details><summary>Why each asset {'is being' if planned else 'was'} rerun</summary>",
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


def last_error_line(index_root: Path, stderr_rel: str) -> str:
    if not stderr_rel:
        return ""
    stderr_file = index_root / stderr_rel
    if not stderr_file.is_file():
        return ""
    text = stderr_file.read_text(errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[-1]
    if len(line) > MAX_ERROR_CHARS:
        line = line[: MAX_ERROR_CHARS - 1] + "…"
    return line.replace("|", "\\|")


def crash_rows(index_root: Path | None) -> list[tuple[str, str]]:
    if index_root is None:
        return []
    events_dir = index_root / "render_index" / "events"
    if not events_dir.is_dir():
        return []

    rows = []
    for event_path in sorted(events_dir.glob("*.json")):
        try:
            payload = json.loads(event_path.read_text())
        except Exception:
            continue
        if payload.get("returncode", 0) == 0:
            continue
        asset = payload.get("asset_dir") or event_path.stem
        error = last_error_line(index_root, payload.get("stderr_path", ""))
        rows.append((asset, error))
    return sorted(rows)


def crash_section(index_root: Path | None) -> list[str]:
    rows = crash_rows(index_root)
    if not rows:
        return []

    lines = [
        f"<details><summary>Crashes: {len(rows)} assets had errors</summary>",
        "",
        "| asset | error |",
        "|---|---|",
    ]
    for asset, error in rows[:MAX_CRASH_ROWS]:
        lines.append(f"| {asset} | {error} |")
    if len(rows) > MAX_CRASH_ROWS:
        lines.append(f"\n_…{len(rows) - MAX_CRASH_ROWS} more omitted_")
    lines += ["", "</details>", ""]
    return lines


def _collapsible_section(summary_path: Path | None, title: str) -> list[str]:
    if summary_path is None or not summary_path.is_file():
        return []
    text = summary_path.read_text().strip()
    return [
        f"<details><summary>{title}</summary>",
        "",
        text,
        "",
        "</details>",
        "",
    ]


def provenance(args: argparse.Namespace) -> list[str]:
    bits = []
    if args.commit:
        bits.append(f"commit `{args.commit[:7]}`")
    if args.run_url:
        bits.append(f"[run log]({args.run_url})")
    return [" · ".join(bits), ""] if bits else []


def planned_body(args: argparse.Namespace) -> list[str]:
    lines = [
        STATUS_MARKER,
        "### Integration render — rendering now",
        "",
    ]
    lines += gating_section(load_json(args.gating_report), planned=True)
    lines += provenance(args)
    lines += ["_Results will replace this comment when the render finishes._", ""]
    return lines


def done_body(args: argparse.Namespace) -> list[str]:
    lines = [
        STATUS_MARKER,
        "### Integration render — done",
        "",
        "Integration renders are ready:",
        f"{args.viewer_base}/?v={args.rel_path}",
        "",
        f"Compare vs {args.baseline}:",
        f"{args.viewer_base}/?v={args.baseline}&v={args.rel_path}",
        "",
        f"Path: {args.target_dir}",
        "",
    ]
    lines += crash_section(args.render_index)
    lines += gating_section(load_json(args.gating_report))
    lines += _collapsible_section(args.pixel_summary, "Pixel diff vs baseline")
    lines += _collapsible_section(args.perf_summary, "Perf diff vs baseline")
    lines += provenance(args)
    return lines


def main() -> int:
    args = parse_args()
    lines = planned_body(args) if args.phase == "planned" else done_body(args)
    args.out.write_text("\n".join(lines).strip() + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
