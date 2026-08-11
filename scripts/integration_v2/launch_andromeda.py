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
        "--changed-only",
        action="store_true",
        default=bool(os.environ.get("INFINIGEN_CHANGED_ONLY")),
        help="Render only assets whose recorded codepath intersects the diff vs --base-ref.",
    )
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("COVERAGE_BASE_REF", ""),
        help="Git ref to diff against when --changed-only is set.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=os.environ.get("ASSET_COVERAGE_BASELINE", "") or None,
        help="asset_coverage.json mapping generator -> covered files (from the base branch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gating report and exit without rendering or touching a GPU.",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args forwarded to infinigen2.list and launch.sh.",
    )
    return parser.parse_args()


# changed files matching these force a full render (coverage cannot vouch for them)
FRAMEWORK_PATTERNS = (
    "pyproject.toml",
    "uv.lock",
    "scripts/integration_v2/",
    ".github/workflows/",
)

MANIFEST_PATH = "src/infinigen2/manifest.json"
PACKAGE_ROOT = "src/infinigen2/"

CATEGORIES = {
    "materials": (["--categories", "Material"], "MATERIAL_LIMIT", "MATERIALS"),
    "objects": (["--categories", "Object"], "OBJECT_LIMIT", "OBJECTS"),
    "scenes": (["--categories", "Scene"], "SCENE_LIMIT", "SCENES"),
    "masks": (["--categories", "Mask"], "MASK_LIMIT", "MASKS"),
    "presets": (["--presets"], "PRESET_LIMIT", "PRESETS"),
    "environments": (
        ["--categories", "Environment"],
        "ENVIRONMENT_LIMIT",
        "ENVIRONMENTS",
    ),
    "cameras": (["--categories", "Cameras"], "CAMERA_LIMIT", "CAMERAS"),
}


def changed_files(base_ref: str) -> set[str] | None:
    cmd = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
    try:
        out = run_capture(cmd)
    except subprocess.CalledProcessError:
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def manifest_entries(text: str) -> dict[str, dict]:
    return {entry["name"]: entry for entry in json.loads(text)}


def manifest_changed_shortnames(base_ref: str) -> set[str] | None:
    try:
        merge_base = run_capture(["git", "merge-base", base_ref, "HEAD"]).strip()
        base = manifest_entries(
            run_capture(["git", "show", f"{merge_base}:{MANIFEST_PATH}"])
        )
        head = manifest_entries(Path(MANIFEST_PATH).read_text())
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None
    changed = {name for name in head if base.get(name) != head[name]}
    return {name.split(".")[-1] for name in changed}


def load_baseline(path: Path | None) -> dict[str, list[str]]:
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def select_changed(
    items: list[str],
    baseline: dict[str, list[str]],
    changed: set[str],
    forced: set[str],
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    kept = []
    triggers = {}
    skipped = []
    for item in items:
        covered = baseline.get(item)
        hits = sorted(changed.intersection(covered)) if covered is not None else []
        if item in forced:
            hits.append("<manifest entry changed>")
        if covered is None:
            kept.append(item)
            triggers[item] = ["<not in baseline coverage>"]
        elif hits:
            kept.append(item)
            triggers[item] = hits
        else:
            skipped.append(item)
    return kept, triggers, skipped


def framework_triggers(changed: set[str]) -> list[str]:
    return sorted(f for f in changed for p in FRAMEWORK_PATTERNS if f.startswith(p))


# coverage records only .py, so a changed data file is invisible to the gate
def opaque_source_triggers(changed: set[str]) -> list[str]:
    return sorted(
        f
        for f in changed
        if f.startswith(PACKAGE_ROOT) and not f.endswith(".py") and f != MANIFEST_PATH
    )


def gate_by_diff(
    args: argparse.Namespace, items_by_category: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict]:
    base_ref = args.base_ref or "HEAD~1"
    changed = changed_files(base_ref)

    def render_all(reason: str, **extra) -> tuple[dict[str, list[str]], dict]:
        print(f"changed-only: {reason}, rendering all assets", file=sys.stderr)
        report = {"enabled": True, "mode": "full", "reason": reason, **extra}
        return items_by_category, report

    # a deleted or unfetched base branch must not stop the whole render
    if changed is None:
        return render_all(f"base ref {base_ref} could not be resolved")

    baseline = load_baseline(args.baseline)
    changed_list = sorted(changed)

    if not baseline:
        return render_all("no baseline coverage available", changed_files=changed_list)

    framework_hits = framework_triggers(changed)
    if framework_hits:
        return render_all(
            "framework file changed",
            changed_files=changed_list,
            framework_triggers=framework_hits,
        )

    opaque_hits = opaque_source_triggers(changed)
    if opaque_hits:
        return render_all(
            "non-python source file changed",
            changed_files=changed_list,
            framework_triggers=opaque_hits,
        )

    forced = set()
    report = {"enabled": True, "mode": "gated", "changed_files": changed_list}
    if MANIFEST_PATH in changed:
        manifest_forced = manifest_changed_shortnames(base_ref)
        if manifest_forced is None:
            return render_all("manifest diff unreadable", changed_files=changed_list)
        forced = manifest_forced
        changed = changed - {MANIFEST_PATH}
        report["manifest_changed"] = sorted(forced)

    report["categories"] = {}
    kept_by_category = {}
    for name, items in items_by_category.items():
        keep, triggers, skipped = select_changed(items, baseline, changed, forced)
        report["categories"][name] = {
            "total": len(items),
            "kept": triggers,
            "skipped": skipped,
        }
        kept_by_category[name] = keep
        print(f"changed-only: {name} {len(keep)}/{len(items)}", file=sys.stderr)
    return kept_by_category, report


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


def list_items(selector: list[str], extra_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "infinigen2.list",
        *selector,
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


def render_events(output_path: Path) -> list[dict]:
    events_dir = output_path / "render_index" / "events"
    if not events_dir.is_dir():
        return []

    events = []
    for event_path in sorted(events_dir.glob("*.json")):
        try:
            payload = json.loads(event_path.read_text())
        except Exception:
            continue
        payload.setdefault("asset_dir", event_path.stem)
        events.append(payload)
    return events


# a render that exits 0 having written no image is a silent failure, not a pass
def failed_render_names(output_path: Path) -> tuple[list[str], list[str]]:
    crashed = []
    empty = []
    for payload in render_events(output_path):
        name = payload.get("asset_dir") or "unknown"
        if payload.get("returncode", 0) != 0:
            crashed.append(name)
        elif not payload.get("images"):
            empty.append(name)
    return crashed, empty


def render_runner(output_path: Path) -> str:
    python_bin = Path(".venv/bin/python")
    infinigen_bin = Path(".venv/bin/infinigen2")
    if not python_bin.exists():
        raise RuntimeError("Expected .venv/bin/python to exist")
    if not infinigen_bin.exists():
        raise RuntimeError("Expected .venv/bin/infinigen2 to exist")

    coverage_prefix = ""
    if os.environ.get("INFINIGEN_COVERAGE"):
        coverage_prefix = (
            f"{python_bin} -m coverage run --parallel-mode --rcfile=pyproject.toml "
        )

    return (
        f"{python_bin} scripts/integration_v2/run_and_index.py "
        f"--index-root {output_path} -- {coverage_prefix}{infinigen_bin}"
    )


def main() -> int:
    args = parse_args()

    if args.jobs_per_gpu <= 0:
        raise SystemExit("--jobs-per-gpu must be > 0")

    extra_args = list(args.args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # Limit semantics:
    #   -1: no limit
    #    0: disable category
    #   >0: use first N entries
    limits = {
        name: int(os.environ.get(limit_env, "-1"))
        for name, (_, limit_env, _) in CATEGORIES.items()
    }

    output_path = args.output_path
    # create the events index up front so the viewer loads even when zero assets render
    (output_path / "render_index" / "events").mkdir(parents=True, exist_ok=True)
    items_all = {
        name: list_items(selector, extra_args)
        for name, (selector, _, _) in CATEGORIES.items()
    }

    gating_report = {"enabled": False}
    if args.changed_only:
        items_all, gating_report = gate_by_diff(args, items_all)
    (output_path / "gating_report.json").write_text(json.dumps(gating_report, indent=2))

    if args.dry_run:
        print(json.dumps(gating_report, indent=2))
        return 0

    # After the gate so --dry-run needs no GPU: this raises without nvidia-smi.
    gpu_ids = resolve_gpu_ids(args.gpus)
    slot_gpus = [gpu_id for gpu_id in gpu_ids for _ in range(args.jobs_per_gpu)]
    slot_count = len(slot_gpus)

    procs: list[tuple[int, str, subprocess.Popen[str]]] = []
    runner = render_runner(output_path)

    for slot_idx, gpu_id in enumerate(slot_gpus):
        shards = {
            name: shard_items(items, slot_count, slot_idx, limits[name])
            for name, items in items_all.items()
        }
        counts = " ".join(f"{n}={count_items(s)}" for n, s in shards.items())
        print(f"slot={slot_idx}/{slot_count - 1} gpu={gpu_id} {counts}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["GPU"] = gpu_id
        env["RENDER_RUNNER"] = runner
        for name, shard in shards.items():
            env[CATEGORIES[name][2]] = shard

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

    crashed, empty = failed_render_names(output_path)
    if crashed:
        raise ValueError(f"{len(crashed)} render(s) exited non-zero: {crashed}")
    if empty:
        raise ValueError(f"{len(empty)} render(s) wrote no images: {empty}")

    if failed_slots:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
