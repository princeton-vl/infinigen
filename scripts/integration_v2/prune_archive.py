#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Enforce a retention policy on the integration render archive. Every PR push
# archives a full render set (~900MB) and nothing ever removed them, so the
# archive grew unbounded until the runner's disk filled and CI stopped.
#
# A run is kept when it is the target of a *_latest baseline symlink, when it is
# one of the --keep-newest most recent, or when it is younger than --keep-days.
# Everything else is deletable. Deletion requires --apply; the default is a
# dry run that only reports.

import argparse
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--keep-days", type=float, default=21.0)
    parser.add_argument("--keep-newest", type=int, default=50)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; without it this only reports what would go.",
    )
    parser.add_argument(
        "--now",
        type=float,
        default=None,
        help="unix timestamp to age against (defaults to wall clock).",
    )
    return parser.parse_args()


def baseline_targets(archive_root: Path) -> set[Path]:
    """Resolved targets of every *_latest symlink. These are what open PRs diff
    against, so they are never deletable however old they get."""
    targets = set()
    for entry in archive_root.iterdir():
        if not entry.is_symlink():
            continue
        try:
            targets.add(entry.resolve(strict=True))
        except OSError:
            continue
    return targets


def run_dirs(archive_root: Path) -> list[Path]:
    return [
        entry
        for entry in archive_root.iterdir()
        if entry.is_dir() and not entry.is_symlink() and not entry.name.startswith(".")
    ]


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def partition(
    archive_root: Path, keep_days: float, keep_newest: int, now: float
) -> tuple[list[Path], list[Path]]:
    """Split the archive into (kept, deletable), newest first."""
    protected = baseline_targets(archive_root)
    candidates = sorted(run_dirs(archive_root), key=lambda p: p.stat().st_mtime)
    candidates.reverse()

    cutoff = now - keep_days * 86400
    kept = []
    deletable = []
    for index, path in enumerate(candidates):
        recent = index < keep_newest or path.stat().st_mtime >= cutoff
        if path.resolve() in protected or recent:
            kept.append(path)
        else:
            deletable.append(path)
    return kept, deletable


def main() -> int:
    args = parse_args()
    if not args.archive_root.is_dir():
        raise SystemExit(f"No archive at {args.archive_root}")

    now = args.now if args.now is not None else time.time()
    kept, deletable = partition(
        args.archive_root, args.keep_days, args.keep_newest, now
    )

    if not deletable:
        print(f"prune: {len(kept)} runs kept, nothing to remove")
        return 0

    freed = 0
    removed = 0
    for path in deletable:
        size = dir_size_bytes(path)
        freed += size
        verb = "removing" if args.apply else "would remove"
        print(f"prune: {verb} {path.name} ({size / 1e9:.2f} GB)", file=sys.stderr)
        if args.apply:
            shutil.rmtree(path)
            removed += 1

    verb = "freed" if args.apply else "would free"
    print(
        f"prune: {len(kept)} runs kept, {removed if args.apply else len(deletable)} removed, "
        f"{verb} {freed / 1e9:.1f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
