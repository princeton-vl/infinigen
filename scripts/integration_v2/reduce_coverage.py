#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Reduce the per-asset coverage data written by run_and_index.py into a
# {generator: [source_file, ...]} map plus one combined coverage.xml. The map
# lets the integration render gate skip any asset whose recorded codepath does
# not intersect a PR's changed files.

import argparse
import json
from pathlib import Path

import coverage
from coverage import CoverageData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=Path, help="coverage_data dir of per-asset subdirs"
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-xml", type=Path, default=None)
    parser.add_argument("--rcfile", type=Path, default=Path("pyproject.toml"))
    return parser.parse_args()


def asset_files(data_files: list[Path]) -> list[str]:
    merged = CoverageData()
    for data_file in data_files:
        single = CoverageData(basename=str(data_file))
        try:
            single.read()
        except Exception:
            continue
        merged.update(single)
    # source=infinigen2 records every package .py at 0%; keep only files with real hits
    return sorted(f for f in merged.measured_files() if merged.lines(f))


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"No coverage data dir at {args.data_dir}")

    per_asset: dict[str, list[str]] = {}
    all_files: list[str] = []
    for generator_dir in sorted(args.data_dir.iterdir()):
        if not generator_dir.is_dir():
            continue
        data_files = sorted(generator_dir.glob("data.*"))
        if not data_files:
            continue
        per_asset[generator_dir.name] = asset_files(data_files)
        all_files.extend(str(p) for p in data_files)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(per_asset, indent=2, sort_keys=True))
    print(f"Wrote {args.out_json} ({len(per_asset)} generators)")

    if args.out_xml and all_files:
        combined = args.data_dir / ".coverage.combined"
        cov = coverage.Coverage(data_file=str(combined), config_file=str(args.rcfile))
        cov.combine(data_paths=all_files, keep=True)
        cov.xml_report(outfile=str(args.out_xml), ignore_errors=True)
        print(f"Wrote {args.out_xml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
