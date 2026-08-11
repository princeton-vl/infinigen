#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

STAGING_BRANCH = "integration-staging"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--ref-name", default="")
    parser.add_argument("--validation", default="")
    return parser.parse_args()


def is_staging(base_ref: str, ref_name: str, validation: str) -> bool:
    return (
        base_ref == STAGING_BRANCH
        or ref_name == STAGING_BRANCH
        or validation.lower() == "true"
    )


def resolve_paths(
    archive_root: Path,
    run_path: str,
    base_ref: str = "",
    ref_name: str = "",
    validation: str = "",
) -> dict[str, str]:
    staging = is_staging(base_ref, ref_name, validation)
    archive_dir = archive_root / "staging" if staging else archive_root
    prefix = "staging/" if staging else ""
    baseline_name = "integration-staging_latest" if staging else "develop_latest"
    if base_ref and (archive_dir / f"{base_ref}_latest").exists():
        baseline_name = f"{base_ref}_latest"
    return {
        "REL_PATH": f"{prefix}{run_path}",
        "ARCHIVE_ROOT": str(archive_dir),
        "TARGET_DIR": str(archive_dir / run_path),
        "BASELINE": f"{prefix}{baseline_name}",
        "BASELINE_COVERAGE": str(archive_dir / baseline_name / "asset_coverage.json"),
    }


def main() -> int:
    args = parse_args()
    for key, value in resolve_paths(
        args.archive_root,
        args.run_path,
        args.base_ref,
        args.ref_name,
        args.validation,
    ).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
