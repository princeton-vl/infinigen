# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import run_paths  # noqa: E402


def test_develop_paths_use_the_production_archive(tmp_path):
    paths = run_paths.resolve_paths(tmp_path, "run", ref_name="develop")

    assert paths == {
        "REL_PATH": "run",
        "ARCHIVE_ROOT": str(tmp_path),
        "TARGET_DIR": str(tmp_path / "run"),
        "BASELINE": "develop_latest",
        "BASELINE_COVERAGE": str(tmp_path / "develop_latest" / "asset_coverage.json"),
    }


def test_staging_push_uses_an_isolated_archive_and_baseline(tmp_path):
    paths = run_paths.resolve_paths(tmp_path, "run", ref_name="integration-staging")

    assert paths["REL_PATH"] == "staging/run"
    assert paths["ARCHIVE_ROOT"] == str(tmp_path / "staging")
    assert paths["TARGET_DIR"] == str(tmp_path / "staging" / "run")
    assert paths["BASELINE"] == "staging/integration-staging_latest"


def test_validation_dispatch_is_isolated_without_a_staging_ref(tmp_path):
    paths = run_paths.resolve_paths(tmp_path, "run", validation="true")

    assert paths["ARCHIVE_ROOT"] == str(tmp_path / "staging")
    assert paths["BASELINE"] == "staging/integration-staging_latest"


def test_pr_uses_its_target_branch_baseline_when_present(tmp_path):
    (tmp_path / "staging" / "integration-staging_latest").mkdir(parents=True)
    paths = run_paths.resolve_paths(tmp_path, "run", base_ref="integration-staging")

    assert paths["BASELINE"] == "staging/integration-staging_latest"
