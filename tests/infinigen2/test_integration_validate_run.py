# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import validate_run  # noqa: E402


def test_validation_accepts_successful_render_with_coverage():
    events = [
        {
            "asset_dir": "camera-chair-demo",
            "generator": "chair_rand",
            "returncode": 0,
            "images": ["camera-chair-demo/0000.png"],
        }
    ]

    assert validate_run.validation_errors(events, {"chair_rand": ["src/a.py"]}) == []


def test_validation_rejects_missing_output_and_coverage():
    events = [
        {
            "asset_dir": "camera-chair-demo",
            "generator": "chair_rand",
            "returncode": 0,
            "images": [],
        }
    ]

    assert validate_run.validation_errors(events, {}) == [
        "camera-chair-demo: render wrote no images"
    ]


def test_validation_rejects_rendered_generator_without_coverage():
    events = [
        {
            "asset_dir": "camera-chair-demo",
            "generator": "chair_rand",
            "returncode": 0,
            "images": ["camera-chair-demo/0000.png"],
        }
    ]

    assert validate_run.validation_errors(events, {}) == [
        "chair_rand: no executed source recorded in coverage"
    ]


def test_validation_requires_a_gating_report(tmp_path):
    assert validate_run.output_errors(tmp_path) == [
        "gating_report.json was not written"
    ]

    (tmp_path / "gating_report.json").write_text("[]")
    assert validate_run.output_errors(tmp_path) == [
        "gating_report.json is not an object"
    ]
