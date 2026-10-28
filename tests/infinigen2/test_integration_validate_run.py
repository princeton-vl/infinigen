# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
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


def _write_backfacing_report(tmp_path, bad_pixels, generator="chair_rand"):
    camera_dir = tmp_path / f"object-{generator}-demo-cycles-0" / "Camera"
    camera_dir.mkdir(parents=True)
    frame = {
        "frame": 0,
        "bad_pixels": bad_pixels,
        "hit_pixels": 1000,
        "bad_by_object": {"chair": bad_pixels} if bad_pixels else {},
        "image": str(camera_dir / "backfacing_0000.png") if bad_pixels else None,
    }
    (camera_dir / "backfacing.json").write_text(json.dumps({"frames": [frame]}))


def test_backfacing_report_fails_validation(tmp_path):
    _write_backfacing_report(tmp_path, 50)
    errors = validate_run.backfacing_errors(tmp_path)
    assert len(errors) == 1
    assert errors[0].startswith("object-chair_rand-demo-cycles-0: 5.0%")


def test_clean_backfacing_report_passes_validation(tmp_path):
    _write_backfacing_report(tmp_path, 0)
    assert validate_run.backfacing_errors(tmp_path) == []


def test_single_sided_generator_is_excused(tmp_path):
    _write_backfacing_report(tmp_path, 50, generator="flower_rand")
    assert validate_run.backfacing_errors(tmp_path) == []
