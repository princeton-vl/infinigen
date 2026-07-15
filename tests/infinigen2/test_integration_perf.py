# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import sys
from pathlib import Path

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import perf_diff  # noqa: E402
from collection import collect_images_structured  # noqa: E402
from display import build_comparison_data, build_version_totals  # noqa: E402


def _write_run(root: Path, name: str, tris: int, cpu: float, gpu: float):
    events = root / name / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)
    for variant in ("0", "1"):
        img = f"{name}/object-chair-obj-cycles-{variant}/camera-0/0001.png"
        (root / name / img).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2)).save(root / name / img)
        event = {
            "generator": "chair_rand",
            "asset_type": "object",
            "variant_key": f"obj-cycles-{variant}",
            "status": "success",
            "cmd": ["infinigen", "chair_rand"],
            "images": [img],
            "duration_sec": cpu + gpu,
            "tris": tris,
            "cpu_time_sec": cpu,
            "gpu_time_sec": gpu,
        }
        (events / f"{variant}.json").write_text(json.dumps(event))
    return root / name


def test_perf_gate_flags_tris_regression(tmp_path):
    base = _write_run(tmp_path, "base", tris=48000, cpu=12.0, gpu=4.0)
    pr = _write_run(tmp_path, "pr", tris=57600, cpu=12.0, gpu=4.0)  # +20% tris
    report = perf_diff.compare(pr, base, threshold=0.05)
    assert report["fail_count"] == 2
    assert all(r["status"] == "fail" for r in report["results"])
    assert perf_diff.asset_verdicts(report)["chair_rand"] == "changed"


def test_perf_gate_no_regression_when_within_threshold(tmp_path):
    base = _write_run(tmp_path, "base", tris=48000, cpu=12.0, gpu=4.0)
    pr = _write_run(tmp_path, "pr", tris=49000, cpu=12.4, gpu=4.1)  # <5% everywhere
    report = perf_diff.compare(pr, base, threshold=0.05)
    assert report["fail_count"] == 0


def test_perf_gate_approval_downgrades_fail(tmp_path):
    base = _write_run(tmp_path, "base", tris=48000, cpu=12.0, gpu=4.0)
    pr = _write_run(tmp_path, "pr", tris=57600, cpu=12.0, gpu=4.0)
    rec = perf_diff.make_approval(tmp_path, "chair_rand", pr, "base", "pr", "tester")
    perf_diff.append_approval(tmp_path, rec)
    report = perf_diff.compare(pr, base, threshold=0.05, archive_root=tmp_path)
    assert report["fail_count"] == 0
    assert report["approved_count"] == 2


def test_display_marks_after_worse_and_better(tmp_path):
    _write_run(tmp_path, "base", tris=48000, cpu=12.0, gpu=4.0)
    _write_run(tmp_path, "pr", tris=57600, cpu=10.0, gpu=4.0)  # tris worse, cpu better
    names = ["base", "pr"]
    results = [
        collect_images_structured(tmp_path / "base", "base"),
        collect_images_structured(tmp_path / "pr", "pr"),
    ]
    rows = build_comparison_data(results, names)
    after = next(o for o in rows[0]["objects"] if o["version"] == "pr")
    before = next(o for o in rows[0]["objects"] if o["version"] == "base")
    assert after["metrics"]["tris_cls"] == "worse"
    assert after["metrics"]["cpu_cls"] == "better"
    assert before["metrics"]["tris_cls"] == "same"

    totals = build_version_totals(rows, names)
    assert totals["pr"]["tris_cls"] == "worse"
    assert totals["base"]["tris_cls"] == "same"
