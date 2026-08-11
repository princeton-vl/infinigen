# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import baseline_diff  # noqa: E402
import compose_pr_comment  # noqa: E402
import launch_andromeda  # noqa: E402
import prune_archive  # noqa: E402

DAY = 86400.0


def _run_dir(root: Path, name: str, mtime: float) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "marker.txt").write_text(name)
    os.utime(path, (mtime, mtime))
    return path


def test_prune_keeps_recent_and_drops_old(tmp_path):
    now = 1_000 * DAY
    for index in range(6):
        _run_dir(tmp_path, f"run{index}", now - (index + 1) * 10 * DAY)

    kept, deletable = prune_archive.partition(tmp_path, 21.0, 2, now)

    assert [p.name for p in kept] == ["run0", "run1"]
    assert sorted(p.name for p in deletable) == ["run2", "run3", "run4", "run5"]


def test_prune_never_drops_a_baseline_target(tmp_path):
    now = 1_000 * DAY
    old = _run_dir(tmp_path, "ancient", now - 400 * DAY)
    _run_dir(tmp_path, "recent", now - DAY)
    (tmp_path / "develop_latest").symlink_to(old)

    kept, deletable = prune_archive.partition(tmp_path, 21.0, 1, now)

    assert old in kept
    assert deletable == []


def test_prune_dry_run_leaves_everything(tmp_path):
    now = 1_000 * DAY
    _run_dir(tmp_path, "old", now - 400 * DAY)
    _run_dir(tmp_path, "new", now - DAY)

    argv = [str(tmp_path), "--keep-newest", "1", "--now", str(now)]
    rc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "prune_archive.py"), *argv],
        capture_output=True,
        text=True,
    )

    assert rc.returncode == 0
    assert (tmp_path / "old").is_dir()
    assert "would remove old" in rc.stderr


def _gate_args(baseline: Path, base_ref: str = "origin/develop"):
    return launch_andromeda.argparse.Namespace(
        base_ref=base_ref, baseline=baseline, changed_only=True
    )


def _write_baseline(tmp_path: Path, mapping: dict) -> Path:
    path = tmp_path / "asset_coverage.json"
    path.write_text(json.dumps(mapping))
    return path


ALL_CATEGORIES = ("materials", "objects", "scenes", "masks")
EXTRA_CATEGORIES = ("presets", "environments", "cameras")


def test_gate_covers_every_category(tmp_path, monkeypatch):
    baseline = _write_baseline(
        tmp_path,
        {
            "chair_rand": ["src/infinigen2/objects/chair.py"],
            "chair_preset": ["src/infinigen2/objects/chair.py"],
            "orbit_90": ["src/infinigen2/cameras/orbit.py"],
            "sky_rand": ["src/infinigen2/environments/sky.py"],
        },
    )
    monkeypatch.setattr(
        launch_andromeda,
        "changed_files",
        lambda ref: {"src/infinigen2/objects/chair.py"},
    )

    items = {name: [] for name in ALL_CATEGORIES}
    items["objects"] = ["chair_rand"]
    items["presets"] = ["chair_preset"]
    items["environments"] = ["sky_rand"]
    items["cameras"] = ["orbit_90"]

    kept, report = launch_andromeda.gate_by_diff(_gate_args(baseline), items)

    assert report["mode"] == "gated"
    assert set(report["categories"]) == set(ALL_CATEGORIES) | set(EXTRA_CATEGORIES)
    assert kept["objects"] == ["chair_rand"]
    assert kept["presets"] == ["chair_preset"]
    # an unrelated diff must not drag in the expensive camera/environment renders
    assert kept["cameras"] == []
    assert kept["environments"] == []


def test_category_config_has_one_entry_per_shard_and_limit():
    assert set(launch_andromeda.CATEGORIES) == {
        "materials",
        "objects",
        "scenes",
        "masks",
        "presets",
        "environments",
        "cameras",
    }
    for selector, limit_env, shard_env in launch_andromeda.CATEGORIES.values():
        assert selector
        assert limit_env.endswith("_LIMIT")
        assert shard_env.isupper()


def test_gate_forces_full_render_on_non_python_change(tmp_path, monkeypatch):
    baseline = _write_baseline(tmp_path, {"chair_rand": ["src/infinigen2/x.py"]})
    monkeypatch.setattr(
        launch_andromeda,
        "changed_files",
        lambda ref: {"src/infinigen2/assets/lookup.csv"},
    )

    items = {"objects": ["chair_rand"]}
    kept, report = launch_andromeda.gate_by_diff(_gate_args(baseline), items)

    assert report["mode"] == "full"
    assert report["reason"] == "non-python source file changed"
    assert kept["objects"] == ["chair_rand"]


def test_gate_ignores_manifest_in_opaque_check(tmp_path, monkeypatch):
    baseline = _write_baseline(tmp_path, {"chair_rand": ["src/infinigen2/x.py"]})
    monkeypatch.setattr(
        launch_andromeda, "changed_files", lambda ref: {launch_andromeda.MANIFEST_PATH}
    )
    monkeypatch.setattr(
        launch_andromeda, "manifest_changed_shortnames", lambda ref: {"chair_rand"}
    )

    _, report = launch_andromeda.gate_by_diff(
        _gate_args(baseline), {"objects": ["chair_rand"]}
    )

    assert report["mode"] == "gated"


def _write_event(events: Path, name: str, returncode: int, images: list[str]) -> None:
    events.mkdir(parents=True, exist_ok=True)
    payload = {"asset_dir": name, "returncode": returncode, "images": images}
    (events / f"{name}.json").write_text(json.dumps(payload))


def test_failed_renders_split_crashes_from_silent_empties(tmp_path):
    events = tmp_path / "render_index" / "events"
    _write_event(events, "good", 0, ["good/Image.png"])
    _write_event(events, "boom", 1, ["boom/Image.png"])
    _write_event(events, "silent", 0, [])

    crashed, empty = launch_andromeda.failed_render_names(tmp_path)

    assert crashed == ["boom"]
    assert empty == ["silent"]


def _pixel_run(root: Path, name: str, size: tuple[int, int], colour: str) -> Path:
    events = root / name / "render_index" / "events"
    events.mkdir(parents=True)
    rel = "object-chair-obj-cycles-0/camera-0/0001.png"
    (root / name / rel).parent.mkdir(parents=True)
    Image.new("RGB", size, colour).save(root / name / rel)
    (events / "0.json").write_text(
        json.dumps({"generator": "chair_rand", "variant_key": "v0", "images": [rel]})
    )
    return root / name


def test_pixel_shape_mismatch_is_a_failure_and_stays_valid_json(tmp_path):
    base = _pixel_run(tmp_path, "base", (4, 4), "black")
    pr = _pixel_run(tmp_path, "pr", (8, 8), "black")

    report = baseline_diff.compare_pixel(pr, base)

    assert report["fail_count"] == 1
    assert report["results"][0]["status"] == "shape_mismatch"
    assert report["results"][0]["mse"] is None
    json.dumps(report, allow_nan=False)


def test_pixel_identical_images_pass(tmp_path):
    base = _pixel_run(tmp_path, "base", (4, 4), "black")
    pr = _pixel_run(tmp_path, "pr", (4, 4), "black")

    assert baseline_diff.compare_pixel(pr, base)["fail_count"] == 0


def test_eventless_baseline_is_skipped_not_reported_green(tmp_path):
    baseline = tmp_path / "baseline"
    (baseline / "render_index" / "events").mkdir(parents=True)
    pr = _pixel_run(tmp_path, "pr", (4, 4), "black")

    args = baseline_diff.argparse.Namespace(
        pr=pr,
        baseline=baseline,
        report=tmp_path / "report.json",
        summary=tmp_path / "summary.md",
    )

    assert baseline_diff._pixel_main(args) == 0
    assert json.loads(args.report.read_text()) == {"skipped": True}


@pytest.mark.parametrize("code,expected", [(0, "success"), (3, "failed")])
def test_run_and_index_status_follows_returncode(tmp_path, code, expected):
    out = tmp_path / "object-chair-demo-cycles-0"
    out.mkdir()
    Image.new("RGB", (2, 2)).save(out / "Image.png")

    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "run_and_index.py"),
            "--index-root",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            f"import sys; sys.exit({code})",
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    events = list((tmp_path / "render_index" / "events").glob("*.json"))
    assert len(events) == 1
    assert json.loads(events[0].read_text())["status"] == expected


def _comment_args(run_id: str) -> argparse.Namespace:
    optional = ["gating_report", "render_index", "pixel_summary", "perf_summary"]
    fields = dict.fromkeys(optional + ["commit", "run_url"])
    fields.update(
        run_id=run_id, viewer_base="", rel_path="", baseline="", target_dir=""
    )
    return argparse.Namespace(**fields)


def test_only_a_pending_comment_carries_the_reap_marker():
    planned = "\n".join(compose_pr_comment.planned_body(_comment_args("1")))
    done = "\n".join(compose_pr_comment.done_body(_comment_args("1")))

    assert compose_pr_comment.PENDING_MARKER in planned
    assert compose_pr_comment.PENDING_MARKER not in done


def test_reaping_spares_this_run_and_older_results():
    mine = "\n".join(compose_pr_comment.planned_body(_comment_args("2")))
    older_pending = "\n".join(compose_pr_comment.planned_body(_comment_args("1")))
    older_done = "\n".join(compose_pr_comment.done_body(_comment_args("1")))

    marker = compose_pr_comment.status_marker("2")
    pending = compose_pr_comment.PENDING_MARKER
    reaped = [
        body
        for body in [mine, older_pending, older_done]
        if pending in body and marker not in body
    ]

    assert reaped == [older_pending]
