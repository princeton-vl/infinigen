# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import baseline_diff  # noqa: E402
import render_trajectory_video  # noqa: E402


def _write_run(root: Path, name: str, color: tuple[int, int, int]) -> Path:
    events = root / name / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)

    still = "camera-traj0/Camera/0000.png"
    video = "camera-traj0/image_Camera.mp4"
    (root / name / still).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(root / name / still)
    (root / name / video).write_bytes(b"not an image")

    event = {
        "generator": "monocular_camera_in_bbox_rand",
        "variant_key": "workbench-traj0",
        "status": "success",
        "images": [still, video],
    }
    (events / "traj0.json").write_text(json.dumps(event))
    return root / name


def test_pixel_diff_skips_videos(tmp_path):
    base = _write_run(tmp_path, "base", (0, 0, 0))
    pr = _write_run(tmp_path, "pr", (0, 0, 0))
    report = baseline_diff.compare_pixel(pr, base)
    assert report["total"] == 1
    assert report["fail_count"] == 0
    assert report["results"][0]["image"].endswith(".png")


def test_pixel_diff_flags_changed_still(tmp_path):
    base = _write_run(tmp_path, "base", (0, 0, 0))
    pr = _write_run(tmp_path, "pr", (255, 255, 255))
    report = baseline_diff.compare_pixel(pr, base)
    assert report["fail_count"] == 1


def test_camera_render_uses_coverage_when_enabled(tmp_path, monkeypatch):
    args = render_trajectory_video.argparse.Namespace(
        output=tmp_path,
        scene="livingroom_rand",
        camera="orbit_rand",
        seed=0,
        frames=[0, 47],
        resolution=[640, 360],
    )
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("INFINIGEN_COVERAGE", "1")
    monkeypatch.setattr(render_trajectory_video.subprocess, "run", fake_run)

    assert render_trajectory_video.render_frames(args) == 0
    assert captured["cmd"][:8] == [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        "--rcfile=pyproject.toml",
        "-m",
        "infinigen2",
    ]
