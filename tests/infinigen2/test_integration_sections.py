# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import math
import sys
from pathlib import Path

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import display  # noqa: E402
from display import (  # noqa: E402
    build_comparison_data,
    build_section_controls,
    collect_images_structured,
)


def _event(root: Path, name: str, idx: int, gen: str, atype: str, variant: str, images):
    for image in images:
        (root / image).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2)).save(root / image)
    events = root / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": gen,
        "asset_type": atype,
        "variant_key": variant,
        "status": "success",
        "cmd": ["infinigen", gen],
        "images": images,
        "base_tris": 4,
        "subdiv_tris": 10,
        "cpu_time_sec": 1.0,
        "gpu_time_sec": 2.0,
    }
    (events / f"{idx}.json").write_text(json.dumps(payload))


def _seed_images(base: str, seed: int) -> list[str]:
    return [
        f"{base}-{seed}/Camera/0000.png",
        f"{base}-{seed}/Camera/surface-normal_0000.png",
    ]


def _object_run(root: Path, name: str, seeds: int = 3) -> Path:
    version = root / name
    base = f"{name}/object-chair_rand-demo-cycles"
    for seed in range(seeds):
        variant = f"demo-cycles-{seed}"
        images = _seed_images(base, seed)
        _event(version, name, seed, "chair_rand", "object", variant, images)
    return version


def _sections_of(rows: list, asset: str) -> dict:
    row = next(r for r in rows if r["asset"] == asset)
    return {s["key"]: s for s in row["objects"][-1]["sections"]}


def _build(paths: list):
    names = [p.name for p in paths]
    results = [collect_images_structured(p, p.name) for p in paths]
    return build_comparison_data(results, names)


def test_normals_form_a_row_mirroring_every_seed(tmp_path):
    """Every seed renders normals, so the Normals row lines up 1:1 under Random seeds."""
    rows = _build(
        [_object_run(tmp_path, "before"), _object_run(tmp_path, "after", seeds=3)]
    )
    row = next(row for row in rows if row["asset"] == "chair_rand")
    sections = _sections_of(rows, "chair_rand")

    assert set(sections) == {"seeds", "normals"}
    assert [i["label"] for i in sections["seeds"]["images"]] == [
        "cycles / 0 / image",
        "cycles / 1 / image",
        "cycles / 2 / image",
    ]
    assert [i["label"] for i in sections["normals"]["images"]] == [
        "cycles / 0 / surface-normal",
        "cycles / 1 / surface-normal",
        "cycles / 2 / surface-normal",
    ]
    assert sections["normals"]["folded"] is True
    assert sections["seeds"]["folded"] is False
    assert sections["seeds"]["scroll_group"] == "visuals"
    assert sections["normals"]["scroll_group"] == "visuals"
    assert row["available_versions"] == ["before", "after"]


def test_normals_section_carries_no_metrics(tmp_path):
    """Normals share a render with the rgb pass beside them, so counting their
    stats again would double-count the seed's tris/cpu/gpu."""
    rows = _build([_object_run(tmp_path, "before"), _object_run(tmp_path, "after")])
    sections = _sections_of(rows, "chair_rand")

    assert sections["normals"]["metrics"] is None
    assert sections["seeds"]["metrics"]["base_tris"] == "4"
    assert sections["seeds"]["metrics"]["subdiv_tris"] == "10"


def _material_run(root: Path, name: str) -> Path:
    version = root / name
    base = f"{name}/material-wood_rand-cube"
    _event(
        version,
        name,
        0,
        "wood_rand",
        "material",
        "cube-cycles-0",
        [
            f"{base}-cycles-0/Camera/0000.png",
            f"{base}-cycles-0/Camera/surface-normal_0000.png",
        ],
    )
    _event(
        version,
        name,
        1,
        "wood_rand",
        "material",
        "cube-cycles-REALIZE_MESH",
        [
            f"{base}-cycles-REALIZE_MESH/Camera/0000.png",
            f"{base}-cycles-REALIZE_MESH/Camera/surface-normal_0000.png",
        ],
    )
    _event(
        version,
        name,
        2,
        "wood_rand",
        "material",
        "cube-eevee-DISPLACEMENT_AND_BUMP",
        [
            f"{base}-eevee-DISPLACEMENT_AND_BUMP/Camera/0000.png",
            f"{base}-eevee-DISPLACEMENT_AND_BUMP/Camera/surface-normal_0000.png",
        ],
    )
    return version


def test_special_variants_keep_their_normals_under_exports(tmp_path):
    """REALIZE_MESH and eevee are non-default settings, so their normals belong
    beside their rgb under Exports rather than in the default Normals row."""
    rows = _build([_material_run(tmp_path, "before"), _material_run(tmp_path, "after")])
    sections = _sections_of(rows, "wood_rand")

    assert set(sections) == {"seeds", "normals", "exports"}
    exports = sections["exports"]["images"]
    assert sorted(i["label"] for i in exports) == [
        "cycles / REALIZE_MESH / image",
        "cycles / REALIZE_MESH / surface-normal",
        "eevee / DISPLACEMENT_AND_BUMP / image",
        "eevee / DISPLACEMENT_AND_BUMP / surface-normal",
    ]
    assert sections["exports"]["folded"] is True


def test_default_cycles_normals_land_in_normals_row(tmp_path):
    rows = _build([_material_run(tmp_path, "before"), _material_run(tmp_path, "after")])
    sections = _sections_of(rows, "wood_rand")

    assert [i["label"] for i in sections["normals"]["images"]] == [
        "cycles / 0 / surface-normal"
    ]
    assert [i["label"] for i in sections["seeds"]["images"]] == ["cycles / 0 / image"]


def _scene_run(root: Path, name: str, seeds: int = 3) -> Path:
    version = root / name
    base = f"{name}/scene-livingroom_rand-demo-cycles"
    for seed in range(seeds):
        images = [
            f"{base}-{seed}/Camera/0000.png",
            f"{base}-{seed}/Camera/surface-normal_0000.png",
            f"{base}-{seed}/Camera/object_0000.png",
            f"{base}-{seed}/bbox3d/0000_bbox3d.png",
        ]
        _event(
            version,
            name,
            seed,
            "livingroom_rand",
            "scene",
            f"demo-cycles-{seed}",
            images,
        )
    return version


def test_scene_boxes_and_segmentation_land_under_exports(tmp_path):
    """Every scene seed exports a 3D box overlay and an object-index mask; both are
    ground truth rather than the look under test, so they fold away under Exports."""
    rows = _build([_scene_run(tmp_path, "before"), _scene_run(tmp_path, "after")])
    sections = _sections_of(rows, "livingroom_rand")

    assert set(sections) == {"seeds", "normals", "exports"}
    assert [i["label"] for i in sections["exports"]["images"]] == [
        "cycles / 0 / bbox3d",
        "cycles / 0 / object-index",
        "cycles / 1 / bbox3d",
        "cycles / 1 / object-index",
        "cycles / 2 / bbox3d",
        "cycles / 2 / object-index",
    ]
    assert [i["label"] for i in sections["seeds"]["images"]] == [
        "cycles / 0 / image",
        "cycles / 1 / image",
        "cycles / 2 / image",
    ]


def test_section_controls_cover_present_sections_only(tmp_path):
    rows = _build([_material_run(tmp_path, "before"), _material_run(tmp_path, "after")])
    controls = build_section_controls(rows)

    assert [c["key"] for c in controls] == ["seeds", "normals", "exports"]
    assert {c["key"]: c["folded"] for c in controls} == {
        "seeds": False,
        "normals": True,
        "exports": True,
    }


def test_section_controls_include_not_rendered_rows(tmp_path):
    before = _object_run(tmp_path, "before")
    after = tmp_path / "after"
    _event(
        after,
        "after",
        0,
        "other_rand",
        "object",
        "demo-cycles-0",
        ["after/object-other_rand-demo-cycles-0/Camera/0000.png"],
    )
    rows = _build([before, after])

    controls = build_section_controls(rows)
    not_run = [c for c in controls if c["key"] == "not-run"]
    assert len(not_run) == 1
    assert not_run[0]["title"] == "1 not-rendered"
    assert not_run[0]["folded"] is True


def _traj_run(root: Path, name: str) -> Path:
    version = root / name
    base = f"{name}/camera-linear_pan_camera_rand-livingroom_rand-workbench-traj2"
    mp4 = f"{base}/image_Camera.mp4"
    (version / mp4).parent.mkdir(parents=True, exist_ok=True)
    (version / mp4).write_bytes(b"not a real video")
    events = version / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": "livingroom_rand",
        "asset_type": "camera",
        "variant_key": "linear_pan-workbench-traj2",
        "status": "success",
        "cmd": ["infinigen", "livingroom_rand"],
        "images": [mp4],
        "tris": 10,
        "cpu_time_sec": 1.0,
        "gpu_time_sec": 2.0,
    }
    (events / "0.json").write_text(json.dumps(payload))
    return version


def test_trajectory_video_does_not_crash_mse(tmp_path):
    """A camera-trajectory .mp4 must display without PIL trying to load it as an
    image, and must not poison the row's avg MSE."""
    rows = _build([_traj_run(tmp_path, "before"), _traj_run(tmp_path, "after")])
    row = next(r for r in rows if r["asset"] == "livingroom_rand")

    paths = [i["path"] for i in row["objects"][-1]["images"] if i["path"]]
    assert any(p.endswith(".mp4") for p in paths)
    avg = row["avg_mse"]
    assert avg is None or math.isfinite(avg)


def test_pairwise_mse_skips_video():
    img = {"pass_type": "image", "filename": "x/image_Camera.mp4"}
    png = {"pass_type": "image", "filename": "x/0000.png"}
    assert display._pairwise_mse(img, png, Path("a"), Path("b")) is None
    assert display._pairwise_mse(img, img, Path("a"), Path("b")) is None
