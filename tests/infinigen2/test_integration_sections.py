# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import math
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import compare  # noqa: E402
import display  # noqa: E402
import freeze  # noqa: E402
import manifest_aliases  # noqa: E402
from display import (  # noqa: E402
    build_comparison_data,
    build_section_controls,
    collect_images_structured,
)


def _event(
    root: Path,
    name: str,
    idx: int,
    gen: str,
    atype: str,
    variant: str,
    images,
    color=0,
):
    for image in images:
        (root / image).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=color).save(root / image)
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
    base = f"{name}/camera-camera_linear_pan_rand-room_livingroom_rand-workbench-traj2"
    mp4 = f"{base}/image_Camera.mp4"
    (version / mp4).parent.mkdir(parents=True, exist_ok=True)
    (version / mp4).write_bytes(b"not a real video")
    events = version / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": "room_livingroom_rand",
        "asset_type": "camera",
        "variant_key": "linear_pan-workbench-traj2",
        "status": "success",
        "cmd": ["infinigen", "room_livingroom_rand"],
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
    row = next(r for r in rows if r["asset"] == "room_livingroom_rand")

    paths = [i["path"] for i in row["objects"][-1]["images"] if i["path"]]
    assert any(p.endswith(".mp4") for p in paths)
    avg = row["avg_mse"]
    assert avg is None or math.isfinite(avg)


def _sort_rows():
    def row(asset, asset_type, new=False, not_run=False, mse=None):
        return {
            "asset": asset,
            "asset_type": asset_type,
            "is_new": new,
            "not_run": not_run,
            "avg_mse": mse,
        }

    return [
        row("unknown_row", "unknown"),
        row("landing_row", "landing"),
        row("preset_row", "preset"),
        row("camera_row", "camera"),
        row("env_row", "environment"),
        row("mask_row", "mask"),
        row("mat_new", "material", new=True),
        row("mat_old", "material", mse=0.1),
        row("obj_notrun", "object", not_run=True),
        row("obj_big", "object", mse=5.0),
        row("obj_small", "object", mse=0.2),
        row("obj_new", "object", new=True),
        row("scene_old", "scene", mse=1.0),
        row("scene_new", "scene", new=True),
    ]


def test_row_sort_defaults_to_mse():
    rows = _sort_rows()
    display._sort_rows(rows, ["before", "after"], sort_order=None)

    assert [r["asset"] for r in rows[:4]] == [
        "obj_big",
        "scene_old",
        "obj_small",
        "mat_old",
    ]
    assert rows[-1]["asset"] == "obj_notrun"


def test_type_row_sort_orders_by_type_then_new():
    rows = _sort_rows()
    display._sort_rows(rows, ["before", "after"], sort_order="type")

    assert [r["asset"] for r in rows] == [
        "scene_new",
        "scene_old",
        "obj_new",
        "obj_big",
        "obj_small",
        "obj_notrun",
        "mat_new",
        "mat_old",
        "mask_row",
        "env_row",
        "camera_row",
        "preset_row",
        "landing_row",
        "unknown_row",
    ]


def _viewer_sort_run(root: Path, name: str, scene_color: int, object_color: int):
    version = root / name
    assets = [
        ("scene_rand", "scene", scene_color),
        ("object_rand", "object", object_color),
    ]
    for idx, (asset, asset_type, color) in enumerate(assets):
        image = f"{name}/{asset}/Camera/0000.png"
        _event(
            version,
            name,
            idx,
            asset,
            asset_type,
            "demo-cycles-0",
            [image],
            color=color,
        )
    return version


def test_viewer_type_sort_is_opt_in(tmp_path):
    before = _viewer_sort_run(tmp_path, "before", 0, 0)
    after = _viewer_sort_run(tmp_path, "after", 10, 255)
    query = urlencode([("v", before), ("v", after)])
    client = compare.app.test_client()

    default_html = client.get(f"/?{query}").get_data(as_text=True)
    type_html = client.get(f"/?{query}&sort=type").get_data(as_text=True)

    object_row = 'data-asset="object_rand" data-not-run'
    scene_row = 'data-asset="scene_rand" data-not-run'
    assert default_html.index(object_row) < default_html.index(scene_row)
    assert type_html.index(scene_row) < type_html.index(object_row)
    assert client.get(f"/?{query}&sort=unknown").status_code == 400


def test_viewer_exposes_displacement_filter(tmp_path):
    before = _viewer_sort_run(tmp_path, "before", 0, 0)
    after = _viewer_sort_run(tmp_path, "after", 10, 255)
    query = urlencode([("v", before), ("v", after)])
    client = compare.app.test_client()

    html = client.get(f"/?{query}").get_data(as_text=True)

    assert 'data-filter-value="displacement" checked' in html


def test_freeze_forwards_type_sort(tmp_path):
    before = _viewer_sort_run(tmp_path, "before", 0, 0)
    after = _viewer_sort_run(tmp_path, "after", 10, 255)
    pages = tmp_path / "pages"
    pages.mkdir()

    freeze.render_pages(
        pages, [("before", before), ("after", after)], sort_order="type"
    )

    html = (pages / "index.html").read_text()
    object_row = 'data-asset="object_rand" data-not-run'
    scene_row = 'data-asset="scene_rand" data-not-run'
    assert html.index(scene_row) < html.index(object_row)


def _renamed_run(root: Path, version_name: str, generator: str) -> Path:
    version = root / version_name
    image = f"{version_name}/{generator}/Camera/0000.png"
    _event(
        version,
        version_name,
        0,
        generator,
        "object",
        "demo-cycles-0",
        [image],
    )
    return version


def _write_alias_manifest(version: Path, old_names: object) -> None:
    manifest = [
        {
            "category": "Object",
            "name": "infinigen2.assets.objects.tables.table_circle_rand",
            "old_names": old_names,
        }
    ]
    (version / "manifest.json").write_text(json.dumps(manifest))


def test_renamed_generators_share_row_with_source_names(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "circle_table_rand")
    after = _renamed_run(tmp_path, "after", "table_circle_rand")
    _write_alias_manifest(after, ["circle_table_rand"])

    rows = _build([before, after])

    assert [row["asset"] for row in rows] == ["table_circle_rand"]
    assert [obj["source_asset"] for obj in rows[0]["objects"]] == [
        "circle_table_rand",
        "table_circle_rand",
    ]
    assert rows[0]["is_new"] is False
    assert rows[0]["not_run"] is False


def test_archive_without_alias_metadata_keeps_literal_names(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "circle_table_rand")
    after = _renamed_run(tmp_path, "after", "table_circle_rand")

    rows = _build([before, after])

    assert {row["asset"] for row in rows} == {
        "circle_table_rand",
        "table_circle_rand",
    }


def test_rightmost_manifest_is_authoritative(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "circle_table_rand")
    after = _renamed_run(tmp_path, "after", "table_circle_rand")
    stale_manifest = [
        {
            "name": "pkg.table_round_rand",
            "old_names": ["circle_table_rand"],
        }
    ]
    (before / "manifest.json").write_text(json.dumps(stale_manifest))
    _write_alias_manifest(after, ["circle_table_rand"])

    rows = _build([before, after])

    assert [row["asset"] for row in rows] == ["table_circle_rand"]


@pytest.mark.parametrize(
    "manifest",
    [
        [
            {
                "name": "pkg.table_circle_rand",
                "old_names": "circle_table_rand",
            }
        ],
        [
            {
                "name": "pkg.table_circle_rand",
                "old_names": ["circle_table_rand"],
            },
            {
                "name": "pkg.table_round_rand",
                "old_names": ["circle_table_rand"],
            },
        ],
    ],
)
def test_alias_manifest_rejects_malformed_or_ambiguous_metadata(
    manifest: object,
) -> None:
    with pytest.raises(ValueError):
        manifest_aliases.parse_ci_compare_aliases(manifest)


def test_archive_with_current_name_keeps_old_name_row(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "circle_table_rand")
    image = "before/table_circle_rand/Camera/0000.png"
    _event(
        before,
        "before",
        1,
        "table_circle_rand",
        "object",
        "demo-cycles-0",
        [image],
    )
    after = _renamed_run(tmp_path, "after", "table_circle_rand")
    _write_alias_manifest(after, ["circle_table_rand"])

    rows = _build([before, after])

    sources = {
        row["asset"]: [obj["source_asset"] for obj in row["objects"]] for row in rows
    }
    assert sources == {
        "circle_table_rand": ["circle_table_rand", None],
        "table_circle_rand": ["table_circle_rand", "table_circle_rand"],
    }


def test_comparison_rejects_two_old_names_for_one_generator(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "circle_table_rand")
    image = "before/round_table_rand/Camera/0000.png"
    _event(before, "before", 1, "round_table_rand", "object", "demo-cycles-0", [image])
    after = _renamed_run(tmp_path, "after", "table_circle_rand")
    _write_alias_manifest(after, ["circle_table_rand", "round_table_rand"])

    with pytest.raises(ValueError, match="both alias"):
        _build([before, after])


def test_renamed_scene_in_camera_variant_aligns_images(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _event(
        before,
        "before",
        0,
        "camera_rrt",
        "camera",
        "livingroom_rand-workbench-traj0",
        ["camera-camera_rrt-livingroom_rand-workbench-traj0/Camera/0000.png"],
    )
    _event(
        after,
        "after",
        0,
        "camera_rrt",
        "camera",
        "room_livingroom_rand-workbench-traj0",
        ["camera-camera_rrt-room_livingroom_rand-workbench-traj0/Camera/0000.png"],
    )
    manifest = [{"name": "pkg.room_livingroom_rand", "old_names": ["livingroom_rand"]}]
    (after / "manifest.json").write_text(json.dumps(manifest))

    rows = _build([before, after])

    assert [len(obj["images"]) for obj in rows[0]["objects"]] == [1, 1]
    assert all(obj["images"][0]["path"] for obj in rows[0]["objects"])


def test_renamed_presets_list_under_renamed_parent(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "stone_rand")
    _event(
        before,
        "before",
        1,
        "stone_grey_preset",
        "preset",
        "cube-cycles-0",
        ["before/stone_grey_preset/Camera/0000.png"],
    )
    (before / "preset_parents.json").write_text(
        json.dumps({"stone_grey_preset": "stone_rand"})
    )
    after = _renamed_run(tmp_path, "after", "stone_smooth_rand")
    _event(
        after,
        "after",
        1,
        "stone_smooth_grey_preset",
        "preset",
        "cube-cycles-0",
        ["after/stone_smooth_grey_preset/Camera/0000.png"],
    )
    (after / "preset_parents.json").write_text(
        json.dumps({"stone_smooth_grey_preset": "stone_smooth_rand"})
    )
    manifest = [{"name": "pkg.stone_smooth_rand", "old_names": ["stone_rand"]}]
    (after / "manifest.json").write_text(json.dumps(manifest))

    rows = _build([before, after])

    assert [row["asset"] for row in rows] == ["stone_smooth_rand"]
    presets = [
        [img["label"] for img in obj["images"] if img.get("is_preset")]
        for obj in rows[0]["objects"]
    ]
    assert presets == [["stone_grey"], ["stone_smooth_grey"]]


def test_freeze_shows_each_column_source_generator_name(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "circle_table_rand")
    after = _renamed_run(tmp_path, "after", "table_circle_rand")
    _write_alias_manifest(after, ["circle_table_rand"])
    pages = tmp_path / "pages"
    pages.mkdir()

    freeze.render_pages(pages, [("before", before), ("after", after)], None)

    html = (pages / "index.html").read_text()
    assert html.count('data-asset="table_circle_rand" data-not-run') == 1
    assert '<div class="source-asset-name">circle_table_rand</div>' in html
    assert '<div class="source-asset-name">table_circle_rand</div>' in html


def test_traceback_paths_link_to_the_rendered_commit() -> None:
    base = compare.source_base_url(
        {"repo": "org/repo", "commit": "abc123", "source_ref": None}
    )
    trace = (
        'File "/home/ci/work/repo/src/infinigen2/a.py", line 42, in f\n'
        'File "/venv/lib/python3.11/site-packages/bpy/b.py", line 7\n'
        "scripts/run.py:3 <x>"
    )

    html = str(compare.link_sources(trace, base))

    assert (
        'href="https://github.com/org/repo/blob/abc123/src/infinigen2/a.py#L42"' in html
    )
    assert 'href="https://github.com/org/repo/blob/abc123/scripts/run.py#L3"' in html
    assert "bpy/b.py" in html and html.count("<a ") == 2
    assert "&lt;x&gt;" in html
    assert compare.link_sources("<x> src/a.py", None) == "&lt;x&gt; src/a.py"


def test_freeze_links_sources_to_public_repo_at_version_tag(tmp_path: Path) -> None:
    src = tmp_path / "run"
    src.mkdir()
    (src / "git_info.toml").write_text('commit = "abc123"\nrepo = "org/internal"\n')

    freeze.alias_version(tmp_path / "v2.0.0a1", src, "v2.0.0a1", [])

    git = compare.get_git_info(tmp_path / "v2.0.0a1")
    assert git["tag"] == "v2.0.0a1"
    assert compare.source_base_url(git) == (
        "https://github.com/princeton-vl/infinigen/blob/v2.0.0a1"
    )


def test_freeze_replaces_unrendered_media_with_placeholder(tmp_path: Path) -> None:
    before = _renamed_run(tmp_path, "before", "rock_rand")
    after = _renamed_run(tmp_path, "after", "rock_rand")
    (after / "after" / "rock_rand" / "Camera" / "0000.png").unlink()
    pages = tmp_path / "pages"
    pages.mkdir()

    refs = freeze.render_pages(pages, [("before", before), ("after", after)], None)

    html = (pages / "index.html").read_text()
    assert all(ref.startswith("before/") for ref in refs) and refs
    assert 'src="../after/' not in html
    assert "Not generated in this run" in html
