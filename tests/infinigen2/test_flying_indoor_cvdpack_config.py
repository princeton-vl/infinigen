import json
from pathlib import Path


def _data_types() -> dict:
    path = Path(__file__).parents[2] / "examples/flying_indoor/cvdpack.json"
    return json.loads(path.read_text())["data_types"]


def test_per_camera_types_pack_to_distinct_paths() -> None:
    for name, spec in _data_types().items():
        if "{cam}" not in spec["original_path_template"]:
            continue
        assert "{cam}" in spec["packed_path_template"], name


def test_metadata_packs_at_scene_root_like_released_datasets() -> None:
    spec = _data_types()["metadata"]
    assert spec["original_path_template"] == "{scene}/{traj}/metadata.json"
    assert spec["packed_path_template"] == "{scene}/{traj}/metadata.json"
