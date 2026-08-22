# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import json
from pathlib import Path

import pytest

PACK_CONFIGS = sorted(
    (Path(__file__).parents[3] / "examples").rglob("cvdpack.json"),
)

SEGMENTATION_TABLES = {
    "object_{frame:04d}.npy": "object-index-table.json",
    "material-index_{frame:04d}.npy": "material-index-table.json",
}


def _original_templates(config: Path) -> set[str]:
    data_types = json.loads(config.read_text())["data_types"]
    return {v["original_path_template"] for v in data_types.values()}


@pytest.mark.parametrize("config", PACK_CONFIGS, ids=lambda p: p.parent.name)
def test_segmentation_index_tables_are_packed(config: Path) -> None:
    templates = _original_templates(config)
    for frames, table in SEGMENTATION_TABLES.items():
        if not any(t.endswith(frames) for t in templates):
            continue
        assert any(t.endswith(table) for t in templates), (
            f"{config} packs {frames} but not {table}, which maps its ids onto names. "
            f"The render's only copy of {table} is deleted with the scratch folder."
        )
