# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from pathlib import Path

import numpy as np

from infinigen2.util import external_assets


def test_resolve_asset_paths_with_glob(tmp_path: Path):
    (tmp_path / "forks").mkdir()
    (tmp_path / "spoons").mkdir()
    (tmp_path / "forks" / "a.obj").write_text("o test\n")
    (tmp_path / "spoons" / "b.fbx").write_text("fake")
    (tmp_path / "spoons" / "ignore.txt").write_text("ignore")

    pattern = Path(str(tmp_path / "*" / "*"))
    paths = external_assets._resolve_asset_paths(pattern)
    assert [p.name for p in paths] == ["a.obj", "b.fbx"]


def test_sample_asset_path_is_deterministic():
    paths = [Path("a.obj"), Path("b.obj"), Path("c.obj")]
    rng_1 = np.random.default_rng(12345)
    rng_2 = np.random.default_rng(12345)

    picks_1 = [external_assets._sample_asset_path(paths, rng_1) for _i in range(10)]
    picks_2 = [external_assets._sample_asset_path(paths, rng_2) for _i in range(10)]
    assert picks_1 == picks_2
