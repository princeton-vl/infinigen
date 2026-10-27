# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import os

import pytest

from infinigen._perf_selection import selected  # noqa: E402

if not selected("TERRAIN"):
    pytest.skip(
        "exercises the optimized terrain implementation; "
        "pass --optimized=terrain (or --optimized) to run",
        allow_module_level=True,
    )


import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402

from infinigen.core.util.organization import AssetFile, Process  # noqa: E402
from infinigen.perf.terrain.land_process import planning, pool  # noqa: E402
from infinigen.perf.terrain.land_process.erosion import plan_erosion  # noqa: E402
from infinigen.perf.terrain.land_process.planning import (  # noqa: E402
    mark_finished,
    run_land_process_steps,
    run_steps,
)
from infinigen.perf.terrain.land_process.pool import land_process_pool  # noqa: E402
from infinigen.perf.terrain.land_process.snowfall import (  # noqa: E402
    landlab_transport_length_diffusion,
    plan_snowfall,
    steepest_receivers,
    transport_length_diffusion,
)

landlab = pytest.importorskip("landlab", reason="landlab reference not installed")
from landlab import RasterModelGrid  # noqa: E402
from landlab.components import FlowDirectorSteepest  # noqa: E402


def _landlab_receivers(z):
    mg = RasterModelGrid(z.shape)
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    mg.add_field("topographic__elevation", z.copy(), at="node")
    FlowDirectorSteepest(mg).run_one_step()
    return (
        mg.at_node["flow__receiver_node"].copy(),
        mg.at_node["topographic__steepest_slope"].copy(),
    )


@pytest.mark.parametrize("shape", [(7, 9), (37, 41)])
def test_steepest_receivers_match_landlab_with_ties(shape):
    rng = np.random.default_rng(0)
    z = rng.integers(0, 4, size=shape).astype(np.float64)  # many exact ties
    receiver, slope = steepest_receivers(z.reshape(-1), *shape)
    ref_receiver, ref_slope = _landlab_receivers(z)
    assert np.array_equal(receiver, ref_receiver)
    assert np.array_equal(slope, ref_slope)


def test_transport_length_diffusion_bitwise_matches_landlab():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(48, 40)).cumsum(axis=0).cumsum(axis=1)
    z += rng.integers(0, 3, size=z.shape)  # inject ties
    ours = transport_length_diffusion(z, 12)
    ref = landlab_transport_length_diffusion(z, 12)
    assert ours.dtype == np.float64
    assert np.array_equal(ours, ref)
    assert not np.array_equal(ours, z)  # the simulation did something


def test_planning_returns_resolved_steps(tmp_path):
    step = plan_erosion(tmp_path, n_iters=[3, 4])
    module, function, kwargs = step
    assert module == "infinigen.perf.terrain.land_process.erosion"
    assert function == "erosion_step"
    assert kwargs["folder"] == tmp_path and kwargs["n_iters"] == [3, 4]
    assert kwargs["Ns"] == [512, 2048]
    step = plan_snowfall(tmp_path)
    assert step[1] == "snowfall_step"
    assert len(step[2]["normal_params"]) == 2
    assert not list(tmp_path.iterdir())  # nothing ran


def _make_tile(folder, seed=3, N=64, tile_size=600.0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, N)
    hm = 30 * np.exp(-(x[:, None] ** 2 + x[None, :] ** 2)) + rng.normal(size=(N, N))
    cv2.imwrite(str(folder / f"{AssetFile.Heightmap}.exr"), hm.astype(np.float32))
    (folder / f"{AssetFile.TileSize}.txt").write_text(f"{tile_size}\n")


def _soil_machine_built():
    """True when the compiled erosion library is present; a default install lacks it."""
    import infinigen
    from infinigen.perf.terrain.land_process import erosion

    root = Path(infinigen.__file__).parent
    return (root / erosion._DEFAULT_LIBRARY).is_file()


needs_soil_machine = pytest.mark.skipif(
    not _soil_machine_built(),
    reason="needs the compiled terrain libraries (INFINIGEN_MINIMAL_INSTALL=False)",
)


def _steps(folder, backend="numpy"):
    snow = plan_snowfall(folder, diffussion_params=[(16, 4, 3), (64, 4, 3)])
    snow[2]["backend"] = backend
    return [plan_erosion(folder, Ns=[32, 64], n_iters=[50, 200]), snow]


OUTPUTS = [
    f"{Process.Erosion}.{AssetFile.Heightmap}.exr",
    f"{Process.Erosion}.{AssetFile.Mask}.exr",
    f"{Process.Snowfall}.{AssetFile.Heightmap}.exr",
    f"{Process.Snowfall}.{AssetFile.Mask}.exr",
]


def _digests(folder):
    return {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in OUTPUTS}


@needs_soil_machine
def test_planned_steps_match_landlab(tmp_path):
    inline = tmp_path / "inline"
    _make_tile(inline)
    run_steps(_steps(inline))
    (inline / AssetFile.Finish).touch()
    assert (inline / AssetFile.Finish).exists()

    ref = tmp_path / "landlab"
    _make_tile(ref)
    run_steps(_steps(ref, backend="landlab"))
    assert _digests(ref) == _digests(inline)


def test_pool_is_off_by_default(tmp_path):
    """Without `enabled`, `land_process_pool` does nothing and steps run inline."""
    folder = tmp_path / "tile"
    folder.mkdir()
    with land_process_pool():
        assert planning._active_pool is None
        run_land_process_steps(folder, [None])
        mark_finished(folder, folder / AssetFile.Finish)
        assert (folder / AssetFile.Finish).exists()
    assert not (folder / pool.LOG_FILE).exists()
    # one worker is no pool either
    with land_process_pool(enabled=True, max_workers=1):
        assert planning._active_pool is None


@needs_soil_machine
def test_pool_workers_match_inline(tmp_path):
    inline = tmp_path / "inline"
    _make_tile(inline)
    run_steps(_steps(inline))

    pooled = [tmp_path / "pool0", tmp_path / "pool1"]
    with land_process_pool(enabled=True, max_workers=2) as active:
        assert planning._active_pool is active
        for folder in pooled:
            _make_tile(folder)
            run_land_process_steps(folder, _steps(folder))
            mark_finished(folder, folder / AssetFile.Finish)
            # the marker waits for the worker
            assert not (folder / AssetFile.Finish).exists()
    assert planning._active_pool is None
    for folder in pooled:
        assert (folder / AssetFile.Finish).exists()
        assert (folder / pool.LOG_FILE).exists()
        assert _digests(folder) == _digests(inline)


def test_pool_reports_worker_failure(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    with pytest.raises(RuntimeError, match="failed with exit code"):
        with land_process_pool(enabled=True, max_workers=2):
            run_land_process_steps(
                folder,
                [("infinigen.perf.terrain.land_process.erosion", "erosion_step", {})],
            )


def test_pool_rejects_double_submission(tmp_path):
    active = pool.LandProcessPool(max_workers=2)
    step = ("infinigen.perf.terrain.land_process.erosion", "erosion_step", {})
    active.submit(tmp_path / "a", [step])
    with pytest.raises(ValueError, match="already submitted"):
        active.submit(tmp_path / "a", [step])
    active.terminate()


def test_landtile_asset_skips_unneeded_processes(monkeypatch, tmp_path):
    from infinigen.perf import options
    from infinigen.perf.terrain.assets.landtiles import core as perf_core
    from infinigen.terrain.assets.landtiles import core

    calls = []

    def fake_asset(folder, tile_size, resolution, device, **kwargs):
        Path(folder).mkdir(parents=True, exist_ok=True)
        calls.append(kwargs)

    monkeypatch.setattr(perf_core, "multi_mountains_asset", fake_asset)
    for needed, expected in [
        (None, {"erosion": False, "snowfall": False}),
        (Process.Erosion, {"snowfall": False}),
        (Process.Snowfall, {}),
    ]:
        folder = tmp_path / str(needed)
        perf_core.landtile_asset_for(needed, folder, "MultiMountains")
        assert calls[-1] == expected
        assert (folder / AssetFile.Finish).exists()
    # asset-library generation says nothing about what is read: everything runs
    folder = tmp_path / "library"
    core.landtile_asset(folder, "MultiMountains")
    assert calls[-1] == {}
    folder = tmp_path / "all"
    monkeypatch.setattr(options, "only_needed_land_processes", lambda: False)
    perf_core.landtile_asset_for(None, folder, "MultiMountains")
    assert calls[-1] == {}
    shutil.rmtree(tmp_path)


def test_caves_asset_step_matches_inline(tmp_path):
    from infinigen.core.util.math import FixedSeed, int_hash
    from infinigen.perf.terrain.assets.caves import core as perf_caves

    seed = int_hash(("Caves", 7, 0))
    planned = tmp_path / "planned"
    with FixedSeed(seed):
        step = perf_caves.plan_caves_asset(planned, seed)
    assert step[1] == "caves_asset_step" and not planned.exists()
    assert step[2]["seed"] == seed and step[2]["N"] == 128
    # smaller than the configured grid, to keep the test fast
    step[2].update(N=16, sample_point_count=20000)
    inline = tmp_path / "inline"
    with FixedSeed(seed):
        # the inline path: no reseeding, the caller's FixedSeed decides
        perf_caves.caves_asset_step(inline, 16, None, 20000)
    run_steps([step])
    for name in ["occupancy.npy", "boundingbox.npy"]:
        assert np.array_equal(np.load(inline / name), np.load(planned / name)), name
    assert (planned / AssetFile.Finish).exists()
    occupancy = np.load(inline / "occupancy.npy")
    assert occupancy.shape == (16, 16, 16) and (occupancy < 0).any()


def test_surface_point_cloud_backends_agree_exactly(monkeypatch):
    import trimesh

    from infinigen.perf.terrain.mesh_to_sdf import surface_point_cloud as optimized
    from infinigen.perf.terrain.mesh_to_sdf.surface_point_cloud import _query
    from infinigen.terrain.mesh_to_sdf.surface_point_cloud import sample_from_mesh

    mesh = trimesh.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, size=(60000, 3))
    results = {}
    for backend in ["sklearn", "scipy"]:
        # scipy is used whenever it is installed; sklearn is the fallback
        monkeypatch.setattr(
            optimized, "_kdtree_backend", lambda backend=backend: backend
        )
        cloud = sample_from_mesh(mesh, sample_point_count=50000)
        assert cloud.backend == backend
        d11, i11 = _query(cloud, q, k=11)
        d1, i1 = _query(cloud, q, k=1)
        results[backend] = [d11, i11, d1, i1, cloud.get_sdf(q)]
    for x, y in zip(results["sklearn"], results["scipy"]):
        assert x.shape == y.shape and np.array_equal(x, y)
    # a pool worker's threaded queries return the same neighbours
    monkeypatch.setenv("INFINIGEN_KDTREE_THREADS", "4")
    for backend in ["sklearn", "scipy"]:
        monkeypatch.setattr(
            optimized, "_kdtree_backend", lambda backend=backend: backend
        )
        cloud = sample_from_mesh(mesh, sample_point_count=50000)
        d11, i11 = _query(cloud, q, k=11)
        assert np.array_equal(d11, results[backend][0])
        assert np.array_equal(i11, results[backend][1])


def test_duplicate_tiles_are_generated_once():
    import inspect

    from infinigen.perf.terrain.elements import landtiles

    src = inspect.getsource(landtiles.LandTiles.load_assets)
    assert "if folder in created:" in src


def test_optimized_soil_machine_matches_default(tmp_path, monkeypatch):
    from infinigen.perf.terrain.land_process import erosion

    if erosion._soil_machine_library() != erosion._OPTIMIZED_LIBRARY:
        pytest.skip("SoilMachine_perf.so not built")
    outputs = {}
    for name, library in [
        ("default", erosion._DEFAULT_LIBRARY),
        ("optimized", erosion._OPTIMIZED_LIBRARY),
    ]:
        folder = tmp_path / name
        _make_tile(folder)
        monkeypatch.setattr(
            erosion, "_soil_machine_library", lambda library=library: library
        )
        run_steps([plan_erosion(folder, Ns=[32, 64], n_iters=[50, 200])])
        outputs[name] = {
            f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
            for f in OUTPUTS[:2]
        }
    assert outputs["default"] == outputs["optimized"]
