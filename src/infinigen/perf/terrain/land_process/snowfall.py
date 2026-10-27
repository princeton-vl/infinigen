# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen snowfall (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/land_process/snowfall.py)
# - David Yan, Jia Deng: optimized implementation

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from infinigen.core.util.organization import AssetFile, Process
from infinigen.perf.terrain.land_process import planning
from infinigen.terrain.land_process import snowfall as _default
from infinigen.terrain.land_process.snowfall import logger, snowfall_params
from infinigen.terrain.utils import get_normal, read, smooth


def run_snowfall(
    folder,
    blending_params=[0, 0.5],
    diffussion_params=[(256, 10, 9), (1024, 10, 5)],
    verbose=0,
):
    """Accumulate snow on `folder`'s eroded heightmap, or return its step in
    `planning()`. Diffusion uses a numpy replica of landlab with bitwise identical
    results; `snowfall_step(backend="landlab")` runs landlab itself."""
    kwargs = dict(
        folder=Path(folder),
        blending_params=list(blending_params),
        diffussion_params=[tuple(p) for p in diffussion_params],
        verbose=verbose,
        backend="numpy",
        normal_params=snowfall_params()["normal_params"],
    )
    if planning.planning_context() is not None:
        return (__name__, "snowfall_step", kwargs)
    return snowfall_step(**kwargs)


def snowfall_step(
    folder, blending_params, diffussion_params, verbose, backend, normal_params
):
    heightmap_path = f"{folder}/{Process.Erosion}.{AssetFile.Heightmap}.exr"
    tile_size = float(np.loadtxt(f"{folder}/{AssetFile.TileSize}.txt"))
    rocks = read(heightmap_path)
    M = rocks.shape[0]

    logger.info(f"Running snowfall simulation for {folder}")

    snows = np.zeros_like(rocks) - 1e9
    for N, n_iters, smoothing_kernel in tqdm(diffussion_params):
        snow = rocks.copy()
        snow = cv2.resize(snow, (N, N))
        if backend == "landlab":
            snow = landlab_transport_length_diffusion(snow, n_iters, verbose)
        elif backend == "numpy":
            snow = transport_length_diffusion(snow, n_iters)
        else:
            raise ValueError(f"unknown snowfall backend {backend!r}")
        snow = cv2.resize(snow, (M, M))
        snow = smooth(snow, smoothing_kernel)
        snows = np.maximum(snows, snow)

    mask = np.zeros_like(rocks)
    for blending in tqdm(blending_params):
        for normal_preference, (th0, th1) in normal_params:
            reference_snow = rocks * blending + snows * (1 - blending)
            normal_map = get_normal(reference_snow, tile_size / snows.shape[0])
            mask_sharpening = 1 / (th1 - th0)
            mask += np.clip(
                (
                    (normal_map * np.array(normal_preference).reshape((1, 1, 3))).sum(
                        axis=-1
                    )
                    - th0
                )
                * mask_sharpening,
                a_min=0,
                a_max=1,
            )
            mask -= np.clip(
                (
                    (-normal_map * np.array(normal_preference).reshape((1, 1, 3))).sum(
                        axis=-1
                    )
                    - th0
                )
                * mask_sharpening,
                a_min=0,
                a_max=1,
            )
            mask = np.clip(mask, a_min=0, a_max=1)

    heightmap = snows * mask + rocks * (1 - mask)
    cv2.imwrite(
        str(folder / f"{Process.Snowfall}.{AssetFile.Heightmap}.exr"),
        heightmap.astype(np.float32),
    )
    cv2.imwrite(
        str(folder / f"{Process.Snowfall}.{AssetFile.Mask}.exr"),
        mask.astype(np.float32),
    )


def steepest_receivers(z, R, C):
    """D4 steepest-descent `(receiver, steepest_slope)` flat arrays on an `R` x `C`
    raster, matching landlab's `FlowDirectorSteepest` with all edges open. Ties go to
    the lowest landlab link id (below, left, right, above); perimeter nodes are their
    own receivers with slope 0."""
    z2 = z.reshape(R, C)
    core = z2[1:-1, 1:-1]
    drops = np.stack(
        [
            core - z2[:-2, 1:-1],
            core - z2[1:-1, :-2],
            core - z2[1:-1, 2:],
            core - z2[2:, 1:-1],
        ]
    )
    best = np.argmax(drops, axis=0)  # first occurrence wins ties
    best_drop = np.take_along_axis(drops, best[None], axis=0)[0]
    rows, cols = np.mgrid[1 : R - 1, 1 : C - 1]
    node = rows * C + cols
    offsets = np.array([-C, -1, 1, C])
    receiver = np.arange(R * C)
    steepest = np.zeros(R * C)
    receiver.reshape(R, C)[1:-1, 1:-1] = np.where(
        best_drop > 0, node + offsets[best], node
    )
    steepest.reshape(R, C)[1:-1, 1:-1] = np.where(best_drop > 0, best_drop, 0.0)
    return receiver, steepest


def transport_length_diffusion(
    elev, n_iters, erodibility=0.001, slope_crit=0.6, dt=1.0
):
    """Transport-length hillslope diffusion for `n_iters` steps of `dt` on an
    open-perimeter raster, a bitwise identical replica of landlab's
    `FlowDirectorSteepest` then `TransportLengthHillslopeDiffuser` (float64, unit grid
    spacing). `np.add.at` sums the fluxes in the same node order as landlab's loop."""
    R, C = elev.shape
    elev = np.array(elev, dtype=np.float64).reshape(-1)
    n = R * C
    rows, cols = np.mgrid[1 : R - 1, 1 : C - 1]
    core = (rows * C + cols).reshape(-1)
    flux_out = np.zeros(n)
    erosion = np.zeros(n)
    depo = np.zeros(n)
    trans = np.zeros(n)
    d_coeff = np.zeros(n)
    dx = 1.0
    for _ in range(n_iters):
        receiver, steepest = steepest_receivers(elev, R, C)
        flux_in = np.zeros(n)
        np.add.at(flux_in, receiver[core], flux_out[core])
        s = steepest[core]
        d_coeff[core] = np.where(
            s >= slope_crit,
            1000000000.0,
            1 / (1 - (np.power((s / slope_crit), 2))),
        )
        depo[core] = flux_in[core] / d_coeff[core]
        erosion[core] = np.where(
            s > slope_crit,
            dx * (s - slope_crit) / (100 * dt),
            erodibility * s,
        )
        elev[core] += (-erosion[core] + depo[core]) * dt
        trans[core] = flux_in[core] - depo[core]
        flux_out[:] = erosion + trans
    return elev.reshape(R, C)


def landlab_transport_length_diffusion(elev, n_iters, verbose=0):
    """Reference implementation using landlab, for equivalence tests."""
    try:
        from landlab import RasterModelGrid
        from landlab.components import (
            FlowDirectorSteepest,
            TransportLengthHillslopeDiffuser,
        )
    except ImportError as e:
        raise ImportError(
            "landlab import failed for terrain snowfall simulation\n"
            f" original error: {e}\n"
            "You may need to install terrain dependencies via `pip install .[terrain]`"
        ) from e
    mg = RasterModelGrid(elev.shape)
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    # landlab stores a view of the array it is given; copy so the input is untouched
    _ = mg.add_field(
        "topographic__elevation", np.array(elev, dtype=np.float64), at="node"
    )
    fdir = FlowDirectorSteepest(mg)
    tl_diff = TransportLengthHillslopeDiffuser(mg, erodibility=0.001, slope_crit=0.6)
    range_t = tqdm(range(n_iters)) if verbose else range(n_iters)
    for t in range_t:
        fdir.run_one_step()
        tl_diff.run_one_step(1.0)
    return mg.at_node["topographic__elevation"].reshape(elev.shape)


def plan_snowfall(folder, **kwargs):
    """`run_snowfall`'s step for a worker, with its gin parameters resolved."""
    with planning.planning():
        return _default.run_snowfall(folder, **kwargs)
