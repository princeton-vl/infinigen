# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson, Zeyu Ma: original Infinigen cave generation (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/assets/caves/core.py)
# - David Yan, Jia Deng: optimized implementation

from pathlib import Path

import bpy
import numpy as np

import infinigen.terrain.mesh_to_sdf as mesh_to_sdf
from infinigen.core.util.blender import ViewportMode, select, spawn_vert
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.organization import AssetFile
from infinigen.perf.terrain.land_process import planning
from infinigen.terrain.assets.caves import core as _default
from infinigen.terrain.assets.caves.core import add_cave, trace_string
from infinigen.terrain.assets.caves.pcfg import generate_string
from infinigen.terrain.utils import Mesh


class Cave(_default.Cave):
    def __init__(self, name="Cave") -> None:
        self.modifier_stack = []
        obj = spawn_vert(name)
        select(obj)
        bpy.context.view_layer.objects.active = obj
        generated_string = generate_string(max_len=5000)
        # print(f"Using String", ''.join(generated_string))
        trace_string(["f"] * 2 + generated_string)


def caves_asset(
    folder,
    N=128,
):
    """Build an `N`^3 cave SDF grid in `folder`, or return its step in `planning()`."""
    context = planning.planning_context()
    seed = None if context is None else context.get("seed")
    kwargs = dict(
        folder=Path(folder),
        N=int(N),
        seed=None if seed is None else int(seed),
        # mesh_to_sdf's default
        sample_point_count=10000000,
    )
    if context is not None:
        return (__name__, "caves_asset_step", kwargs)
    return caves_asset_step(**kwargs)


def caves_asset_step(folder, N, seed, sample_point_count):
    if seed is None:
        return _caves_asset(folder, N, sample_point_count)
    with FixedSeed(seed):
        return _caves_asset(folder, N, sample_point_count)


def _caves_asset(folder, N, sample_point_count):
    folder.mkdir(parents=True, exist_ok=True)
    add_cave(rescale=1, cave_z=0)
    name = "Cave"
    obj = bpy.data.objects[name]
    with ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bounding_box = np.array([v[:] for v in obj.bound_box])
    min_gen, max_gen = [0, 0, 0], [0, 0, 0]
    for j in range(3):
        min_gen[j], max_gen[j] = bounding_box[:, j].min(), bounding_box[:, j].max()
    bounding_box = np.array([min_gen, max_gen])
    dim = bounding_box[1] - bounding_box[0]
    bounding_box[0] -= dim / 4
    bounding_box[1] += dim / 4
    cave_mesh = Mesh(obj=obj).to_trimesh()
    bpy.data.objects.remove(obj, do_unlink=True)
    x = np.linspace(bounding_box[0, 0], bounding_box[1, 0], N)
    y = np.linspace(bounding_box[0, 1], bounding_box[1, 1], N)
    z = np.linspace(bounding_box[0, 2], bounding_box[1, 2], N)
    query_points = np.zeros((N, N, N, 3))
    for j in range(N):
        query_points[j, :, :, 0] = x[j]
        query_points[:, j, :, 1] = y[j]
        query_points[:, :, j, 2] = z[j]
    query_points = query_points.reshape(-1, 3)
    voxels = mesh_to_sdf.mesh_to_sdf(
        cave_mesh,
        query_points,
        surface_point_method="sample",
        sample_point_count=sample_point_count,
    ).reshape((N, N, N))
    np.save(folder / "occupancy.npy", voxels)
    np.save(folder / "boundingbox.npy", bounding_box)
    (folder / AssetFile.Finish).touch()


def plan_caves_asset(folder, seed):
    """`caves_asset`'s gin-resolved step, seeded so a worker builds the same cave."""
    with planning.planning(seed=seed):
        return _default.caves_asset(folder)
