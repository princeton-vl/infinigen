# COPYRIGHT

# Original files authored by Marian Kleineberg: https://github.com/marian42/mesh_to_sdf/tree/master


# All other contributions:
# Copyright (C) 2023, Princeton University.
# Licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen mesh-to-SDF surface point cloud (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/mesh_to_sdf/surface_point_cloud.py)
# - David Yan, Jia Deng: optimized implementation

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.neighbors import KDTree

from infinigen.terrain.mesh_to_sdf import surface_point_cloud as _default

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


def _kdtree_backend():
    """`scipy` (cKDTree) when it is installed, else `sklearn`."""
    return "scipy" if cKDTree is not None else "sklearn"


class SurfacePointCloud(_default.SurfacePointCloud):
    def __init__(self, mesh, points, normals=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans
        self.backend = _kdtree_backend()
        if self.backend == "scipy":
            # Surface samples are a thin 2D set in 3D and most grid queries lie
            # far from it, where balanced trees with compacted nodes degrade to
            # near-linear scans; sliding-midpoint splits do not. The tree layout
            # does not change which neighbours are returned.
            self.kd_tree = cKDTree(
                points, leafsize=32, balanced_tree=False, compact_nodes=False
            )
        else:
            self.kd_tree = KDTree(points)

    def get_sdf(
        self,
        query_points,
        use_depth_buffer=False,
        sample_count=11,
        return_gradients=False,
    ):
        if use_depth_buffer:
            distances, indices = _query(self, query_points)
            distances = distances.astype(np.float32).reshape(-1)
            inside = ~self.is_outside(query_points)
            distances[inside] *= -1

            if return_gradients:
                gradients = query_points - self.points[indices[:, 0]]
                gradients[inside] *= -1

        else:
            distances, indices = _query(self, query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_from_surface = query_points[:, np.newaxis, :] - closest_points
            inside = (
                np.einsum("ijk,ijk->ij", direction_from_surface, self.normals[indices])
                < 0
            )
            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1

            if return_gradients:
                gradients = direction_from_surface[:, 0]
                gradients[inside] *= -1

        if return_gradients:
            near_surface = (
                np.abs(distances) < math.sqrt(0.0025**2 * 3) * 3
            )  # 3D 2-norm stdev * 3
            gradients = np.where(
                near_surface[:, np.newaxis], self.normals[indices[:, 0]], gradients
            )
            gradients /= np.linalg.norm(gradients, axis=1)[:, np.newaxis]
            return distances, gradients
        else:
            return distances


def _kdtree_query_threads():
    """KD-tree query threads: `INFINIGEN_KDTREE_THREADS` (set by the pool), else one."""
    try:
        return max(1, int(os.environ.get("INFINIGEN_KDTREE_THREADS", 1)))
    except ValueError:
        return 1


def _query(cloud, query_points, k=1):
    """Nearest neighbours `(distances, indices)` of shape `(n, k)`. Both backends find
    the true neighbours on any thread split, so they agree up to ties in distance."""
    n_threads = _kdtree_query_threads()
    if cloud.backend == "scipy":
        distances, indices = cloud.kd_tree.query(query_points, k=k, workers=n_threads)
        n = len(query_points)
        return distances.reshape(n, k), indices.reshape(n, k)
    if n_threads == 1 or len(query_points) < 20000:
        return cloud.kd_tree.query(query_points, k=k)
    chunks = np.array_split(query_points, n_threads * 4)
    with ThreadPoolExecutor(n_threads) as ex:
        results = list(ex.map(lambda c: cloud.kd_tree.query(c, k=k), chunks))
    return (
        np.concatenate([r[0] for r in results]),
        np.concatenate([r[1] for r in results]),
    )
