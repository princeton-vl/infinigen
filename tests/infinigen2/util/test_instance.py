# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import procfunc as pf

from infinigen.core.util import blender as butil
from infinigen2.util.instance import instanced_objects


def test_instance_parent_child_instancing_data(rng: pf.RNG):
    butil.clear_scene()

    parent_dense = pf.ops.primitives.mesh_plane(size=12.0, location=(2, 0, 0))
    parent_sparse = pf.ops.primitives.mesh_plane(size=12.0, location=(-20, 0, 0))
    child = pf.Collection([pf.ops.primitives.mesh_uv_sphere(radius=0.35)])

    dense_aliases = instanced_objects(
        rng, parent=parent_dense, child=child, distance_min=0.0
    )
    sparse_aliases = instanced_objects(
        rng, parent=parent_sparse, child=child, distance_min=3.5
    )

    # Higher point spacing should produce fewer instance aliases.
    assert len(dense_aliases) > len(sparse_aliases)

    # Every alias should have mesh data with at least one vertex.
    assert len(dense_aliases) > 0
    for alias in dense_aliases:
        obj = alias.item()
        assert obj.data is not None
        assert len(obj.data.vertices) > 0

    # Ensure instanced child geometry is present above the parent plane.
    zmins = []
    zmaxs = []
    for alias in dense_aliases:
        bbox_min, bbox_max = pf.ops.attr.bbox_min_max(alias, global_coords=True)
        zmins.append(float(bbox_min[2]))
        zmaxs.append(float(bbox_max[2]))
    assert max(zmaxs) > min(zmins) + 0.3
