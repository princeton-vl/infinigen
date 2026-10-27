# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen Laplacian coral growth (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/corals/laplacian.py)
# - David Yan, Jia Deng: optimized implementation

"""Cauliflower coral subdivided on arrays in place of the Subdivision Surface modifier.

The modifier is slow on the coral's marching-cubes surface, where every vertex is
extraordinary for its adaptive patch tables.
"""

import bpy
import numpy as np

import infinigen.core.util.blender as butil
from infinigen.assets.objects.corals import laplacian as _default
from infinigen.assets.utils.decorate import geo_extension
from infinigen.assets.utils.object import mesh2obj
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.perf.assets.objects.corals.subdivide import catmull_clark_limit
from infinigen.perf.assets.objects.monocot.leaf_direct import mesh_from_arrays


def subdivide_limit(obj):
    """Subdivide `obj` one level onto the limit surface, like the Subdivision Surface
    modifier, which still runs for meshes that are not closed triangle manifolds."""
    mesh = obj.data
    nv, nf, ne = len(mesh.vertices), len(mesh.polygons), len(mesh.edges)
    co = np.empty(nv * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    totals = np.empty(nf, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    if nf == 0 or totals.min() != 3 or totals.max() != 3:
        butil.modify_mesh(obj, "SUBSURF", levels=1, render_levels=1)
        return obj
    faces = np.empty(nf * 3, dtype=np.int32)
    mesh.polygons.foreach_get("vertices", faces)
    edges = np.empty(ne * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    try:
        verts, quads = catmull_clark_limit(
            co.reshape(nv, 3), faces.reshape(nf, 3), edges.reshape(ne, 2)
        )
    except ValueError:
        butil.modify_mesh(obj, "SUBSURF", levels=1, render_levels=1)
        return obj
    new = mesh_from_arrays(verts, quads, name=mesh.name)
    if "sharp_face" in mesh.attributes:
        sharp = np.empty(nf, dtype=np.bool_)
        mesh.attributes["sharp_face"].data.foreach_get("value", sharp)
        new.attributes.new("sharp_face", "BOOLEAN", "FACE").data.foreach_set(
            "value", np.repeat(sharp, 3)
        )
    for m in mesh.materials:
        new.materials.append(m)
    obj.data = new
    bpy.data.meshes.remove(mesh)
    # refresh the object's bounds, which the coral's scaling reads, as applying
    # the modifier would
    bpy.context.view_layer.update()
    return obj


class CauliflowerBaseCoralFactory:
    """Counterpart of the default factory, defining only the method that differs."""

    def create_asset(self, face_size=0.01, **params):
        mesh = _default.build_laplacian_3d()
        obj = mesh2obj(mesh)
        surface.add_geomod(obj, geo_extension, apply=True)
        subdivide_limit(obj)
        tag_object(obj, "cauliflower_coral")
        return obj
