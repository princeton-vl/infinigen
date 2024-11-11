# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials import fabrics
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring
from infinigen.assets.utils.object import new_bbox, new_cube
from infinigen.core import surface
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


def make_coiled(obj, dot_distance, dot_depth, dot_size):
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.poke()
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.mesh.poke()
        bpy.ops.mesh.poke()
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        for v in bm.verts:
            if len(v.link_edges) == 16:
                v.select_set(True)
        bm.select_flush(False)
        bmesh.update_edit_mesh(obj.data)
        radius = dot_distance * uniform(0.06, 0.08)
        bpy.ops.mesh.bevel(offset=radius, affect="VERTICES")
        bpy.ops.mesh.extrude_region_shrink_fatten(
            TRANSFORM_OT_shrink_fatten={"value": -dot_depth}
        )
        bpy.ops.mesh.extrude_region_shrink_fatten(
            TRANSFORM_OT_shrink_fatten={"value": dot_depth}
        )
        bpy.ops.mesh.select_more()
        bpy.ops.mesh.select_more()
    write_attr_data(obj, "tip", np.zeros(len(obj.data.polygons)), domain="FACE")
    with butil.ViewportMode(obj, "EDIT"):
        surface.set_active(obj, "tip")
        bpy.ops.mesh.attribute_set(value_float=1)

    def geo_scale(nw: NodeWrangler):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        selection = nw.new_node(Nodes.NamedAttribute, ["tip"])
        geometry = nw.new_node(
            Nodes.ScaleElements,
            [geometry, selection, nw.combine(*([dot_size / radius] * 3))],
        )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    surface.add_geomod(obj, geo_scale, apply=True)
    butil.modify_mesh(obj, "TRIANGULATE", min_vertices=4)
    butil.modify_mesh(obj, "SMOOTH", factor=uniform(0.5, 1.0), iterations=5)


class MattressFactory(AssetFactory):
    types = "weighted_choice", (1, "coiled"), (1, "wrapped")

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = log_uniform(0.9, 2.0)
            self.size = uniform(2, 2.4)
            self.thickness = uniform(0.2, 0.35)
            self.dot_distance = log_uniform(0.16, 0.2)
            self.dot_size = uniform(0.005, 0.02)
            self.dot_depth = uniform(0.04, 0.08)
            self.wrap_distance = 0.05
            self.surface = fabrics.fabric_random
            self.type = rg(self.types)
            materials = AssetList["MattressFactory"]()
            self.surface = materials["surface"].assign_material()

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.width / 2,
            self.width / 2,
            -self.size / 2,
            self.size / 2,
            -self.thickness / 2,
            self.thickness / 2,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_cube()
        obj.scale = self.width / 2, self.size / 2, self.thickness / 2
        butil.apply_transform(obj)
        match self.type:
            case "coiled":
                self.make_coiled(obj)
            case "wrapped":
                self.make_wrapped(obj)
        return obj

    def make_coiled(self, obj):
        for i, size in enumerate(obj.dimensions):
            axis = np.zeros(3)
            axis[i] = 1
            subdivide_edge_ring(obj, int(np.ceil(size / self.dot_distance)), axis)
        make_coiled(obj, self.dot_distance, self.dot_depth, self.dot_size)

    def make_wrapped(self, obj):
        for i, size in enumerate([self.width, self.size, self.thickness]):
            axis = np.zeros(3)
            axis[i] = 1
            subdivide_edge_ring(obj, int(np.ceil(size / self.wrap_distance)), axis)
        butil.modify_mesh(obj, "BEVEL", width=self.wrap_distance / 3, segments=2)
        vg = obj.vertex_groups.new(name="pin")
        vg.add(
            np.nonzero((read_co(obj)[:, -1] < 1e-1 - self.thickness / 2))[0].tolist(),
            1,
            "REPLACE",
        )
        clothes.cloth_sim(
            obj,
            gravity=0,
            use_pressure=True,
            uniform_pressure_force=uniform(0.1, 0.2),
            vertex_group_mass="pin",
        )

    def finalize_assets(self, assets):
        self.surface.apply(assets)
