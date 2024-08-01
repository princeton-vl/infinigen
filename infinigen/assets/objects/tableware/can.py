# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
import shapely
from numpy.random import uniform
from shapely import Point, affinity

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import write_co
from infinigen.assets.utils.object import join_objects, new_circle, new_cylinder
from infinigen.assets.utils.uv import wrap_four_sides
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class CanFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.x_length = log_uniform(0.05, 0.1)
            self.z_length = self.x_length * log_uniform(0.5, 2.5)
            self.shape = np.random.choice(["circle", "rectangle"])
            self.skewness = uniform(1, 2.5) if uniform() < 0.5 else 1

            material_assignments = AssetList["CanFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.wrap_surface = material_assignments["wrap_surface"].assign_material()
            if self.wrap_surface == text.Text:
                self.wrap_surface = text.Text(self.factory_seed, False)

            scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
            self.scratch, self.edge_wear = material_assignments["wear_tear"]
            self.scratch = None if uniform() > scratch_prob else self.scratch
            self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

            self.texture_shared = uniform() < 0.2

    def create_asset(self, **params) -> bpy.types.Object:
        coords = self.make_coords()
        obj = new_circle(vertices=len(coords))
        write_co(obj, np.array([[x, y, 0] for x, y in coords]))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.edge_face_add()
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.z_length)
        surface.add_geomod(obj, self.geo_cap, apply=True)
        self.surface.apply(obj)
        wrap = self.make_wrap(coords)
        obj = join_objects([obj, wrap])
        return obj

    @staticmethod
    def geo_cap(nw: NodeWrangler):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        selection = nw.compare(
            "GREATER_THAN",
            nw.math("ABSOLUTE", nw.separate(nw.new_node(Nodes.InputNormal))[-1]),
            1 - 1e-3,
        )
        geometry, top = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, selection, None, 0]
        ).outputs[:2]
        geometry = nw.new_node(
            Nodes.ScaleElements,
            input_kwargs={
                "Geometry": geometry,
                "Selection": top,
                "Scale": uniform(0.96, 0.98),
            },
        )
        geometry = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, None, -uniform(0.005, 0.01)]
        ).outputs[0]
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def make_coords(self):
        match self.shape:
            case "circle":
                p = Point(0, 0).buffer(self.x_length, quad_segs=64)
            case _:
                side = self.x_length * uniform(0.2, 0.8)
                p = shapely.box(-side, -side, side, side).buffer(
                    self.x_length - side, quad_segs=16
                )
        p = affinity.scale(p, yfact=1 / self.skewness)
        coords = p.boundary.segmentize(0.01).coords[:][:-1]
        return coords

    def make_wrap(self, coords):
        obj = new_cylinder(vertices=len(coords))
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)
        lowest, highest = (
            self.z_length * uniform(0, 0.1),
            self.z_length * uniform(0.9, 1.0),
        )
        write_co(
            obj,
            np.concatenate(
                [np.array([[x, y, lowest], [x, y, highest]]) for x, y in coords]
            ),
        )
        obj.scale = 1 + 1e-3, 1 + 1e-3, 1
        butil.apply_transform(obj)
        wrap_four_sides(obj, self.wrap_surface, self.texture_shared)
        return obj
