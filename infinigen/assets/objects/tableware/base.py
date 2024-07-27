# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import read_co, write_attribute
from infinigen.assets.utils.misc import assign_material
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class TablewareFactory(AssetFactory):
    is_fragile = False
    allow_transparent = False

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.thickness = 0.01
            material_assignments = AssetList["TablewareFactory"](
                fragile=self.is_fragile, transparent=self.allow_transparent
            )

            self.surface = material_assignments["surface"].assign_material()
            self.inside_surface = material_assignments["inside"].assign_material()
            self.guard_surface = material_assignments["guard"].assign_material()

            scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
            self.scratch, self.edge_wear = material_assignments["wear_tear"]

            self.scratch = None if uniform() > scratch_prob else self.scratch
            self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

            self.guard_depth = self.thickness
            self.has_guard = False
            self.has_inside = False
            self.lower_thresh = uniform(0.5, 0.8)
            self.scale = 1.0
            self.metal_color = "bw+natural"

    def create_asset(self, **params) -> bpy.types.Object:
        raise NotImplementedError

    def add_guard(self, obj, selection):
        if not self.has_guard:
            selection = False

        def geo_guard(nw: NodeWrangler):
            geometry = nw.new_node(
                Nodes.GroupInput,
                expose_input=[("NodeSocketGeometry", "Geometry", None)],
            )
            normal = nw.new_node(Nodes.InputNormal)
            x = nw.separate(nw.new_node(Nodes.InputPosition))[0]
            sel = surface.eval_argument(nw, selection, x=x, normal=normal)
            geometry, top, side = nw.new_node(
                Nodes.ExtrudeMesh,
                input_args=[geometry, sel, None, self.guard_depth, False],
            ).outputs[:3]
            guard = nw.boolean_math("OR", top, side)
            geometry = nw.new_node(
                Nodes.StoreNamedAttribute,
                input_kwargs={"Geometry": geometry, "Name": "guard", "Value": guard},
                attrs={"domain": "FACE"},
            )
            nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

        surface.add_geomod(obj, geo_guard, apply=True)

    @staticmethod
    def make_double_sided(selection):
        return lambda nw, x, normal: nw.boolean_math(
            "AND",
            surface.eval_argument(nw, selection, x=x, normal=normal),
            nw.compare(
                "GREATER_THAN", nw.math("ABSOLUTE", nw.separate(normal)[-1]), 0.8
            ),
        )

    def finalize_assets(self, assets):
        assign_material(assets, [])
        self.surface.apply(assets, metal_color=self.metal_color)
        if self.has_inside:
            self.inside_surface.apply(
                assets, selection="inside", clear=True, metal_color="bw+natural"
            )
        if self.has_guard:
            self.guard_surface.apply(
                assets, selection="guard", metal_color=self.metal_color
            )
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)

    def solidify_with_inside(self, obj, thickness):
        max_z = np.max(read_co(obj)[:, -1])
        obj.vertex_groups.new(name="inside_")
        butil.modify_mesh(
            obj, "SOLIDIFY", thickness=thickness, offset=1, shell_vertex_group="inside_"
        )
        write_attribute(obj, "inside_", "inside", "FACE")

        def inside(nw: NodeWrangler):
            lower = nw.compare(
                "LESS_THAN",
                nw.separate(nw.new_node(Nodes.InputPosition))[-1],
                max_z * self.lower_thresh,
            )
            inside = nw.compare(
                "GREATER_THAN", surface.eval_argument(nw, "inside"), 0.8
            )
            return nw.boolean_math("AND", inside, lower)

        write_attribute(obj, inside, "lower_inside", "FACE")
        obj.vertex_groups.remove(obj.vertex_groups["inside_"])
