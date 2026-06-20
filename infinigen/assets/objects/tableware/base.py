# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import read_co, write_attribute
from infinigen.assets.utils.misc import assign_material
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


def sample_tableware_base(seed: int) -> dict[str, Any]:
    """Sample init-scoped tableware base parameters under FixedSeed."""
    with FixedSeed(seed):
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        thickness = 0.01
        return {
            "thickness": thickness,
            "surface": weighted_sample(material_assignments.cup)()(),
            "inside_surface": weighted_sample(material_assignments.cup)()(),
            "guard_surface": weighted_sample(material_assignments.woods)()(),
            "scratch_draw": scratch_draw,
            "edge_wear_draw": edge_wear_draw,
            "scratch_prob": scratch_prob,
            "edge_wear_prob": edge_wear_prob,
            "scratch_fn": scratch_fn,
            "edge_wear_fn": edge_wear_fn,
            "guard_depth": thickness,
            "has_guard": False,
            "has_inside": False,
            "lower_thresh": uniform(0.5, 0.8),
            "scale": 1.0,
            "metal_color": "bw+natural",
        }


def apply_tableware_base(factory: "TablewareFactory", params: Any) -> None:
    if hasattr(params, "thickness"):
        factory.thickness = params.thickness
    elif hasattr(params, "thickness_ratio"):
        factory.thickness = params.thickness_ratio * params.scale
    factory.surface = params.surface
    factory.inside_surface = params.inside_surface
    factory.guard_surface = params.guard_surface
    factory.scratch = params.scratch
    factory.edge_wear = params.edge_wear
    if hasattr(params, "guard_depth"):
        factory.guard_depth = params.guard_depth
    elif hasattr(params, "guard_depth_mult"):
        factory.guard_depth = params.guard_depth_mult * factory.thickness
    else:
        factory.guard_depth = factory.thickness
    if hasattr(params, "has_guard"):
        factory.has_guard = params.has_guard
    if hasattr(params, "has_inside"):
        factory.has_inside = params.has_inside
    else:
        factory.has_inside = False
    factory.lower_thresh = params.lower_thresh
    factory.scale = params.scale
    factory.metal_color = getattr(params, "metal_color", "bw+natural")


class TablewareFactory(AssetFactory):
    is_fragile = False
    allow_transparent = False

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        if isinstance(self, ParameterizedAssetFactory):
            return
        with FixedSeed(factory_seed):
            self._init_tableware_base()

    def _init_tableware_base(self) -> None:
        base = sample_tableware_base(self.factory_seed)
        self.thickness = base["thickness"]
        self.surface = base["surface"]
        self.inside_surface = base["inside_surface"]
        self.guard_surface = base["guard_surface"]
        self.scratch = (
            None
            if base["scratch_draw"] > base["scratch_prob"]
            else base["scratch_fn"]()
        )
        self.edge_wear = (
            None
            if base["edge_wear_draw"] > base["edge_wear_prob"]
            else base["edge_wear_fn"]()
        )
        self.guard_depth = base["guard_depth"]
        self.has_guard = base["has_guard"]
        self.has_inside = base["has_inside"]
        self.lower_thresh = base["lower_thresh"]
        self.scale = base["scale"]
        self.metal_color = base["metal_color"]

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
        surface.assign_material(assets, self.surface)
        if self.has_inside:
            surface.assign_material(assets, self.inside_surface, selection="inside")
        if self.has_guard:
            surface.assign_material(assets, self.guard_surface, selection="guard")
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
