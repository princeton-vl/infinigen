# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
import shapely
import shapely.affinity
from numpy.random import uniform

from infinigen.assets.materials import metal, plastic
from infinigen.assets.materials.woods import wood
from infinigen.assets.utils.decorate import (
    read_edge_center,
    read_edge_direction,
    select_edges,
)
from infinigen.assets.utils.object import join_objects, new_bbox, new_bbox_2d
from infinigen.assets.utils.shapes import polygon2obj
from infinigen.core import tagging as t
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.tags import Subpart
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class WallShelfFactory(AssetFactory):
    support_sides_ = (
        "weighted_choice",
        (0.5, "none"),
        (1, "bottom"),
        (1, "top"),
        (1.5, "both"),
    )
    support_margins = "weighted_choice", (2, 0), (1, ("uniform", 0.0, 0.2))
    support_ratios = "weighted_choice", (2, 1), (1, ("uniform", 0.5, 0.9))
    support_alphas = (
        "weighted_choice",
        (1, 1),
        (
            1,
            (
                "weighted_choice",
                (1, ("log_uniform", 0.4, 0.7)),
                (2, ("log_uniform", 1.5, 3)),
                (1, 10),
            ),
        ),
    )
    support_joins = "mitre", "round", "bevel"
    plate_bevels = "weighted_choice", (1, "none"), (1, "front"), (1, "side")

    plate_surfaces = "weighted_choice", (2, wood), (1, metal)
    support_surfaces = "weighted_choice", (2, metal), (1, wood), (2, plastic)

    def __init__(self, factory_seed, coarse=False):
        super(WallShelfFactory, self).__init__(factory_seed, coarse)
        self.support_side = rg(self.support_sides_)
        self.support_margin = rg(self.support_margins)
        if self.support_margin == 0:
            n_support = np.random.choice([2, 3, 4], p=[0.7, 0.2, 0.1])
        else:
            n_support = np.random.choice([2, 3], p=[0.8, 0.2])
        self.support_locs = np.linspace(
            -0.5 + self.support_margin, 0.5 - self.support_margin, n_support
        )
        self.length = log_uniform(0.3, 0.8)
        self.width = log_uniform(0.1, 0.2)
        match self.support_side:
            case "none":
                self.thickness = log_uniform(0.03, 0.08)
            case _:
                self.thickness = log_uniform(0.01, 0.05)
        self.support_width = log_uniform(0.01, 0.015)
        self.support_thickness = self.support_width * log_uniform(0.4, 1.0)
        self.support_length = self.width * uniform(0.7, 1.1)
        self.plate_bevel = rg(self.plate_bevels)
        self.support_join = np.random.choice(self.support_joins)
        self.plate_surface = rg(self.plate_surfaces)
        self.support_surface = rg(self.support_surfaces)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        box = new_bbox(
            0,
            self.width,
            -self.length / 2,
            self.length / 2,
            -self.support_length,
            self.support_length,
        )
        plane = new_bbox_2d(
            0, self.width, -self.length / 2, self.length / 2, self.thickness / 2
        )
        write_attr_data(
            plane,
            f"{t.PREFIX}{Subpart.SupportSurface.value}",
            np.ones(1).astype(bool),
            "INT",
            "FACE",
        )
        return join_objects([box, plane])

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_plate()
        self.plate_surface.apply(obj)
        if self.support_side != "none":
            support = self.make_support()
            supports = [support] + [
                deep_clone_obj(support) for _ in range(len(self.support_locs) - 1)
            ]
            for s, l in zip(supports, self.support_locs):
                s.location[1] = self.length * l
            self.support_surface.apply(supports)
            obj = join_objects([obj] + supports)
        return obj

    def make_plate(self):
        obj = new_bbox(
            0,
            self.width,
            -self.length / 2,
            self.length / 2,
            -self.thickness / 2,
            self.thickness / 2,
        )
        c = read_edge_center(obj)
        d = read_edge_direction(obj)
        front = (np.abs(d[:, 1]) > 0.5) & (c[:, 0] > 0.1)
        side = np.abs(d[:, 0]) > 0.5
        match self.plate_bevel:
            case "front":
                selection = front
            case "side":
                selection = front + side
            case _:
                selection = np.zeros_like(front)
        with butil.ViewportMode(obj, "EDIT"):
            select_edges(obj, selection)
            bpy.ops.mesh.bevel(
                offset=uniform(0.3, 0.5) * self.thickness,
                segments=np.random.randint(4, 9),
            )
        return obj

    def make_support_contour(self):
        l = shapely.LineString(np.array([(1, 0), (0, 0), (0, 1)]) * self.support_length)
        theta = np.linspace(0, np.pi / 2, 31)
        alpha = rg(self.support_alphas)
        r = 1 / ((np.cos(theta) + 1e-6) ** alpha + (np.sin(theta) + 1e-6) ** alpha) ** (
            1 / alpha
        )
        xy = r[:, np.newaxis] * np.stack([np.cos(theta), np.sin(theta)], -1)
        d = shapely.LineString(xy * self.support_length * rg(self.support_ratios))
        return shapely.union(l, d)

    def make_support(self):
        lines = []
        if self.support_side in ["top", "both"]:
            lines.append(self.make_support_contour())
        if self.support_side in ["bottom", "both"]:
            lines.append(
                shapely.affinity.scale(self.make_support_contour(), 1, -1, 1, (0, 0, 0))
            )

        contour = shapely.union_all(lines).buffer(
            self.support_thickness / 2, join_style=self.support_join
        )
        obj = polygon2obj(contour)
        obj.rotation_euler[0] = np.pi / 2
        obj.location = self.support_thickness / 2, -self.support_width / 2, 0
        butil.apply_transform(obj, True)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (0, self.support_width, 0)}
            )
        return obj
