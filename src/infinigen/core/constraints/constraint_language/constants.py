# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei

import gin
import numpy as np
import shapely
from shapely.ops import orient

from infinigen.assets.utils.shapes import (
    is_valid_polygon,
    segment_filter,
    simplify_polygon,
)
from infinigen.core import tags as t
from infinigen.core.util.random import random_general as rg


@gin.configurable
class RoomConstants:
    def __init__(
        self,
        n_stories=("cat", 0.0, 0.5, 0.4, 0.1),
        room_type=None,
        aspect_ratio_range=(0.7, 1.0),
        fixed_contour=("bool", 0.5),
    ):
        self.n_stories = rg(n_stories)
        self.unit, self.segment_margin, self.wall_thickness, self.wall_height = (
            self.global_params().values()
        )
        self.door_width, self.door_margin, self.door_size = self.door_params().values()
        (
            self.max_window_length,
            self.window_height,
            self.window_margin,
            self.window_size,
            self.window_top,
        ) = self.window_params().values()
        self.staircase_snap, self.staircase_thresh = self.staircase_params().values()
        if room_type is None:
            self.room_types = self.home_room_types
        else:
            self.room_types = room_type
        self.aspect_ratio_range = aspect_ratio_range
        self.fixed_contour = rg(fixed_contour)

    @gin.configurable(module="RoomConstants")
    def global_params(
        self,
        unit=0.5,
        segment_margin=1.4,
        wall_thickness=("uniform", 0.2, 0.3),
        wall_height=("uniform", 2.8, 3.2),
    ):
        wall_thickness = rg(wall_thickness)
        wall_height = rg(wall_height)
        return {
            "unit": unit,
            "segment_margin": segment_margin,
            "wall_thickness": wall_thickness,
            "wall_height": wall_height,
        }

    def door_params(
        self, door_width_ratio=("uniform", 0.7, 0.8), door_size=("uniform", 2.0, 2.4)
    ):
        door_width = (self.segment_margin - self.wall_thickness) * rg(door_width_ratio)
        assert door_width > 0
        door_margin = (self.segment_margin - door_width) / 2
        door_size = rg(door_size)
        return {
            "door_width": door_width,
            "door_margin": door_margin,
            "door_size": door_size,
        }

    def window_params(
        self,
        max_window_length=("uniform", 6, 8),
        window_height=("uniform", 0.8, 1.2),
        window_margin=("uniform", 0.2, 0.25),
        window_size=("uniform", 1.0, 1.5),
    ):
        max_window_length = rg(max_window_length)
        window_height = rg(window_height)
        window_size = rg(window_size)
        window_margin = rg(window_margin)
        window_top = (
            self.wall_height - self.wall_thickness - window_height - window_size
        )
        window_top = max(self.wall_thickness / 2, window_top)
        window_size = (
            self.wall_height - self.wall_thickness - window_top - window_height
        )
        assert window_size > 0
        return {
            "max_window_length": max_window_length,
            "window_height": window_height,
            "window_margin": window_margin,
            "window_size": window_size,
            "window_top": window_top,
        }

    def staircase_params(
        self,
    ):
        return {"staircase_snap": 1.2, "staircase_thresh": 0.6}

    def unit_cast(self, x):
        x = np.round(x / self.unit) * self.unit
        if x.size == 1:
            return x.item()
        return x

    def canonicalize(self, p):
        p = p.buffer(0)
        try:
            while True:
                p_ = shapely.force_2d(simplify_polygon(p))
                if p.area == 0:
                    raise NotImplementedError("Polygon empty.")
                p = orient(p_)
                coords = np.array(
                    p.boundary.coords[:]
                    if not hasattr(p.boundary, "geoms")
                    else p.exterior.coords[:]
                )
                l = len(coords)
                rounded = np.round(coords / self.unit) * self.unit
                coords = np.where(
                    np.all(np.abs(coords - rounded) < 1e-3, -1)[:, np.newaxis],
                    rounded,
                    coords,
                )
                diff = coords[1:] - coords[:-1]
                diff = diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6)
                product = (diff[[-1] + list(range(len(diff) - 1))] * diff).sum(-1)
                valid_indices = list(range(len(coords) - 1))
                invalid_indices = np.nonzero((product < -0.8) | (product > 1 - 1e-6))[
                    0
                ].tolist()
                if len(invalid_indices) > 0:
                    i = invalid_indices[len(invalid_indices) // 2]
                    valid_indices.remove(i)
                p = shapely.Polygon(coords[valid_indices + [valid_indices[0]]])
                if len(p.exterior.coords) == l:
                    break
            if not is_valid_polygon(p):
                raise NotImplementedError("Invalid polygon")
            return orient(p)
        except AttributeError:
            raise NotImplementedError("Invalid multi polygon")

    def filter(self, ses, margin=None):
        margin = self.segment_margin if margin is None else margin
        return list(l for l, se in ses.items() if segment_filter(se, margin))

    @property
    def home_room_types(self):
        return {
            t.Semantics.Kitchen,
            t.Semantics.Bedroom,
            t.Semantics.LivingRoom,
            t.Semantics.Closet,
            t.Semantics.Hallway,
            t.Semantics.Bathroom,
            t.Semantics.Garage,
            t.Semantics.Balcony,
            t.Semantics.DiningRoom,
            t.Semantics.Utility,
            t.Semantics.StaircaseRoom,
        }

    @property
    def floors(self):
        return [
            t.Semantics.GroundFloor,
            t.Semantics.SecondFloor,
            t.Semantics.ThirdFloor,
        ]
