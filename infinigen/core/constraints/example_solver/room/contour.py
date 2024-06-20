# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import random

import gin
import numpy as np
from numpy.random import uniform
from shapely import Polygon, box

from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.object import new_plane
from infinigen.core.constraints.example_solver.room.configs import (
    TYPICAL_AREA_ROOM_TYPES,
)
from infinigen.core.constraints.example_solver.room.types import RoomType
from infinigen.core.constraints.example_solver.room.utils import unit_cast
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

LARGE = 100


@gin.configurable(denylist=["width", "height"])
class ContourFactory:
    def __init__(self, width=17, height=9):
        self.width = width
        self.height = height
        self.n_trials = 1000

    def make_contour(self, i):
        with FixedSeed(i):
            obj = new_plane()
            obj.location = self.width / 2, self.height / 2, 0
            obj.scale = self.width / 2, self.height / 2, 1
            butil.apply_transform(obj, loc=True)
            corners = list(
                (x, y)
                for x in [0, unit_cast(self.width)]
                for y in [0, unit_cast(self.height)]
            )
            random.shuffle(corners)
            corners = dict(enumerate(corners))

            def nearest(t):
                if len(corners) == 0:
                    return -1, np.inf
                c = np.array(list(corners.values()))
                dist = np.abs(c - np.array([[t[0], t[1]]])).sum(1)
                return list(corners.keys())[np.argmin(dist)], np.min(dist)

            while len(corners) > 0:
                _, (x, y) = corners.popitem()
                r = uniform(0, 1)
                if r < 0.2:
                    axes = []
                    if nearest((self.width - x, y))[1] < 0.1:
                        axes.append(0)
                    elif nearest((x, self.height - y))[1] < 0.1:
                        axes.append(1)
                    if len(axes) > 0:
                        axis = np.random.choice(axes)
                        self.add_long_corner(obj, x, y, axis)
                        t = (self.width - x, y) if axis == 0 else (x, self.height - y)
                        corners.pop(nearest(t)[0])
                elif r < 0.35:
                    self.add_round_corner(obj, x, y)
                elif r < 0.5:
                    self.add_straight_corner(obj, x, y)
                elif r < 0.65:
                    self.add_sharp_corner(obj, x, y)

            vertices = obj.data.polygons[0].vertices
            p = Polygon(read_co(obj)[:, :2][vertices])
            butil.delete(obj)
            return p

    def add_round_corner(self, obj, x, y):
        vg = obj.vertex_groups.new(name="corner")
        for i, v in enumerate(obj.data.vertices):
            vg.add([i], v.co[0] == x and v.co[1] == y, "REPLACE")
        width = unit_cast(uniform(0.2, 0.3) * min(self.width, self.height))
        try:
            butil.modify_mesh(
                obj,
                "BEVEL",
                affect="VERTICES",
                limit_method="VGROUP",
                vertex_group="corner",
                segments=np.random.randint(2, 5),
                width=width,
            )
        except Exception:
            pass
        obj.vertex_groups.remove(obj.vertex_groups["corner"])

    def add_straight_corner(self, obj, x, y):
        vg = obj.vertex_groups.new(name="corner")
        for i, v in enumerate(obj.data.vertices):
            vg.add([i], v.co[0] == x and v.co[1] == y, "REPLACE")
        width = unit_cast(uniform(0.1, 0.3) * min(self.width, self.height))
        if width > 0:
            butil.modify_mesh(
                obj,
                "BEVEL",
                affect="VERTICES",
                limit_method="VGROUP",
                vertex_group="corner",
                segments=1,
                width=width,
            )
        obj.vertex_groups.remove(obj.vertex_groups["corner"])

    def add_sharp_corner(self, obj, x, y):
        cutter = new_plane(size=LARGE)
        butil.modify_mesh(cutter, "SOLIDIFY", offset=0, thickness=1)
        x_ratio, y_ratio = uniform(0.1, 0.3, 2)
        cutter.location = (
            x + (LARGE / 2 - unit_cast(x_ratio * self.width)) * (-1) ** (x <= 0),
            y + (LARGE / 2 - unit_cast(y_ratio * self.height)) * (-1) ** (y <= 0),
            0,
        )
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete(cutter)

    def add_long_corner(self, obj, x, y, axis):
        x_, y_, z_ = read_co(obj).T
        i = np.nonzero((x_ == x) & (y_ == y))[0]
        if axis == 0:
            y_[i] -= self.height * uniform(0.1, 0.3) * (-1) ** (y_[i] <= 0)
        else:
            x_[i] -= self.width * uniform(0.1, 0.3) * (-1) ** (x_[i] <= 0)
        write_co(obj, np.stack([x_, y_, z_], -1))

    def add_staircase(self, contour):
        x, y = contour.boundary.xy
        x_, x__ = np.min(x), np.max(x)
        y_, y__ = np.min(y), np.max(y)
        for _ in range(self.n_trials):
            area = TYPICAL_AREA_ROOM_TYPES[RoomType.Staircase] * uniform(1.4, 1.6)
            skewness = log_uniform(0.6, 0.8)
            if uniform() < 0.5:
                skewness = 1 / skewness
            width, height = (
                unit_cast(np.sqrt(area * skewness).item()),
                unit_cast(np.sqrt(area / skewness).item()),
            )
            x = unit_cast(uniform(x_, x__ - width))
            y = unit_cast(uniform(y_, y__ - height))
            b = box(x, y, x + width, y + height)
            if contour.contains(b):
                return b
        else:
            raise ValueError("Invalid staircase")
