# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.elements.doors.base import BaseDoorFactory
from infinigen.assets.utils.decorate import read_area, select_faces, write_attribute
from infinigen.assets.utils.mesh import prepare_for_boolean
from infinigen.assets.utils.object import new_cube
from infinigen.core.surface import read_attr_data, write_attr_data
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class PanelDoorFactory(BaseDoorFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(PanelDoorFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            self.x_subdivisions = 1 if uniform() < 0.5 else 2
            self.y_subdivisions = np.clip(np.random.binomial(5, 0.45), 1, None)

    def bevel(self, obj, panel):
        x_min, x_max, y_min, y_max = panel["dimension"]
        assert x_min <= x_max and y_min <= y_max
        cutter = new_cube()
        butil.apply_transform(cutter, loc=True)
        if panel["attribute_name"] is not None:
            write_attribute(cutter, 1, panel["attribute_name"], "FACE")
        cutter.location = (
            (x_max + x_min) / 2,
            self.bevel_width * 0.5 - 0.1,
            (y_max + y_min) / 2,
        )
        cutter.scale = (x_max - x_min) / 2 - 2e-3, 0.1, (y_max - y_min) / 2 - 2e-3
        butil.apply_transform(cutter, loc=True)
        # butil.modify_mesh(cutter, 'BEVEL', width=self.bevel_width)
        write_attr_data(
            cutter, "cut", np.ones(len(cutter.data.polygons), dtype=int), "INT", "FACE"
        )
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        prepare_for_boolean(obj)
        cutter.location[1] += 0.2 + self.depth - self.bevel_width
        butil.apply_transform(cutter, loc=True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        prepare_for_boolean(obj)
        butil.delete(cutter)
        select_faces(obj, (read_area(obj) > 0.01) & (read_attr_data(obj, "cut")))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.inset(thickness=self.shrink_width)
            bpy.ops.mesh.inset(thickness=self.bevel_width, depth=self.bevel_width)
        obj.data.attributes.remove(obj.data.attributes["cut"])
        return []

    def make_panels(self):
        panels = []
        x_cuts = np.random.randint(1, 4, self.x_subdivisions)
        x_cuts = np.cumsum(x_cuts / x_cuts.sum())
        y_cuts = np.sort(np.random.randint(2, 5, self.y_subdivisions))[::-1]
        y_cuts = np.cumsum(y_cuts / y_cuts.sum())
        for j in range(len(y_cuts)):
            for i in range(len(x_cuts)):
                x_min = self.panel_margin + (self.width - self.panel_margin) * (
                    x_cuts[i - 1] if i > 0 else 0
                )
                x_max = (self.width - self.panel_margin) * x_cuts[i]
                y_min = self.panel_margin + (self.height - self.panel_margin) * (
                    y_cuts[j - 1] if j > 0 else 0
                )
                y_max = (self.height - self.panel_margin) * y_cuts[j]
                panels.append(
                    {
                        "dimension": (x_min, x_max, y_min, y_max),
                        "func": self.bevel,
                        "attribute_name": None,
                    }
                )
        return panels


class GlassPanelDoorFactory(PanelDoorFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(GlassPanelDoorFactory, self).__init__(factory_seed, coarse, constants)
        with FixedSeed(self.factory_seed):
            self.x_subdivisions = 2
            self.y_subdivisions = np.clip(np.random.binomial(5, 0.5), 2, None)
            self.merge_glass = self.y_subdivisions < 4
            self.has_glass = True

    def make_panels(self):
        panels = super(GlassPanelDoorFactory, self).make_panels()
        if self.merge_glass:
            first_dimension = panels[-self.x_subdivisions]["dimension"]
            last_dimension = panels[-1]["dimension"]
            merged = {
                "dimension": (
                    first_dimension[0],
                    last_dimension[1],
                    first_dimension[2],
                    last_dimension[3],
                ),
                "func": self.bevel,
                "attribute_name": "glass",
            }
            return [merged, *panels[: self.x_subdivisions]]
        else:
            for panel in panels[-self.x_subdivisions :]:
                panel["attribute_name"] = "glass"
            return panels
