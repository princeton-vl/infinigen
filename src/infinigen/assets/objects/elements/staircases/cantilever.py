# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np
import shapely
import shapely.affinity

from infinigen.assets.objects.elements.staircases.straight import (
    StraightStaircaseFactory,
)
from infinigen.assets.utils.decorate import read_co
from infinigen.assets.utils.object import join_objects
from infinigen.core.util import blender as butil


class CantileverStaircaseFactory(StraightStaircaseFactory):
    support_types = "wall"
    handrail_types = "weighted_choice", (2, "horizontal-post"), (2, "vertical-post")

    def valid_contour(self, offset, contour, doors, lower=True):
        valid = super().valid_contour(offset, contour, doors, lower)
        if not valid or not lower:
            return valid
        obj = join_objects([self.make_line_offset(0), self.make_line_offset(1)])
        co = read_co(obj)[:, :-1]
        butil.delete(obj)
        if self.mirror:
            co[:, 0] = -co[:, 0]
        points = [
            shapely.affinity.translate(
                shapely.affinity.rotate(p, self.rot_z, (0, 0)), *offset
            )
            for p in shapely.points(co)
        ]
        others = [shapely.ops.nearest_points(p, contour.boundary)[0] for p in points]
        distance = np.array(
            [np.abs(p.x - o.x) + np.abs(p.y - o.y) for p, o in zip(points, others)]
        )
        return (distance < 0.1).sum() / len(distance) > 0.5
