# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.cactus.base import BaseCactusFactory
from infinigen.assets.utils.decorate import geo_extension
from infinigen.assets.utils.object import new_cube
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


class GlobularBaseCactusFactory(BaseCactusFactory):
    spike_distance = 0.08

    @staticmethod
    def geo_globular(nw: NodeWrangler):
        star_resolution = np.random.randint(6, 12)
        resolution = 64
        frequency = uniform(-0.2, 0.2)
        circle = nw.new_node(Nodes.MeshCircle, [star_resolution * 3])
        selection = nw.compare(
            "EQUAL", nw.math("MODULO", nw.new_node(Nodes.Index), 2), 0
        )
        circle, selection = nw.new_node(
            Nodes.CaptureAttribute, [circle, selection]
        ).outputs[:2]
        circle = nw.new_node(
            Nodes.SetPosition,
            [
                circle,
                selection,
                nw.scale(nw.new_node(Nodes.InputPosition), uniform(1.1, 1.2)),
            ],
        )
        profile_curve = nw.new_node(Nodes.MeshToCurve, [circle])
        curve = nw.new_node(
            Nodes.ResampleCurve, [nw.new_node(Nodes.CurveLine), None, resolution]
        )
        anchors = [
            (0, uniform(0.2, 0.4)),
            (uniform(0.4, 0.6), log_uniform(0.5, 0.8)),
            (uniform(0.8, 0.85), uniform(0.4, 0.6)),
            (1.0, 0.05),
        ]
        radius = nw.scalar_multiply(
            nw.build_float_curve(nw.new_node(Nodes.SplineParameter), anchors, "AUTO"),
            log_uniform(0.5, 1.0),
        )
        curve = nw.new_node(Nodes.SetCurveRadius, [curve, None, radius])
        curve = nw.new_node(
            Nodes.SetCurveTilt,
            [
                curve,
                None,
                nw.scalar_multiply(
                    nw.new_node(Nodes.SplineParameter), 2 * np.pi * frequency
                ),
            ],
        )
        geometry = nw.curve2mesh(curve, profile_curve)
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": geometry, "Selection": selection},
        )

    def create_asset(self, face_size=0.01, **params) -> bpy.types.Object:
        obj = new_cube()
        surface.add_geomod(obj, self.geo_globular, apply=True, attributes=["selection"])
        surface.add_geomod(
            obj, geo_extension, apply=True, input_kwargs={"musgrave_dimensions": "2D"}
        )

        obj.scale = uniform(0.8, 1.5, 3)
        obj.rotation_euler[-1] = uniform(0, np.pi * 2)
        butil.apply_transform(obj)

        return obj
