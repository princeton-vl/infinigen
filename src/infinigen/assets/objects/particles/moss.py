# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.


# Authors: Lingjie Mei
import math

from numpy.random import uniform as U

from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import new_cube
from infinigen.core import surface
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util.color import hsv2rgba


class MossFactory(AssetFactory):
    def __init__(self, factory_seed):
        super(MossFactory, self).__init__(factory_seed)
        self.max_polygon = 1e4
        self.base_hue = U(0.2, 0.24)

    @staticmethod
    def shader_moss(nw: NodeWrangler, base_hue=0.3):
        h_perturb = U(-0.02, 0.02)
        s_perturb = U(-0.1, -0.0)
        v_perturb = U(1.0, 1.5)

        def map_perturb(h, s, v):
            return hsv2rgba(h + h_perturb, s + s_perturb, v / v_perturb)

        subsurface_ratio = 0.05
        roughness = 1.0
        mix_ratio = 0.2

        cr = build_color_ramp(
            nw,
            nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 5.0}).outputs["Fac"],
            [0, 0.5, 1],
            [
                map_perturb(base_hue, 0.8, 0.1),
                map_perturb(base_hue - 0.05, 0.8, 0.1),
                (0.0, 0.0, 0.0, 1.0),
            ],
        )

        background = map_perturb(base_hue, 0.8, 0.02)
        mix_rgb = nw.new_node(
            Nodes.MixRGB,
            [
                nw.new_node(Nodes.ObjectInfo_Shader).outputs["Random"],
                cr.outputs["Color"],
                background,
            ],
        )

        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": mix_rgb,
                "Subsurface Weight": subsurface_ratio,
                "Subsurface Radius": (0.01, 0.01, 0.01),
                "Subsurface Color": background,
                "Roughness": roughness,
            },
        )

        translucent_bsdf = nw.new_node(
            Nodes.TranslucentBSDF, input_kwargs={"Color": mix_rgb}
        )

        mix_shader = nw.new_node(
            Nodes.MixShader, [mix_ratio, principled_bsdf, translucent_bsdf]
        )
        return mix_shader

    def create_asset(self, face_size=0.01, **params):
        obj = new_cube()
        surface.add_geomod(
            obj, self.geo_moss_instance, apply=True, input_args=[face_size]
        )
        assign_material(
            obj,
            surface.shaderfunc_to_material(
                MossFactory.shader_moss, (self.base_hue + U(-0.02, 0.02) % 1)
            ),
        )
        tag_object(obj, "moss")
        return obj

    @staticmethod
    def geo_moss_instance(nw: NodeWrangler, face_size):
        radius = 0.008
        start = (0.0, 0.0, 0.0)
        start_handle = (-0.03, 0.0, 0.02)
        end = (-0.04, 0.0, U(0.04, 0.05))
        end_handle = (end[0] + U(-0.03, -0.02), 0.0, end[2] + U(-0.01, 0.0))
        bezier = nw.new_node(
            Nodes.CurveBezierSegment,
            input_kwargs={
                "Resolution": 10 * math.ceil(0.01 / face_size),
                "Start": start,
                "Start Handle": start_handle,
                "End Handle": end_handle,
                "End": end,
            },
        )
        circle = nw.new_node(
            Nodes.CurveCircle, input_kwargs={"Resolution": 4, "Radius": radius}
        ).outputs["Curve"]
        mesh = nw.curve2mesh(bezier, circle)
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": mesh})
