# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import normal as N
from numpy.random import uniform

from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import data2mesh
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.infinigen_gpl.extras.diff_growth import build_diff_growth


class LichenFactory(AssetFactory):
    def __init__(self, factory_seed):
        super(LichenFactory, self).__init__(factory_seed)
        self.max_polygon = 1e4
        self.base_hue = uniform(0.15, 0.3)

    @staticmethod
    def build_lichen_circle_mesh(n):
        angles = polygon_angles(n)
        z_jitter = N(0.0, 0.02, n)
        r_jitter = np.exp(uniform(-0.2, 0.0, n))
        vertices = np.concatenate(
            [
                np.stack(
                    [np.cos(angles) * r_jitter, np.sin(angles) * r_jitter, z_jitter]
                ).T,
                np.zeros((1, 3)),
            ],
            0,
        )
        faces = np.stack([np.arange(n), np.roll(np.arange(n), 1), np.full(n, n)]).T
        mesh = data2mesh(vertices, [], faces, "circle")
        return mesh

    @staticmethod
    def shader_lichen(nw: NodeWrangler, base_hue=0.2, **params):
        h_perturb = uniform(-0.02, 0.02)
        s_perturb = uniform(-0.05, -0.0)
        v_perturb = uniform(1.0, 1.5)

        def map_perturb(h, s, v):
            return hsv2rgba(h + h_perturb, s + s_perturb, v / v_perturb)

        subsurface_ratio = 0.02
        roughness = 1.0

        cr = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": nw.musgrave(5000)})
        elements = cr.color_ramp.elements
        elements.new(1)
        elements[0].position = 0.0
        elements[1].position = 0.5
        elements[2].position = 1.0
        elements[0].color = map_perturb(base_hue, 1, 0.05)
        elements[1].color = map_perturb((base_hue + 0.05) % 1, 1, 0.05)
        elements[2].color = 0.0, 0.0, 0.0, 1.0

        background = map_perturb(base_hue, 0.5, 0.3)
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

        return principled_bsdf

    def create_asset(self, **kwargs):
        n = np.random.randint(4, 6)
        mesh = self.build_lichen_circle_mesh(n)
        obj = bpy.data.objects.new("lichen", mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj

        boundary = obj.vertex_groups.new(name="Boundary")
        boundary.add(list(range(n)), 1.0, "REPLACE")

        growth_scale = 1, 1, 0.5
        build_diff_growth(
            obj,
            boundary.index,
            max_polygons=self.max_polygon * uniform(0.2, 1),
            growth_scale=growth_scale,
            inhibit_shell=4,
            repulsion_radius=2,
            dt=0.25,
        )
        obj.scale = [0.004] * 3
        butil.apply_transform(obj)
        assign_material(
            obj,
            surface.shaderfunc_to_material(
                LichenFactory.shader_lichen, (self.base_hue + uniform(-0.04, 0.04)) % 1
            ),
        )

        tag_object(obj, "lichen")
        return obj
