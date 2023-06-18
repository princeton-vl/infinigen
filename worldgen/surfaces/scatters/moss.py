# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

import math

import colorsys

import numpy as np
from numpy.random import uniform as U

from placement.instance_scatter import scatter_instances
from assets.utils.object import new_cube
from assets.utils.misc import build_color_ramp
from assets.utils.decorate import assign_material
from placement.factory import AssetFactory, make_asset_collection
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from surfaces import surface
from assets.utils.tag import tag_object, tag_nodegroup
from placement.instance_scatter import scatter_instances

class MossFactory(AssetFactory):

    def __init__(self, factory_seed):
        super(MossFactory, self).__init__(factory_seed)
        self.max_polygon = 1e4
        self.base_hue = U(.2, .24)

    @staticmethod
    def shader_moss(nw: NodeWrangler, base_hue=.3):
        h_perturb = U(-0.02, .02)
        s_perturb = U(-.1, -.0)
        v_perturb = U(1., 1.5)

        def map_perturb(h, s, v):
            return *colorsys.hsv_to_rgb(h + h_perturb, s + s_perturb, v / v_perturb), 1.

        subsurface_ratio = .05
        roughness = 1.
        mix_ratio = .2

        cr = build_color_ramp(nw, nw.musgrave(20), [0, .5, 1],
                              [map_perturb(base_hue, .8, .1), map_perturb(base_hue - 0.05, .8, .1),
                                  (0., 0., 0., 1.)])

        background = map_perturb(base_hue, .8, .02)
        mix_rgb = nw.new_node(Nodes.MixRGB,
                              [nw.new_node(Nodes.ObjectInfo_Shader).outputs["Random"], cr.outputs["Color"],
                                  background])

        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': mix_rgb,
            'Subsurface': subsurface_ratio,
            'Subsurface Radius': (.01, .01, .01),
            'Subsurface Color': background,
            'Roughness': roughness
        })

        translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF, input_kwargs={'Color': mix_rgb})

        mix_shader = nw.new_node(Nodes.MixShader, [mix_ratio, principled_bsdf, translucent_bsdf])
        return mix_shader

    def create_asset(self, face_size=.01, **params):
        obj = new_cube()
        surface.add_geomod(obj, self.geo_moss_instance, apply=True, input_args=[face_size])
        assign_material(obj, surface.shaderfunc_to_material(MossFactory.shader_moss,
                                                            (self.base_hue + U(-.02, .02) % 1)))
        tag_object(obj, 'moss')
        return obj

    @staticmethod
    def geo_moss_instance(nw: NodeWrangler, face_size):
        radius = .008
        start = (0.0, 0.0, 0.0)
        start_handle = (-.03, 0.0, .02)
        end = (-0.04, 0.0, U(.04, .05))
        end_handle = (end[0] + U(-.03, -.02), 0., end[2] + U(-.01, .0))
        bezier = nw.new_node(Nodes.CurveBezierSegment, input_kwargs={
            'Resolution': 10 * math.ceil(.01 / face_size),
            'Start': start,
            'Start Handle': start_handle,
            'End Handle': end_handle,
            'End': end
        })
        circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': 4, 'Radius': radius}).outputs[
            "Curve"]
        mesh = nw.curve2mesh(bezier, circle)
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': mesh})


class MossCover:

    def __init__(self):
        self.col = make_asset_collection(MossFactory(np.random.randint(1e5)), name='moss', n=3)
        base_hue = U(.24, .28)
        for o in self.col.objects:
            assign_material(o, surface.shaderfunc_to_material(MossFactory.shader_moss,
                                                              (base_hue + U(-.02, .02)) % 1))

    def apply(self, obj, selection=None):

        def instance_index(nw: NodeWrangler, n):
            return nw.math('MODULO',
                           nw.new_node(Nodes.FloatToInt, [nw.scalar_multiply(nw.musgrave(10), 2 * n)]), n)

        scatter_obj = scatter_instances(
            base_obj=obj, collection=self.col, 
            density=2e4, min_spacing=.005, 
            scale=1, scale_rand=U(0.3, 0.7),
            selection=selection, 
            instance_index=instance_index)

        return scatter_obj
