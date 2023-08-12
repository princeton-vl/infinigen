import math

import colorsys

from numpy.random import uniform as U

from infinigen.assets.utils.object import new_cube
from infinigen.assets.utils.misc import build_color_ramp
from infinigen.assets.utils.decorate import assign_material
from infinigen.assets.utils.tag import tag_object

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface

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

        cr = build_color_ramp(nw, 
            nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 5.}).outputs["Fac"], 
            [0, .5, 1],
            [map_perturb(base_hue, .8, .1), map_perturb(base_hue - 0.05, .8, .1), (0., 0., 0., 1.)]
        )

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