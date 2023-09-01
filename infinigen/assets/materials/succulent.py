# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgements: This file draws inspiration from https://blenderartists.org/t/extrude-face-along-curve-with-geometry-nodes/1432653/3

from numpy.random import uniform, normal , randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.core.util.color import hsv2rgba



def shader_green_transition_succulent(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
                                     input_kwargs={'Scale': uniform(5.0, 20.0)})

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': musgrave_texture_1})
    colorramp_3.color_ramp.elements[0].position = 0.1182
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.7727
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': mapping})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.85},
                            attrs={'operation': 'LESS_THAN'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: colorramp_3.outputs["Color"], 1: less_than},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: uniform(2.0, 10.0)},
                             attrs={'operation': 'MULTIPLY'})

    main_hsv_color = (uniform(0.35, 0.42), uniform(0.5, 0.93), uniform(0.20, 0.80))
    main_color = hsv2rgba(main_hsv_color)
    diffuse_color = hsv2rgba((uniform(0.34, 0.43),) + main_hsv_color[1:])
    split_point = uniform(0.82, 0.92)
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': separate_xyz.outputs["Z"]})
    colorramp_1.color_ramp.interpolation = "B_SPLINE"
    colorramp_1.color_ramp.elements[0].position = split_point
    colorramp_1.color_ramp.elements[0].color = main_color
    colorramp_1.color_ramp.elements[1].position = split_point + uniform(0.01, 0.05)
    colorramp_1.color_ramp.elements[1].color = (uniform(0.6, 1.0), uniform(0.0, 0.08), uniform(0.0, 0.08), 1.0)
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_1.outputs["Color"], 'Subsurface': uniform(0.01, 0.03), 
        'Subsurface Radius': (0.01, 0.1, 0.1), 'Subsurface Color': colorramp_1.outputs["Color"], 
        'Subsurface IOR': 0.0, 'Specular': 0.0, 'Roughness': 2.0, 'Sheen Tint': 0.0, 
        'Clearcoat Roughness': 0.0, 'IOR': 1.3, 'Emission Strength': 0.0})

    diffuse_bsdf_1 = nw.new_node(Nodes.DiffuseBSDF,
                                 input_kwargs={'Color': diffuse_color, 'Roughness': uniform(0.3, 0.8)})

    mix_shader_2 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': multiply_1, 1: principled_bsdf, 2: diffuse_bsdf_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': mix_shader_2})


def shader_pink_transition_succulent(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
                                     input_kwargs={'Scale': uniform(5.0, 20.0)})

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': musgrave_texture_1})
    colorramp_3.color_ramp.elements[0].position = 0.1182
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.7727
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': mapping})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.85},
                            attrs={'operation': 'LESS_THAN'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: colorramp_3.outputs["Color"], 1: less_than},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: uniform(2.0, 8.0)},
                             attrs={'operation': 'MULTIPLY'})

    main_hsv_color = (uniform(0.93, 0.99), uniform(0.64, 0.90), uniform(0.50, 0.90))
    main_color = hsv2rgba(main_hsv_color)
    diffuse_color = hsv2rgba((uniform(0.93, 1.), ) + main_hsv_color[1:])
    split_point = uniform(0.82, 0.92)
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': separate_xyz.outputs["Z"]})
    colorramp_1.color_ramp.interpolation = "B_SPLINE"
    colorramp_1.color_ramp.elements[0].position = split_point
    colorramp_1.color_ramp.elements[0].color = main_color
    colorramp_1.color_ramp.elements[1].position = split_point + uniform(0.01, 0.05)
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_1.outputs["Color"], 'Subsurface': uniform(0.01, 0.05), 
        'Subsurface Radius': (0.01, 0.03, 0.03), 'Subsurface Color': colorramp_1.outputs["Color"], 
        'Subsurface IOR': 0.0, 'Specular': 0.0, 'Roughness': 2.0, 'Sheen Tint': 0.0, 
        'Clearcoat Roughness': 0.0, 'IOR': 1.3, 'Emission Strength': 0.0})

    diffuse_bsdf_1 = nw.new_node(Nodes.DiffuseBSDF,
                                 input_kwargs={'Color': diffuse_color, 'Roughness': uniform(0.0, 0.5)})

    mix_shader_2 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': multiply_1, 1: principled_bsdf, 2: diffuse_bsdf_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': mix_shader_2})


def shader_green_succulent(nw: NodeWrangler):
    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
                                     input_kwargs={'Scale': uniform(4.0, 15.0)})

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': musgrave_texture_1})
    colorramp_3.color_ramp.elements[0].position = 0.1182
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.7727
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': mapping})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: separate_xyz.outputs["Z"], 1: 1.0},
                            attrs={'operation': 'LESS_THAN'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: colorramp_3.outputs["Color"], 1: less_than},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: uniform(2.0, 8.0)},
                             attrs={'operation': 'MULTIPLY'})

    main_hsv_color = (uniform(0.33, 0.39), uniform(0.5, 0.93), uniform(0.20, 0.70))
    main_color = hsv2rgba(main_hsv_color)
    diffuse_color = hsv2rgba((uniform(0.34, 0.38),) + main_hsv_color[1:])
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': separate_xyz.outputs["Z"]})
    colorramp_1.color_ramp.interpolation = "B_SPLINE"
    colorramp_1.color_ramp.elements[0].position = 1.0
    colorramp_1.color_ramp.elements[0].color = main_color
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_1.outputs["Color"], 'Subsurface': uniform(0.01, 0.05), 
        'Subsurface Radius': (0.1, 0.1, 0.1), 'Subsurface Color': colorramp_1.outputs["Color"], 
        'Subsurface IOR': 0.0, 'Specular': 0.0, 'Roughness': 2.0, 'Sheen Tint': 0.0, 
        'Clearcoat Roughness': 0.0, 'IOR': 1.3, 'Emission Strength': 0.0})

    diffuse_bsdf_1 = nw.new_node(Nodes.DiffuseBSDF,
                                 input_kwargs={'Color': diffuse_color, 'Roughness': uniform(0.0, 0.5)})

    mix_shader_2 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': multiply_1, 1: principled_bsdf, 2: diffuse_bsdf_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': mix_shader_2})


def shader_yellow_succulent(nw: NodeWrangler):
    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
                                     input_kwargs={'Scale': uniform(5.0, 8.0)})

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': musgrave_texture_1})
    colorramp_3.color_ramp.elements[0].position = 0.1182
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.7727
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': mapping})

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: separate_xyz.outputs["Z"], 1: 1.0},
                            attrs={'operation': 'LESS_THAN'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: colorramp_3.outputs["Color"], 1: less_than},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: uniform(1.0, 3.0)},
                             attrs={'operation': 'MULTIPLY'})

    main_color = hsv2rgba((uniform(0.1, 0.15), uniform(0.8, 1.0), uniform(0.5, 0.7)))
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': separate_xyz.outputs["Z"]})
    colorramp_1.color_ramp.interpolation = "B_SPLINE"
    colorramp_1.color_ramp.elements[0].position = 0.3114
    colorramp_1.color_ramp.elements[0].color = main_color
    colorramp_1.color_ramp.elements[1].position = 0.6864
    colorramp_1.color_ramp.elements[1].color = hsv2rgba((uniform(0.0, 0.06), uniform(0.8, 1.0), uniform(0.5, 0.7)))

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': colorramp_1.outputs["Color"], 'Subsurface': 0.01,
                                                'Subsurface Radius': (1.0, 1.0, 1.0),
                                                'Subsurface Color': colorramp_1.outputs["Alpha"], 'Subsurface IOR': 1.3,
                                                'Specular': 0.0, 'Roughness': 2.0, 'Sheen Tint': 0.0,
                                                'Clearcoat Roughness': 0.0, 'IOR': 1.3, 'Emission Strength': 0.0})

    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
                                   input_kwargs={'Color': main_color})

    mix_shader_1 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': 0.4, 1: principled_bsdf, 2: translucent_bsdf})

    diffuse_bsdf_1 = nw.new_node(Nodes.DiffuseBSDF,
                                 input_kwargs={'Color': main_color,
                                               'Roughness': uniform(0.2, 1.0)})

    mix_shader_2 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': multiply_1, 1: mix_shader_1, 2: diffuse_bsdf_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': mix_shader_2})


def shader_whitish_green_succulent(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
                                     input_kwargs={'Scale': uniform(5.0, 8.0)})

    colorramp_3 = nw.new_node(Nodes.ColorRamp,
                              input_kwargs={'Fac': musgrave_texture_1})
    colorramp_3.color_ramp.elements[0].position = uniform(0.0, 0.3)
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.5273
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
                          input_kwargs={'Vector': texture_coordinate.outputs["Generated"]})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': mapping})

    l = uniform(0.88, 0.98)
    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: separate_xyz.outputs["Z"], 1: l - 0.05},
                            attrs={'operation': 'LESS_THAN'})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: colorramp_3.outputs["Color"], 1: less_than},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: multiply, 1: uniform(1.0, 4.0)},
                             attrs={'operation': 'MULTIPLY'})

    main_color = hsv2rgba((uniform(0.23, 0.25), uniform(0.40, 0.60), uniform(0.18, 0.25)))
    colorramp = nw.new_node(Nodes.ColorRamp,
                            input_kwargs={'Fac': separate_xyz.outputs["Z"]})
    colorramp.color_ramp.elements[0].position = l - uniform(0.04, 0.1)
    colorramp.color_ramp.elements[0].color = hsv2rgba((uniform(0.20, 0.38), uniform(0.12, 0.25), uniform(0.50, 0.70)))
    colorramp.color_ramp.elements[1].position = l
    colorramp.color_ramp.elements[1].color = main_color

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': colorramp.outputs["Color"], 'Subsurface': 0.01,
                                                'Subsurface Radius': (1.0, 1.0, 1.0),
                                                'Subsurface Color': colorramp.outputs["Color"], 'Subsurface IOR': 1.3,
                                                'Specular': 0.0, 'Roughness': 2.0, 'Sheen Tint': 0.0,
                                                'Clearcoat Roughness': 0.0, 'IOR': 1.3, 'Emission Strength': 0.0})

    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
                                   input_kwargs={'Color': main_color})

    mix_shader_1 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': 0.7, 1: principled_bsdf, 2: translucent_bsdf})
    diffuse = hsv2rgba((uniform(0.23, 0.25), uniform(0.40, 0.60), uniform(0.10, 0.15)))
    diffuse_bsdf_1 = nw.new_node(Nodes.DiffuseBSDF,
                                 input_kwargs={'Color': diffuse, 'Roughness': 0.5})

    mix_shader_2 = nw.new_node(Nodes.MixShader,
                               input_kwargs={'Fac': multiply_1, 1: mix_shader_1, 2: diffuse_bsdf_1})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': mix_shader_2})


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_green_transition_succulent, selection=selection)