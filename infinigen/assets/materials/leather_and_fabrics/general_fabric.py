from infinigen.assets.materials import common


from infinigen.assets.utils.uv import ensure_uv, unwrap_faces

def func_fabric(nw: NodeWrangler, **kwargs):

    group_input = {
        'Weave Scale': 0.,
        'Color Pattern Scale': 0.,
        'Color1': (0.7991, 0.1046, 0.1195, 1.0000),
        'Color2': (1.0000, 0.5271, 0.5711, 1.0000)
    }
    group_input.update(kwargs)

    wave_texture_1 = nw.new_node(Nodes.WaveTexture, input_kwargs={
        'Vector': texture_coordinate.outputs["UV"],
        'Scale': group_input["Weave Scale"],
        'Distortion': 7.0000,
        'Detail': 15.0000
    }, attrs={'bands_direction': 'Y'})



    wave_texture = nw.new_node(Nodes.WaveTexture, input_kwargs={
        'Vector': texture_coordinate.outputs["UV"],
        'Scale': group_input["Weave Scale"],
        'Distortion': 7.0000,
        'Detail': 15.0000
    })


                      input_kwargs={6: map_range.outputs["Result"], 7: map_range_1.outputs["Result"]},
                      attrs={'data_type': 'RGBA'})

    greater_than = nw.new_node(Nodes.Math, input_kwargs={0: mix.outputs[2], 1: 0.1000},
                               attrs={'operation': 'GREATER_THAN'})


    less_than = nw.new_node(Nodes.Math, input_kwargs={0: group_input["Color Pattern Scale"], 1: 0.0001},
                            attrs={'operation': 'LESS_THAN'})

    brick_texture_2 = nw.new_node(Nodes.BrickTexture, input_kwargs={
        'Vector': texture_coordinate.outputs["UV"],
        'Color1': group_input["Color1"],
        'Mortar': group_input["Color2"],
        'Scale': group_input["Color Pattern Scale"],
        'Mortar Size': 0.0000,
        'Bias': -1.0000,
        'Row Height': 0.5000
    }, attrs={'offset_frequency': 1, 'squash': 0.0000})

    vector_rotate = nw.new_node(Nodes.VectorRotate, input_kwargs={
        'Vector': texture_coordinate.outputs["UV"],
        'Rotation': (0.0000, 0.0000, 1.5708)
    }, attrs={'rotation_type': 'EULER_XYZ'})

    brick_texture = nw.new_node(Nodes.BrickTexture, input_kwargs={
        'Vector': vector_rotate,
        'Color1': group_input["Color1"],
        'Mortar': group_input["Color2"],
        'Scale': group_input["Color Pattern Scale"],
        'Mortar Size': 0.0000,
        'Bias': -1.0000,
        'Row Height': 0.5000
    }, attrs={'offset_frequency': 1, 'squash': 0.0000})

    mix_2 = nw.new_node(Nodes.Mix, input_kwargs={
        0: 1.0000,
        6: brick_texture_2.outputs["Color"],
        7: brick_texture.outputs["Color"]
    }, attrs={'data_type': 'RGBA', 'blend_type': 'ADD'})

    mix_4 = nw.new_node(Nodes.Mix, input_kwargs={0: less_than, 6: mix_2.outputs[2], 7: group_input["Color1"]},
                        attrs={'data_type': 'RGBA'})

    mix_3 = nw.new_node(Nodes.Mix, input_kwargs={
        0: mix.outputs[2],
        6: (0.0000, 0.0000, 0.0000, 1.0000),
        7: mix_4.outputs[2]
    }, attrs={'data_type': 'RGBA'})


    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': mix_3.outputs[2],
        'Roughness': map_range_2.outputs["Result"],
        'Sheen': 1.0000,
        'Sheen Tint': 1.0000
    })

    mix_shader = nw.new_node(Nodes.MixShader,
                             input_kwargs={'Fac': greater_than, 1: transparent_bsdf, 2: principled_bsdf})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input["Weave Scale"], 1: 5.0000},
                           attrs={'operation': 'MULTIPLY'})


    mix_1 = nw.new_node(Nodes.Mix, input_kwargs={6: musgrave_texture, 7: mix.outputs[2]},
                        attrs={'data_type': 'RGBA'})



    displacement = nw.new_node(
        'ShaderNodeDisplacement', input_kwargs={'Height': multiply_1, 'Midlevel': 0.0000}
    )

    return {'Shader': mix_shader, 'Displacement': displacement}

    group = func_fabric(nw, **{
        'Weave Scale': weave_scale,
        'Color Pattern Scale': color_scale,
        'Color1': color_1,
        'Color2': color_2
    })

    displacement = nw.new_node('ShaderNodeDisplacement',
                               input_kwargs={'Height': group["Displacement"], 'Midlevel': 0.0000})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': group["Shader"], 'Displacement': displacement
                                  }, attrs={'is_active_output': True})


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_fabric, selection, **kwargs)
