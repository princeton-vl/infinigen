from numpy.random import uniform as U, normal as N, randint, uniform
from assets.utils.misc import log_uniform
from placement import placement

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
        ('NodeSocketFloat', 'Prewarp', 0.15), ('NodeSocketFloat', 'Scale', 0.0),
        ('NodeSocketFloat', 'Smoothness', 0.0), ('NodeSocketFloat', 'AnimSpeed', .02)])

    w = nw.new_node(Nodes.Value, label='W')

    multiply = nw.new_node(Nodes.Math, input_kwargs={1: group_input.outputs["AnimSpeed"]},
                           attrs={'operation': 'MULTIPLY'})
    driver = multiply.inputs[0].driver_add('default_value').driver
    driver.expression = f"frame / {log_uniform(100, 200)}"

    noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={
        'Vector': group_input.outputs["Vector"],
        'W': multiply,
        'Scale': log_uniform(2, 8),
        'Roughness': N(0.5, 0.05),
        'Distortion': N(0.5, 0.02)
    }, attrs={'noise_dimensions': '4D'})

    scale = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: noise_texture.outputs["Color"], 'Scale': group_input.outputs["Prewarp"]
                        }, attrs={'operation': 'SCALE'})

                      input_kwargs={0: group_input.outputs["Vector"], 1: scale.outputs["Vector"]})

    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture, input_kwargs={
        'Vector': add.outputs["Vector"],
        'W': multiply,
        'Scale': group_input.outputs["Scale"],
        'Smoothness': group_input.outputs["Smoothness"]
    }, attrs={'voronoi_dimensions': '4D', 'feature': 'SMOOTH_F1'})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Smoothness"], 1: U(.04, .08)})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture, input_kwargs={
        'Vector': add.outputs["Vector"],
        'W': multiply,
        'Scale': group_input.outputs["Scale"],
        'Smoothness': add_1
    }, attrs={'voronoi_dimensions': '4D', 'feature': 'SMOOTH_F1'})

    difference = nw.scalar_multiply(nw.math('ABSOLUTE', nw.scalar_sub(voronoi_texture, voronoi_texture_1)),
                                    20.0)

    noise = nw.math('ABSOLUTE',
                    nw.scalar_sub(nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': uniform(2, 5)}), .5))
    noise = nw.new_node(Nodes.MapRange, [noise, 0, 1, .6, 1.])

def shader_caustic_lamp(nw: NodeWrangler, params: dict):
    caustics = nw.new_node(nodegroup_caustics().name,
                           input_kwargs={'Vector': coord.outputs['Normal'], **params})
    nw.new_node(Nodes.LightOutput, [emission])

        lamp.data.spot_size = np.pi * .4
        lamp.rotation_euler = 0, 0, uniform(0, np.pi * 2)

