from nodes import node_utils
from surfaces import surface


@node_utils.to_nodegroup(
    "nodegroup_displacement_to_offset", singleton=False, type="GeometryNodeTree"
)
def nodegroup_displacement_to_offset(nw):
    # Code generated using version 2.3.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Vector", (0.0, 0.0, 0.0)),
            ("NodeSocketFloat", "Magnitude", 1.0),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Vector"],
            1: group_input.outputs["Magnitude"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    normal = nw.new_node(Nodes.InputNormal)

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply, 1: normal},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Vector": multiply_1.outputs["Vector"]}
    )


def nodegroup_pebble(nw):
    # Code generated using version 2.3.1 of the node_transpiler

    noise1_w = nw.new_node(Nodes.Value, label="noise1_w ~ U(0, 10)")
    noise1_w.outputs[0].default_value = uniform(0.0, 10.0)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "PebbleScale", 5.0),
            ("NodeSocketFloat", "NoiseMag", 0.2),
        ],
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"W": noise1_w, "Scale": group_input.outputs["PebbleScale"]},
        attrs={"noise_dimensions": "4D"},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: noise_texture.outputs["Color"],
            1: group_input.outputs["NoiseMag"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: multiply.outputs["Vector"], 1: position}
    )

    vornoi1_w = nw.new_node(Nodes.Value, label="vornoi1_w ~ U(0, 10)")
    vornoi1_w.outputs[0].default_value = uniform(0.0, 10.0)

    voronoi_texture_2 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": add.outputs["Vector"],
            "W": vornoi1_w,
            "Scale": group_input.outputs["PebbleScale"],
        },
        attrs={"voronoi_dimensions": "4D"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance": voronoi_texture_2.outputs["Distance"]},
    )


def shader_soil(nw):
    # Code generated using version 2.3.1 of the node_transpiler
    darkness = 1.5
    soil_col_1 = random_color_neighbour((0.28 / darkness, 0.11 / darkness, 0.042 / darkness, 1.0), 0.05, 0.1, 0.1)
    soil_col_2 = random_color_neighbour((0.22 / darkness , 0.0906 / darkness , 0.035 / darkness, 1.0), 0.05, 0.1, 0.1)
    peb_col_1 = random_color_neighbour((0.3813, 0.1714, 0.0782, 1.0), 0.1, 0.1, 0.1)
    peb_col_2 = random_color_neighbour((0.314, 0.1274, 0.0578, 1.0), 0.1, 0.1, 0.1)

    ambient_occlusion = nw.new_node(Nodes.AmbientOcclusion)

    colorramp_1 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": ambient_occlusion.outputs["Color"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.8
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 1.0
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    noise1_w = nw.new_node(Nodes.Value, label="noise1_w ~ U(0, 10)")
    noise1_w.outputs[0].default_value = uniform(0.0, 10.0)

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"W": noise1_w, "Scale": 10.0},
        attrs={"noise_dimensions": "4D"},
    )

    mix_1 = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Color1": colorramp_1.outputs["Color"],
            "Color2": noise_texture.outputs["Fac"],
        },
    )

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": mix_1})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = soil_col_1
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = soil_col_2

    colorramp_3 = nw.new_node(
    )
    colorramp_3.color_ramp.elements[0].position = 0.0
    colorramp_3.color_ramp.elements[0].color = peb_col_1
    colorramp_3.color_ramp.elements[1].position = 1.0
    colorramp_3.color_ramp.elements[1].color = peb_col_2

    mix = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Color1": colorramp.outputs["Color"],
            "Color2": colorramp_3.outputs["Color"],
        },
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY_ADD"},
    )

    colorramp_2 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": multiply_add})
    colorramp_2.color_ramp.elements.new(0.0)
    colorramp_2.color_ramp.elements.new(0.0)
    colorramp_2.color_ramp.elements[0].position = 0.0
    colorramp_2.color_ramp.elements[0].color = (0.5, 0.5, 0.5, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.8636
    colorramp_2.color_ramp.elements[1].color = (0.7, 0.7, 0.7, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.9427
    colorramp_2.color_ramp.elements[2].color = (0.95, 0.95, 0.95, 1.0)
    colorramp_2.color_ramp.elements[3].position = 1.0
    colorramp_2.color_ramp.elements[3].color = (0.98, 0.98, 0.98, 1.0)

    principled_bsdf_4 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": mix,
            "Specular": 0.2,
            "Roughness": colorramp_2.outputs["Color"],
        },
    )

    return principled_bsdf_4



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(
    )
    surface.add_material(obj, shader_soil, selection=selection)
