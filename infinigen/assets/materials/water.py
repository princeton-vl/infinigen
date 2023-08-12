# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma, Alex Raistrick
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=X3LlsdddMLo by Kev Binge


import os

import bpy
import gin
import numpy as np
from mathutils import Vector
from infinigen.core.nodes.node_wrangler import Nodes
from numpy.random import normal, uniform
from infinigen.core import surface
from infinigen.terrain.assets.ocean import ocean_asset, spatial_size
from infinigen.core.util.organization import SurfaceTypes
from infinigen.terrain.utils import drive_param
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_general as rg
from infinigen.core.util.organization import Attributes

type = SurfaceTypes.BlenderDisplacement
mod_name = "geo_water"
name = "water"
info = {}

@gin.configurable('geo')
def geo_water(
    nw,
    asset_paths,
    coastal,
    tile_size=40,
    with_waves=True,
    with_ripples=("bool", 0.3),
    waves_animation_speed=("uniform", 0.005, 0.025),
    animate_ripples=True,
    water_scale=("uniform", 4, 6),
    water_detail=("uniform", 5, 10),
    water_height=("uniform", 0.002, 0.02),
    water_dimension=("uniform", 1.0, 1.4),
    water_lacunarity=("uniform", 1.8, 2.0),
    height_modulation_scale=("uniform", 1, 5),
    ripples_lattice=2,
    selection=None
):
    nw.force_input_consistency()
    group_input = nw.new_node(Nodes.GroupInput)
    position0 = nw.new_node("GeometryNodeInputPosition")

    if asset_paths == []:
        with_ripples = rg(with_ripples)
        water_scale_node = nw.new_node(Nodes.Value, label=f"water_scale")
        water_scale_node.outputs[0].default_value = rg(water_scale)
        water_detail_node = nw.new_node(Nodes.Value, label=f"water_detail")
        water_detail_node.outputs[0].default_value = rg(water_detail)
        water_height_node = nw.new_node(Nodes.Value, label=f"water_height")
        water_height = rg(water_height)
        if with_waves:
            water_height_node.outputs[0].default_value = water_height
        else:
            water_height_node.outputs[0].default_value = 0
        ripple_height_node = nw.new_node(Nodes.Value, label=f"ripple_height")
        ripple_height_node.outputs[0].default_value = 0
        if with_ripples:
            water_height_node.outputs[0].default_value *= 0.1
            ripple_height_node.outputs[0].default_value = water_height
        water_height_node = nw.scalar_multiply(water_height_node, nw.scalar_add(0.5, nw.new_node("ShaderNodeTexMusgrave", input_kwargs={"Scale": rg(height_modulation_scale)})))
        water_dimension_node = nw.new_node(Nodes.Value, label=f"water_dimension")
        water_dimension_node.outputs[0].default_value = rg(water_dimension)
        water_lacunarity_node = nw.new_node(Nodes.Value, label=f"water_lacunarity")
        water_lacunarity_node.outputs[0].default_value = rg(water_lacunarity)


        position_shift = nw.new_node(Nodes.Vector, label="wave")
        position_shift.vector = nw.get_position_translation_seed(f"wave")

        animated_position = nw.add(Vector([0, 0, 0]), position0, position_shift)
        if waves_animation_speed is not None:
            drive_param(animated_position.inputs[0], rg(waves_animation_speed), offset=uniform(0, 10), index=1)

        wave0 = nw.new_node("ShaderNodeTexMusgrave", [
            animated_position, None,
            water_scale_node,
            water_detail_node,
            water_dimension_node,
            water_lacunarity_node,
        ])

        # normal_direction = nw.new_node("GeometryNodeInputNormal", [])
        # temporarily assume flat water
        normal_direction = Vector([0, 0, 1])
        waves = []
        for i in range(ripples_lattice):
            position_shift = nw.new_node(Nodes.Vector, label=f"ripple{i}")
            position_shift.vector = nw.get_position_translation_seed(f"ripple{i}")
            position = nw.add(position_shift, position0)
            voronoi = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': position, 'Scale': 0.1, 'Randomness': 1})
            instance_offset = nw.new_node(
                Nodes.WaveTexture, [nw.sub(position, (voronoi, 2)), 1],
                attrs={'wave_type': 'RINGS', 'rings_direction': 'SPHERICAL'},
            )
            if animate_ripples:
                drive_param(instance_offset.inputs["Phase Offset"], -uniform(0.2, 1), offset=uniform(0, 10))
            edgeweight = nw.new_node(Nodes.VoronoiTexture,
                                        input_kwargs={'Vector': position, 'Scale': 0.1, 'Randomness': 1},
                                        attrs={'feature': 'DISTANCE_TO_EDGE'})
            waves.append(nw.multiply(edgeweight, instance_offset))
        offset = nw.multiply(
            nw.scalar_add(
                nw.scalar_multiply(wave0, water_height_node),
                nw.scalar_multiply(nw.add(*waves), ripple_height_node),
            ),
            normal_direction,
        )
    else:
        # Simple repetitive tiling
        directory = asset_paths[0] / "cache"
        filepath = directory / "disp_0001.exr"
        seq = bpy.data.images.load(str(filepath))
        seq.source = 'SEQUENCE'
        angle = np.random.uniform(0, np.pi * 2)
        position_shift = nw.get_position_translation_seed(f"wave")
        position = nw.add(position0, position_shift)
        position = nw.multiply(nw.new_node(Nodes.VectorRotate, input_kwargs={"Vector": position, "Angle": angle}), [1/tile_size] * 3)
        sampled_disp = nw.new_node(Nodes.ImageTexture, [seq, position])
        drive_param(sampled_disp.inputs["Frame"], 1, 0)
        offset = nw.multiply(sampled_disp, Vector([tile_size / spatial_size, tile_size / spatial_size, -tile_size / spatial_size]))
        offset = nw.new_node(Nodes.VectorRotate, input_kwargs={"Vector": offset, "Angle": -angle})
        filepath = directory / "foam_0001.exr"
        seq = bpy.data.images.load(str(filepath))
        seq.source = 'SEQUENCE'
        foam = nw.new_node(Nodes.ImageTexture, [seq, position])
        drive_param(foam.inputs["Frame"], 1, 0)
        if coastal:
            X = nw.new_node(Nodes.SeparateXYZ, [position0])
            weight1 = nw.scalar_multiply(1 / np.pi, nw.scalar_sub(np.pi / 2, nw.new_node(Nodes.Math, [nw.scalar_multiply(0.1, nw.scalar_add(30, X))], attrs={'operation': 'ARCTANGENT'})))
            weight2 = nw.scalar_add(0.5, nw.scalar_multiply(1 / np.pi, nw.new_node(Nodes.Math, [nw.scalar_multiply(0.1, nw.scalar_add(60, X))], attrs={'operation': 'ARCTANGENT'})))
            offset = nw.multiply(offset, nw.scalar_multiply(weight1, weight2))
            offset = nw.add(offset, nw.multiply(nw.new_node("ShaderNodeTexMusgrave", input_kwargs={"Scale": 1}), [0, 0, 0.03]))
            foam = nw.multiply(foam, weight2)

        group_input = nw.new_node(
            Nodes.CaptureAttribute,
            input_kwargs={
                "Geometry": group_input,
                "Value": foam
            },
            attrs={"data_type": "FLOAT"},
        )

    if selection is not None:
        offset = nw.multiply(offset, surface.eval_argument(nw, selection))

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input,
            "Offset": offset,
        },
    )
    input_kwargs = {'Geometry': set_position}
    if asset_paths != []:
        input_kwargs["foam"] = (group_input, "Attribute")
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs=input_kwargs)


@gin.configurable
def shader(
    nw,
    asset_paths,
    coastal,
    color=("color_category", 'water'),
    enable_scatter=True,
    colored=False,
    emissive_foam=False,
    volume_density=("uniform", 0.07, 0.09),
    anisotropy=("clip_gaussian", 0.75, 0.2, 0.5, 1),
    random_seed=0,
):
    nw.force_input_consistency()
    position = nw.new_node('ShaderNodeNewGeometry', [])
    # Code generated using version 2.3.1 of the node_transpiler (partly)
    with FixedSeed(random_seed):
        color = rg(color)
        light_path = nw.new_node(Nodes.LightPath)

        if colored:
            color_of_transparent_bsdf_principled_bsdf = color
        else:
            color_of_transparent_bsdf_principled_bsdf = (1, 1, 1, 1)

        transparent_bsdf = nw.new_node(Nodes.TransparentBSDF, input_kwargs={"Color": color_of_transparent_bsdf_principled_bsdf})
        principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            "Base Color": color_of_transparent_bsdf_principled_bsdf, "Roughness": 0.0, "IOR": 1.33, "Transmission": 1.0
        })
        surface_shader = nw.new_node(Nodes.MixShader, input_kwargs={
            'Fac': nw.scalar_multiply(1.0, light_path.outputs["Is Camera Ray"]),
            1: transparent_bsdf,
            2: principled_bsdf
        })

        if asset_paths != []:
            if emissive_foam:
                foam_bsdf = nw.new_node(Nodes.Emission, input_kwargs={'Strength': 1})
            else:
                foam_bsdf = nw.new_node(Nodes.DiffuseBSDF)
            foam = nw.new_node(Nodes.Attribute, attrs={"attribute_name": "foam"})
            if coastal:
                weight = nw.scalar_multiply(3, nw.scalar_sub2(1, nw.scalar_multiply(5, nw.new_node(Nodes.Attribute, attrs={"attribute_name": Attributes.BoundarySDF}))))
                weight.use_clamp = 1
                interior_weight = nw.scalar_multiply(1 / np.pi, nw.scalar_sub(
                    np.pi / 2,
                    nw.new_node(Nodes.Math, [nw.scalar_multiply(0.1, nw.scalar_add(30, nw.new_node(Nodes.SeparateXYZ, [position])))], attrs={'operation': 'ARCTANGENT'})
                ))
                weight = nw.scalar_add(weight, interior_weight)
                weight.use_clamp = 1
            else:
                weight = 1
            foam = nw.scalar_multiply(foam, weight)
            surface_shader = nw.new_node(Nodes.MixShader, input_kwargs={'Fac': foam, 1: surface_shader, 2: foam_bsdf})
    
        rgb = nw.new_node(Nodes.RGB)
        rgb.outputs[0].default_value = color
        principled_volume = nw.new_node(Nodes.PrincipledVolume, input_kwargs={
            'Color': rgb, 
            'Absorption Color': rgb,
            'Density': rg(volume_density) if enable_scatter else 0,
            'Anisotropy': rg(anisotropy),
        })

        material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': surface_shader, 'Volume': principled_volume})

@gin.configurable("water")
def apply(objs, is_ocean=False, coastal=0, selection=None, **kwargs):
    info["is_ocean"] = is_ocean = rg(is_ocean)
    asset_paths = []
    if is_ocean:
        ocean_folder = kwargs["ocean_folder"]
        (ocean_folder / "cache").mkdir(parents=1, exist_ok=1)
        (ocean_folder / "cache/disp_0001.exr").touch()
        (ocean_folder / "cache/foam_0001.exr").touch()
        asset_paths.append(ocean_folder)
    input_kwargs = {"asset_paths": asset_paths, "coastal": coastal}
    surface.add_geomod(objs, geo_water, selection=selection, input_kwargs=input_kwargs, attributes=["foam"] if is_ocean else None)
    surface.add_material(objs, shader, selection=selection, input_kwargs=input_kwargs)
    if is_ocean:
        (ocean_folder / "cache/disp_0001.exr").unlink()
        (ocean_folder / "cache/foam_0001.exr").unlink()
