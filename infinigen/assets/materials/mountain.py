# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import clip_hsv, random_color, random_color_neighbour
from infinigen.core.util.random import random_general as rg

from infinigen.terrain.land_process.snowfall import snowfall_params

type = SurfaceTypes.SDFPerturb
mod_name = "geo_MOUNTAIN"
name = "mountain"

def geo_MOUNTAIN_general(
    nw: NodeWrangler,
    n_noise,
    noise_params,
    n_crack,
    crack_params,
    crack_modulation_params,
    selection=None
):
    position = nw.new_node("GeometryNodeInputPosition", [])
    normal = nw.new_node("GeometryNodeInputNormal", [])

    noises = []

    for i in range(n_noise):
        scale = nw.new_node(Nodes.Value, label=f"scale{i}")
        scale.outputs[0].default_value = rg(noise_params["scale"])
        detail = nw.new_node(Nodes.Value, label=f"detail{i}")
        detail.outputs[0].default_value = rg(noise_params["detail"])
        roughness = nw.new_node(Nodes.Value, label=f"roughness{i}")
        roughness.outputs[0].default_value = rg(noise_params["roughness"])
        zscale = nw.new_node(Nodes.Value, label=f"zscale{i}")
        zscale.outputs[0].default_value = rg(noise_params["zscale"])
        position_shift = nw.new_node(Nodes.Vector, label=f"position_shift{i}")
        position_shift.vector = nw.get_position_translation_seed(f"content{i}")

        content = nw.scalar_multiply(
            nw.scalar_sub(
                nw.new_node(
                    Nodes.NoiseTexture,
                    input_kwargs={
                        'Vector': nw.add(position, position_shift),
                        'Scale': scale, 'Detail': detail, 'Roughness': roughness
                    }
                ),
                0.5
            ),
            zscale
        )
        noises.append(content)
    offset = nw.scalar_max(*noises)


    if n_crack > 0:
        cracks = []
        for i in range(n_crack):
            scale = nw.new_node(Nodes.Value, label=f"crack_modulation_scale{i}")
            scale.outputs[0].default_value = rg(crack_modulation_params["scale"])
            detail = nw.new_node(Nodes.Value, label=f"crack_modulation_detail{i}")
            detail.outputs[0].default_value = rg(crack_modulation_params["detail"])
            roughness = nw.new_node(Nodes.Value, label=f"crack_modulation_roughness{i}")
            roughness.outputs[0].default_value = rg(crack_modulation_params["roughness"])

            position_shift = nw.new_node(Nodes.Vector, label=f"position_shift_mask{i}")
            position_shift.vector = nw.get_position_translation_seed(f"mask{i}")

            mask = nw.new_node(
                Nodes.NoiseTexture,
                input_kwargs={'Vector': nw.add(position, position_shift), 'Scale': scale, 'Detail': detail, 'Roughness': roughness}
            )

            position_shift = nw.new_node(Nodes.Vector, label=f"position_shift_slope_modulation{i}")
            position_shift.vector = nw.get_position_translation_seed(f"slope_modulation{i}")

            slope_modulation = nw.new_node(Nodes.MapRange, input_kwargs={
                "Value": nw.new_node(
                    Nodes.NoiseTexture,
                    input_kwargs={'Vector': nw.add(position, position_shift), 'Scale': scale, 'Detail': detail, 'Roughness': roughness}
                ),
                "From Min": 0.45,
                "From Max": 0.55
            })

            scale = nw.new_node(Nodes.Value, label=f"crack_scale{i}")
            scale.outputs[0].default_value = rg(crack_params["scale"])
            zscale_scale = nw.new_node(Nodes.Value, label=f"crack_zscale_scale{i}")
            zscale_scale.outputs[0].default_value = rg(crack_params["zscale_scale"])
            slope_base = nw.new_node(Nodes.Value, label=f"crack_slope_base{i}")
            slope_base.outputs[0].default_value = rg(crack_params["slope_base"])
            slope_scale = nw.new_node(Nodes.Value, label=f"crack_slope_scale{i}")
            slope_scale.outputs[0].default_value = rg(crack_params["slope_scale"])
            mask_rampmin = nw.new_node(Nodes.Value, label=f"crack_mask_rampmin{i}")
            mask_rampmin.outputs[0].default_value = rg(crack_params["mask_rampmin"])
            mask_rampmax = nw.new_node(Nodes.Value, label=f"crack_mask_rampmax{i}")
            mask_rampmax.outputs[0].default_value = rg(crack_params["mask_rampmax"])

            mask_crack = nw.new_node(Nodes.MapRange, input_kwargs={
                "Value": mask,
                "From Min": nw.scalar_add(mask_rampmin, 0.5),
                "From Max": nw.scalar_add(mask_rampmax, 0.5)
            })
            zscale_modulation = nw.scalar_multiply(
                zscale_scale,
                nw.power(
                    slope_base,
                    slope_modulation # reuse
                )
            )
            slope_modulation = nw.scalar_multiply(
                nw.scalar_divide(1.0, slope_scale),
                nw.power(
                    nw.scalar_divide(1.0, slope_base),
                    slope_modulation
                )
            )
            position_shift = nw.new_node(Nodes.Vector, label=f"position_shift_crack{i}")
            position_shift.vector = nw.get_position_translation_seed(f"crack{i}")

            crack = nw.scalar_multiply(
                nw.new_node(Nodes.MapRange, input_kwargs={
                    "Value": nw.new_node(Nodes.VoronoiTexture,
                        input_kwargs={'Vector': nw.add(position, position_shift), 'Scale': scale},
                        attrs={"feature": "DISTANCE_TO_EDGE"}
                    ),
                    "From Max": slope_modulation,
                    "To Min": -1.0,
                    "To Max": 0.0
                }),
                mask_crack,
                zscale_modulation
            )
            cracks.append(crack)
        offset = nw.scalar_add(offset, nw.scalar_add(*cracks))
    offset = nw.multiply(offset, normal)
    if selection is not None:
        offset = nw.multiply(offset, surface.eval_argument(nw, selection))
    return offset


@gin.configurable("geo")
def geo_MOUNTAIN(
    nw: NodeWrangler,
    n_noise=3,
    noise_params={"scale": ("uniform", 1, 5), "detail": 8, "roughness": 0.7, "zscale": ("power_uniform", -1, -0.5)},
    n_crack=8,
    crack_params={"scale": ("uniform", 1, 5), "zscale_scale": 0.02, "slope_scale": 5, "slope_base": 3, "mask_rampmin": 0.0, "mask_rampmax": 0.3},
    crack_modulation_params={"scale": 1, "detail": 5, "roughness": 0.5},
    selection=None
):
    nw.force_input_consistency()
    groupinput = nw.new_node(Nodes.GroupInput)
    offset = geo_MOUNTAIN_general(nw, n_noise, noise_params, n_crack, crack_params, crack_modulation_params)
    if selection is not None: offset = nw.multiply(offset, surface.eval_argument(nw, selection))
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": groupinput,  "Offset": offset})
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})


@gin.configurable("shader")
def shader_MOUNTAIN(
        nw,
        obj,
        is_rock=False,
        spherical=False,
        preset_zrange=(0, 120),
        color=None,
        shader_roughness=1,
        num_layers=16,
        random_seed=0,
        layered_mountain=True,
        prob_arranged_layers=0.1,
        hue_diff=0.1,
        max_sat=0.4,
        max_val=0.4,
        snowy=False,
        *args,
        **kwargs
    ):
    nw.force_input_consistency()
    with FixedSeed(random_seed):

        if np.random.uniform() > prob_arranged_layers:
            arranged_layers = True
        else:
            arranged_layers = False


        shader_roughness = rg(shader_roughness)
        layered_mountain = rg(layered_mountain)

        if layered_mountain:
            tex_coor = nw.new_node('ShaderNodeNewGeometry', [])
            if spherical:
                z = nw.new_node(Nodes.VectorMath, [(tex_coor, 0)], attrs={"operation": "LENGTH"})
                z = nw.new_node('ShaderNodeMapRange', [z])
            else:
                z = nw.new_node('ShaderNodeSeparateXYZ', [(tex_coor, 0)])
                z = nw.new_node('ShaderNodeMapRange', [(z, 2)])
            z_noise_mag = np.random.uniform(0.1, 0.4)

            if preset_zrange is None:
                # map value from (-z_noise_mag / 2) to (1 - z_noise_mag / 2)
                z_min = 0 # obj.bound_box[0][-1]
                z_max = obj.bound_box[1][-1]
            else:
                z_min, z_max = preset_zrange
            # z_min must be 0 to avoid sediment under water
            
            z.inputs[1].default_value = z_min  # from min
            z.inputs[2].default_value = z_max  # from max
            z.inputs[3].default_value = -1 * (z_noise_mag / 2)  # to min
            z.inputs[4].default_value = 1 - (z_noise_mag / 2)  # to max

            z_noise = nw.new_node('ShaderNodeTexNoise',
                input_kwargs={'Vector': (tex_coor, 0), 'Scale': 0.1, "Detail": 9},
            )
            # noise scale
            z_noise = nw.new_node('ShaderNodeMath', [z_noise])
            z_noise.operation = 'MULTIPLY'
            z_noise.inputs[1].default_value = np.random.uniform(0.1, 0.3)
            z = nw.add2(z, z_noise)

            ramp = nw.new_node('ShaderNodeValToRGB', [z])
            elements = ramp.color_ramp.elements
            elements.remove(elements[0])
            # todo: better way to sample the initial color
            if color is None:
                cur_color = random_color()
                cur_color = clip_hsv(cur_color, max_s=max_sat, max_v=max_val)
            else:
                cur_color = rg(color)
            elements[-1].color = cur_color


            cur_loc = 1
            for _ in range(num_layers):

                if arranged_layers:
                    cur_loc -= (np.random.uniform() * 2 / num_layers)
                    cur_loc = max(0, cur_loc)
                else:
                    cur_loc = np.random.uniform()

                element = elements.new(cur_loc)
                if color is None:
                    cur_color = random_color_neighbour(cur_color, sat_diff=None, val_diff=None, hue_diff=hue_diff)
                    cur_color = clip_hsv(cur_color, max_s=max_sat, max_v=max_val)
                else:
                    cur_color = rg(color)
                element.color = cur_color

            # ambient occlusion
            amb_occl = nw.new_node('ShaderNodeAmbientOcclusion', [])
            ramp = nw.new_node('ShaderNodeMixRGB', [amb_occl, (0.0, 0.0, 0.0, 1.0), ramp])

        else:
            if color is None:
                ramp = random_color()[:3]
            else:
                ramp = rg(color)[:3]
        color_ = nw.multiply(
            ramp,
            nw.scalar_max(0.2,
                nw.new_node(Nodes.Math, [
                    nw.new_node(Nodes.VectorMath, [
                        nw.new_node(Nodes.VectorMath, [
                            (0.0, 0.0, 1.0),
                            (nw.new_node("ShaderNodeNewGeometry", []), 1)
                        ], attrs={'operation': 'DOT_PRODUCT'})
                    ], attrs={'operation': 'ABSOLUTE'}),
                    3
                ], attrs={'operation': 'POWER'})
            )
        )
        if snowy:
            if not is_rock:
                normal_params = snowfall_params()["detailed_normal_params"]
            else:
                normal_params = snowfall_params()["on_rock_normal_params"]
            normal = (nw.new_node('ShaderNodeNewGeometry'), 1)
            weights = [0]
            for normal_preference, (th0, th1) in normal_params:
                disturb = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 0.1, 'Detail': 9})
                th0 = nw.scalar_add(disturb, th0 - 0.5)
                th1 = nw.scalar_add(disturb, th1 - 0.5)
                map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': nw.dot(normal, normal_preference), 1: th0, 2: th1})
                weights.append(map_range)
            weights = nw.scalar_add(*weights)
            weights.use_clamp = 1
            color_ = nw.new_node('ShaderNodeMixRGB', [weights, color_, [0.904]*3 + [1]])


        bsdf_mountain = nw.new_node("ShaderNodeBsdfPrincipled",
                                    [color_, None, None, None, None, None, None,
                                        None, None, shader_roughness])

    return bsdf_mountain

def apply(objs, selection=None, **kwargs):
    if isinstance(objs, list) and len(objs) == 0:
        return
    surface.add_geomod(objs, geo_MOUNTAIN, selection=selection)
    surface.add_material(objs, shader_MOUNTAIN, selection=selection, 
        input_kwargs={"obj": objs[0] if isinstance(objs, list) else objs, **kwargs})
