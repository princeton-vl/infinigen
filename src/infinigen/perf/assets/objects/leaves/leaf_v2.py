# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen v2 leaf (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/leaves/leaf_v2.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np
from numpy.random import normal

from infinigen.assets import colors
from infinigen.assets.objects.leaves.leaf_v2 import (
    LeafFactoryV2 as _DefaultLeafFactoryV2,
)
from infinigen.assets.objects.leaves.leaf_v2 import (
    geo_leaf_v2,
    shader_leaf_new,
)
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.perf.assets.utils.heightfield import mesh_route


@node_utils.to_nodegroup(
    "shader_nodegroup_sub_vein", singleton=False, type="ShaderNodeTree"
)
def shader_nodegroup_sub_vein(nw, output_max=3.0):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "X Modulated", 0.5),
            ("NodeSocketFloat", "Y", 0.0),
        ],
    )

    absolute = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["X Modulated"]},
        attrs={"operation": "ABSOLUTE", "use_clamp": True},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": absolute, "Y": group_input.outputs["Y"]}
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": combine_xyz, "Scale": 30.0, "Randomness": 0.754},
        attrs={"feature": "DISTANCE_TO_EDGE"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": voronoi_texture.outputs["Distance"],
            2: 0.1,
            4: output_max,
        },
    )

    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": combine_xyz, "Scale": 10.0, "Randomness": 0.754},
        attrs={"feature": "DISTANCE_TO_EDGE"},
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": voronoi_texture_1.outputs["Distance"],
            2: 0.1,
            4: output_max,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": multiply})


@node_utils.to_nodegroup(
    "shader_nodegroup_apply_vein_midrib", singleton=False, type="ShaderNodeTree"
)
def shader_nodegroup_apply_vein_midrib(nw, vein_width_max=0.03):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Vein Coord", 0.0),
            ("NodeSocketFloat", "Midrib Value", 0.5),
            ("NodeSocketFloat", "Leaf Shape", 1.0),
            ("NodeSocketFloat", "Vein Density", 6.0),
        ],
    )

    map_range_5 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Leaf Shape"],
            1: -0.3,
            2: 0.0,
            3: 0.015,
            4: 0.0,
        },
    )

    vein = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "W": group_input.outputs["Vein Coord"],
            "Scale": group_input.outputs["Vein Density"],
            "Randomness": 0.2,
        },
        label="Vein",
        attrs={"voronoi_dimensions": "1D"},
    )

    map_range_3 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": vein.outputs["Distance"],
            1: 0.001,
            2: 0.05,
            3: 1.0,
            4: 0.0,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: map_range_5.outputs["Result"],
            1: map_range_3.outputs["Result"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    map_range_10 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": multiply,
            1: 0.001,
            2: vein_width_max,
            3: 1.0,
            4: 0.0,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Midrib Value"],
            1: map_range_10.outputs["Result"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Vein Value": multiply_1}
    )


class LeafFactoryV2(_DefaultLeafFactoryV2):
    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        # add noise to the genotype output
        # hue_noise = np.random.randn() * 0
        # hsv_blade = self.hsv_blade + hue_noise
        # hsv_vein = self.hsv_vein + hue_noise

        phenome = self.genome.copy()

        phenome["y_wave_control_points"] = [
            (0.0, 0.5),
            (np.random.uniform(0.25, 0.75), np.random.uniform(0.50, 0.60)),
            (1.0, 0.5),
        ]
        x_wave_val = np.random.uniform(0.50, 0.58)
        phenome["x_wave_control_points"] = [
            (0.0, 0.5),
            (0.4, x_wave_val),
            (0.5, 0.5),
            (0.6, x_wave_val),
            (1.0, 0.5),
        ]

        material_kwargs = phenome.copy()
        material_kwargs["blade_color"] = list(colors.hsv2rgba(self.blade_color_hsv))
        material_kwargs["blade_color"][0] += np.random.normal(0.0, 0.03)
        material_kwargs["blade_color"][1] += np.random.normal(0.0, 0.03)
        material_kwargs["blade_color"][2] += np.random.normal(0.0, 0.03)

        material_kwargs["blight_color"] = colors.hsv2rgba(self.blight_color_hsv)

        material_kwargs["vein_color_mix_factor"] = self.vein_color_mix_factor
        material_kwargs["blight_weight"] = np.random.binomial(1, 0.1)
        material_kwargs["dotted_blight_weight"] = np.random.binomial(1, 0.1)
        material_kwargs["blight_random_seed"] = np.random.uniform(0.0, 100.0)
        material_kwargs["blight_area_factor"] = np.random.uniform(0.2, 0.8)

        # TODO: add more phenome attributes

        from . import planar_native

        # the native fields when their kernels are built, else the default graph
        native_fields = mesh_route.enabled()
        if not native_fields:
            surface.add_geomod(
                obj,
                geo_leaf_v2,
                apply=False,
                attributes=["offset", "coordinate"],
                input_kwargs=phenome,
            )
        surface.add_material(
            obj, shader_leaf_new, reuse=False, input_kwargs=material_kwargs
        )

        asset_scale = None
        if mesh_route.requested_leaf_face_size() is not None:
            asset_scale = normal(1, 0.05) * self.scale
        with mesh_route.leaf_asset_scale(asset_scale or self.scale):
            if native_fields:
                obj = planar_native.create_mesh(obj, "v2", phenome)
            else:
                obj = mesh_route.convert_active_to_mesh(obj)
        if asset_scale is None:
            asset_scale = normal(1, 0.05) * self.scale
        obj.scale *= asset_scale
        butil.apply_transform(obj)
        mesh_route.mark_final_mesh(obj)
        tag_object(obj, "leaf")

        return obj
