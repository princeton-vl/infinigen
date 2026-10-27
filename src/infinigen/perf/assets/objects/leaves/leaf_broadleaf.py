# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen broadleaf leaf (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/leaves/leaf_broadleaf.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.objects.leaves.leaf_broadleaf import (
    LeafFactoryBroadleaf as _DefaultLeafFactoryBroadleaf,
)
from infinigen.assets.objects.leaves.leaf_broadleaf import (
    geo_leaf_broadleaf,
    shader_material,
)
from infinigen.assets.objects.leaves.leaf_v2 import (
    nodegroup_apply_wave,
)
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.perf.assets.utils.heightfield import mesh_route


def apply_deferred_wave(obj):
    """Apply the broadleaf wave that the native field route defers to the final mesh."""

    y_values = list(obj["deferred_wave_y"])
    x_values = list(obj["deferred_wave_x"])
    y_points = list(zip(y_values[::2], y_values[1::2]))
    x_points = list(zip(x_values[::2], x_values[1::2]))

    def geo_deferred_wave(nw: NodeWrangler):
        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[("NodeSocketGeometry", "Geometry", None)],
        )
        x_modulated = nw.new_node(
            Nodes.NamedAttribute,
            input_kwargs={"Name": "x modulated"},
            attrs={"data_type": "FLOAT"},
        )
        apply_wave = nw.new_node(
            nodegroup_apply_wave(
                y_wave_control_points=y_points,
                x_wave_control_points=x_points,
            ).name,
            input_kwargs={
                "Geometry": group_input.outputs["Geometry"],
                "Wave Scale X": 0.2,
                "Wave Scale Y": 1.0,
                "X Modulated": x_modulated.outputs["Attribute"],
            },
        )
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": apply_wave},
        )

    surface.add_geomod(
        obj,
        geo_deferred_wave,
        name="deferred_leaf_wave",
        apply=True,
    )
    del obj["deferred_wave_y"]
    del obj["deferred_wave_x"]
    x_modulated = obj.data.attributes.get("x modulated")
    if x_modulated is not None:
        obj.data.attributes.remove(x_modulated)


class LeafFactoryBroadleaf(_DefaultLeafFactoryBroadleaf):
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
            (uniform(0.25, 0.75), uniform(0.50, 0.60)),
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
        # the native fields when their kernels are built, else the default graph,
        # which applies the wave itself
        native_fields = mesh_route.enabled()

        material_kwargs = phenome.copy()
        material_kwargs["color_base"] = np.copy(
            self.blade_color
        )  # (0.2346, 0.4735, 0.0273, 1.0),
        material_kwargs["color_base"][0] += np.random.normal(0.0, 0.02)
        material_kwargs["color_base"][1] += np.random.normal(0.0, self.color_randomness)
        material_kwargs["color_base"][2] += np.random.normal(0.0, self.color_randomness)
        material_kwargs["color_base"] = hsv2rgba(material_kwargs["color_base"])

        material_kwargs["color_vein"] = np.copy(
            self.vein_color
        )  # (0.2346, 0.4735, 0.0273, 1.0),
        material_kwargs["color_vein"][0] += np.random.normal(0.0, 0.02)
        material_kwargs["color_vein"][1] += np.random.normal(0.0, self.color_randomness)
        material_kwargs["color_vein"][2] += np.random.normal(0.0, self.color_randomness)
        material_kwargs["color_vein"] = hsv2rgba(material_kwargs["color_vein"])

        surface.add_material(
            obj, shader_material, reuse=False, input_kwargs=material_kwargs
        )
        asset_scale = None
        if mesh_route.requested_leaf_face_size() is not None:
            asset_scale = normal(1, 0.1) * self.scale
        with mesh_route.leaf_asset_scale(asset_scale or self.scale):
            if native_fields:
                from .broadleaf_native import rasterize_broadleaf_native_fields

                fields = rasterize_broadleaf_native_fields(
                    mesh_route.raster_shape("broadleaf")[0], phenome
                )
                obj = mesh_route.convert_raster_fields_to_mesh(obj, "broadleaf", fields)
            else:
                surface.add_geomod(
                    obj,
                    geo_leaf_broadleaf,
                    apply=False,
                    attributes=["offset", "coordinate", "subvein offset", "vein value"],
                    input_kwargs=phenome,
                )
                obj = mesh_route.convert_active_to_mesh(obj)
        if asset_scale is None:
            asset_scale = normal(1, 0.1) * self.scale
        obj.scale *= asset_scale
        if native_fields:
            obj["deferred_wave_y"] = [
                value for point in phenome["y_wave_control_points"] for value in point
            ]
            obj["deferred_wave_x"] = [
                value for point in phenome["x_wave_control_points"] for value in point
            ]
            apply_deferred_wave(obj)
        butil.apply_transform(obj)
        mesh_route.mark_final_mesh(obj)
        tag_object(obj, "leaf_broadleaf")

        return obj
