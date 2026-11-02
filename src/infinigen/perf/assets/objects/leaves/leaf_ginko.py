# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen ginkgo leaf (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/leaves/leaf_ginko.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.objects.leaves.leaf_ginko import (
    LeafFactoryGinko as _DefaultLeafFactoryGinko,
)
from infinigen.assets.objects.leaves.leaf_ginko import (
    geo_leaf_ginko,
    shader_material,
)
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.perf.assets.utils.heightfield import mesh_route


class LeafFactoryGinko(_DefaultLeafFactoryGinko):
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

        phenome["stem_curve_control_points"] = [
            (0.0, 0.5),
            (uniform(0.2, 0.3), uniform(0.45, 0.55)),
            (uniform(0.7, 0.8), uniform(0.45, 0.55)),
            (1.0, 0.5),
        ]
        phenome["shape_curve_control_points"] = [
            (0.0, 0.0),
            (0.523, 0.1156),
            (0.5805, 0.7469),
            (0.7742, 0.7719),
            (0.9461, 0.7531),
            (1.0, 0.0),
        ]
        phenome["vein_length"] = uniform(0.4, 0.5)
        phenome["angle"] = uniform(-110.0, -70.0)
        phenome["multiplier"] = uniform(1.90, 1.98)

        phenome["scale_vein"] = uniform(70.0, 90.0)
        phenome["scale_wave"] = uniform(4.0, 6.0)
        phenome["scale_margin"] = uniform(5.5, 7.5)

        material_kwargs = phenome.copy()
        material_kwargs["color_base"] = np.copy(
            self.blade_color
        )  # (0.2346, 0.4735, 0.0273, 1.0),
        material_kwargs["color_base"][0] += np.random.normal(0.0, 0.02)
        material_kwargs["color_base"][1] += np.random.normal(0.0, self.color_randomness)
        material_kwargs["color_base"][2] += np.random.normal(0.0, self.color_randomness)
        material_kwargs["color_base"] = hsv2rgba(material_kwargs["color_base"])

        material_kwargs["color_vein"] = hsv2rgba(np.copy(self.blade_color))

        from . import planar_native

        # the native fields when their kernels are built, else the default graph
        native_fields = mesh_route.enabled()
        if not native_fields:
            surface.add_geomod(
                obj,
                geo_leaf_ginko,
                apply=False,
                attributes=["vein", "shape"],
                input_kwargs=phenome,
            )
        surface.add_material(
            obj, shader_material, reuse=False, input_kwargs=material_kwargs
        )

        asset_scale = None
        if mesh_route.requested_leaf_face_size() is not None:
            asset_scale = normal(1, 0.2) * self.scale
        with mesh_route.leaf_asset_scale(asset_scale or self.scale):
            if native_fields:
                obj = planar_native.create_mesh(obj, "ginko", phenome)
            else:
                obj = mesh_route.convert_active_to_mesh(obj)
        if asset_scale is None:
            asset_scale = normal(1, 0.2) * self.scale
        obj.scale *= asset_scale
        butil.apply_transform(obj)
        mesh_route.mark_final_mesh(obj)
        tag_object(obj, "leaf_ginko")

        return obj
