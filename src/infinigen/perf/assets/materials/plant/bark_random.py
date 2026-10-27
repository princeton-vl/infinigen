# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen random bark material (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/materials/plant/bark_random.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.assets.materials.plant import bark_random as _default
from infinigen.core import surface

# BarkRandom's displacement parameters, in the order the native kernel takes them
_BARK_PARAMETERS = (
    "Displacement Scale",
    "Z Noise Scale",
    "Z Noise Amount",
    "Z Multiplier",
    "Primary Voronoi Scale",
    "Primary Voronoi Randomness",
    "Secondary Voronoi Mix Weight",
    "Secondary Voronoi Scale",
    "Noise Texture Scale",
    "Noise Texture Detail",
    "Noise Texture Weight",
)


class BarkRandom:
    """The method of the default BarkRandom that differs."""

    def apply(self, obj, selection=None, **kwargs):
        """Defer a natively meshed tree's bark displacement to the native kernel, run
        once the stems are retessellated, instead of displacing the tubes with
        geometry nodes. Trees call this unbound with the tree factory as `self`; any
        other call runs the default."""
        from infinigen.perf.assets.objects.trees.generate import native_meshing
        from infinigen.perf.assets.utils import _native_fields, surface_fields

        factory = self
        if (
            selection is not None
            or not hasattr(factory, "adapt_mesh_method")
            or not hasattr(obj, "data")
            or not native_meshing(factory)
        ):
            return _default.BarkRandom.apply.default(
                self, obj, selection=selection, **kwargs
            )

        geo_params, color_params = _default.get_random_bark_params(
            seed=np.random.randint(1e5)
        )
        parameters = np.asarray(
            [geo_params[key] for key in _BARK_PARAMETERS], dtype=np.float32
        )
        material = surface.shaderfunc_to_material(
            _default.shader_random_bark_mat,
            geo_params=geo_params,
            base_color_hsv=color_params["Color"],
        )

        def evaluate(positions, attributes):
            return _native_fields.evaluate_bark_displacement(
                positions,
                attributes["parent_skeleton_loc"],
                attributes["skeleton_loc"],
                parameters,
            )

        surface_fields.defer_scalar_displacement(
            obj,
            evaluate,
            attributes=("parent_skeleton_loc", "skeleton_loc"),
            material=material,
            capture_position_as="initial_position",
        )
