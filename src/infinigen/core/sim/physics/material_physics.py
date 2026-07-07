# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author
from typing import Dict

import bpy

from infinigen.core.sim.physics.material_definitions import MATERIALS, BaseMaterial


def sample_mat_physics(mat_name: str) -> Dict[str, float] | None:
    """Sample physics parameters for a material by name"""
    mat_name_lower = mat_name.lower()

    # Map material names to registry keys
    if "metal" in mat_name_lower:
        mat_key = "metal"
    elif "plastic" in mat_name_lower:
        mat_key = "plastic"
    elif "wood" in mat_name_lower:
        mat_key = "wood"
    else:
        # Use MATERIAL_DEFAULTS for unknown materials
        material_instance = MATERIALS.get(mat_name_lower, BaseMaterial)()
        return material_instance.sample_parameters()

    material_instance = MATERIALS.get(mat_key)()
    return material_instance.sample_parameters()


def get_material_properties(obj: bpy.types.Object) -> Dict:
    default_mat_physics = {"friction": 0.0, "density": 1000}
    if len(obj.data.materials) == 0:
        return default_mat_physics

    # getting material physical properties
    material = obj.data.materials[obj.data.polygons[0].material_index]
    material_name = material.name

    shader_idx = material_name.find("shader_")
    if shader_idx != -1:
        material_name = material_name[shader_idx + 7 :]

    deepcopy_idx = material_name.find("_deepcopy")
    if deepcopy_idx != -1:
        material_name = material_name[:deepcopy_idx]

    return sample_mat_physics(material_name)
