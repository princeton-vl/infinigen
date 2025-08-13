# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author
from typing import Dict

import bpy
import gin
import numpy as np


def sample_parameters(
    min_friction: float,
    max_friction: float,
    min_density: float,
    max_density: float,
):
    return {
        "friction": np.random.uniform(min_friction, max_friction),
        "density": np.random.uniform(min_density, max_density),
    }


MATERIAL_DEFAULTS = {
    "granite": {
        "min_friction": 0.7,
        "max_friction": 0.9,
        "min_density": 2550,
        "max_density": 2750,
    },
    "brick": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 1700,
        "max_density": 1900,
    },
    "aluminum": {
        "min_friction": 0.4,
        "max_friction": 0.6,
        "min_density": 2600,
        "max_density": 2800,
    },
    "ceramic": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 2300,
        "max_density": 2500,
    },
    "glass": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 4900,
        "max_density": 5100,
    },
    "marble": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 2550,
        "max_density": 2750,
    },
    "plaster": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 600,
        "max_density": 800,
    },
    "fabric": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 50,
        "max_density": 250,
    },
    "rubber": {
        "min_friction": 0.9,
        "max_friction": 1.1,
        "min_density": 1200,
        "max_density": 1400,
    },
    "wood": {
        "min_friction": 0.8,
        "max_friction": 1.2,
        "min_density": 600,
        "max_density": 800,
    },
    "metal": {
        "min_friction": 0.8,
        "max_friction": 1.2,
        "min_density": 7000,
        "max_density": 8000,
    },
    "plastic": {
        "min_friction": 0.8,
        "max_friction": 1.2,
        "min_density": 1000,
        "max_density": 1400,
    },
    # add new materials here
}


def make_material_sampler(material_name, defaults):
    # define the sampler

    @gin.configurable(module=material_name)
    def physics_sampler(
        min_friction=defaults["min_friction"],
        max_friction=defaults["max_friction"],
        min_density=defaults["min_density"],
        max_density=defaults["max_density"],
    ):
        return sample_parameters(min_friction, max_friction, min_density, max_density)

    return physics_sampler


MATERIAL_SAMPLERS = {
    name: make_material_sampler(name, defs) for name, defs in MATERIAL_DEFAULTS.items()
}


def sample_mat_physics(mat_name: str):
    if mat_name not in MATERIAL_SAMPLERS:
        return None
    return MATERIAL_SAMPLERS[mat_name]()


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

    # get the material physics properties
    mat_physics = sample_mat_physics(material_name)
    return default_mat_physics if mat_physics is None else mat_physics
