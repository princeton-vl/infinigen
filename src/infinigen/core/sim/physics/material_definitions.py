# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Max Gonzalez Saez-Diez: primary author

from dataclasses import dataclass
from typing import Dict, Type

import numpy as np


@dataclass
class BaseMaterial:
    """Base material class with physics properties"""

    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 500  # kg/m³
    max_density: float = 1200  # kg/m³

    def sample_parameters(self) -> Dict[str, float]:
        """Sample random parameters within the material's ranges"""
        return {
            "friction": np.random.uniform(self.min_friction, self.max_friction),
            "density": np.random.uniform(self.min_density, self.max_density),
        }


@dataclass
class Metal(BaseMaterial):
    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 2500
    max_density: float = 5000


@dataclass
class Wood(BaseMaterial):
    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 600
    max_density: float = 1000


@dataclass
class Plastic(BaseMaterial):
    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 850
    max_density: float = 1400


@dataclass
class Ceramic(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 2300
    max_density: float = 2500


@dataclass
class Glass(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 4900
    max_density: float = 5100


@dataclass
class Marble(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 2550
    max_density: float = 2750


@dataclass
class Granite(BaseMaterial):
    min_friction: float = 0.7
    max_friction: float = 0.9
    min_density: float = 2550
    max_density: float = 2750


@dataclass
class Brick(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 1700
    max_density: float = 1900


@dataclass
class Plaster(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 600
    max_density: float = 800


@dataclass
class Fabric(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 50
    max_density: float = 250


@dataclass
class Rubber(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 1200
    max_density: float = 1400


# ===================== MATERIAL REGISTRY =====================
MATERIALS: Dict[str, Type[BaseMaterial]] = {
    "base": BaseMaterial,
    "metal": Metal,
    "wood": Wood,
    "plastic": Plastic,
    "granite": Granite,
    "brick": Brick,
    "ceramic": Ceramic,
    "glass": Glass,
    "marble": Marble,
    "plaster": Plaster,
    "fabric": Fabric,
}
