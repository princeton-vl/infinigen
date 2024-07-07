# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections.abc import Iterable

import numpy as np
from numpy.random import uniform

from infinigen.assets.materials.utils import common
from infinigen.core.util.color import hex2rgba, hsv2rgba, rgb2hsv
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg

from . import (
    brushed_metal,
    galvanized_metal,
    grained_and_polished_metal,
    hammered_metal,
    metal_basic,
    metal_random,
)
