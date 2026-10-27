# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Dense field rasterization and simplification for procedural assets."""

from .mesh import (
    precondition_flat_grid_regions,
    sample_masked_raster,
    simplify_geometry,
    simplify_geometry_with_fields,
)
from .reconstruction import correct_field_residuals
