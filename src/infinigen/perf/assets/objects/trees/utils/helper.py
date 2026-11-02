# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alejandro Newell: original Infinigen tree helpers (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/trees/utils/helper.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.assets.objects.trees.utils import helper as _default


def rodrigues_rot(vec, axis, angle, convert_to_rad=False):
    """Default rotation of one 3-vector written out element-wise, which is cheaper
    than numpy's cross and norm on three values."""
    if np.ndim(vec) != 1 or np.ndim(axis) != 1:
        return _default.rodrigues_rot.default(vec, axis, angle, convert_to_rad)
    axis = axis / np.sqrt(np.dot(axis, axis))
    cs, sn = _default.get_cos_sin(angle, convert_to_rad)
    cross = (
        axis[1] * vec[2] - axis[2] * vec[1],
        axis[2] * vec[0] - axis[0] * vec[2],
        axis[0] * vec[1] - axis[1] * vec[0],
    )
    projection = np.dot(axis, vec)
    complement = 1 - cs
    return np.array(
        (
            vec[0] * cs + sn * cross[0] + axis[0] * projection * complement,
            vec[1] * cs + sn * cross[1] + axis[1] * projection * complement,
            vec[2] * cs + sn * cross[2] + axis[2] * projection * complement,
        )
    )
