# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yihan Wang, Karhan Kayan: face-based tagging, canonical surface tagging, mask extraction and support tags in the original Infinigen tagging (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/tagging.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.core import surface


def _unique_tagints(values):
    """`np.unique` of non-negative tag ids, by a bincount instead of a sort."""
    if len(values) == 0 or values.min() < 0:
        return np.unique(values)
    return np.flatnonzero(np.bincount(values))


def _write_mask_tag(obj, attr, tagint):
    # the optimized write_attr_data casts to the property's dtype, so
    # foreach_set copies the buffer instead of converting per element
    surface.write_attr_data(obj, attr, tagint)
