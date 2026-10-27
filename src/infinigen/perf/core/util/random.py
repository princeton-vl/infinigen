# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma, Alexander Raistrick: original Infinigen random utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/util/random.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.core.util import random as _default


def random_general(var):
    """`random_general` with `weighted_choice` searching one `random_sample()` in the
    CDF of `p`, the same draw and index as `np.random.choice` without its checks."""
    if isinstance(var, (tuple, list)) and var and var[0] == "weighted_choice":
        weights, recargs = zip(*var[1:])
        p = np.array(weights) / sum(weights)
        cdf = p.cumsum()
        cdf /= cdf[-1]
        i = int(cdf.searchsorted(np.random.random_sample(), side="right"))
        return _default.random_general(recargs[i])
    return _default.random_general.default(var)
