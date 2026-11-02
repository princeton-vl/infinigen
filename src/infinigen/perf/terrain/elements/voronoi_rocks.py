# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen Voronoi rock terrain element (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/elements/voronoi_rocks.py)
# - David Yan, Jia Deng: optimized implementation


def _configure_library(element):
    """Turn on the rocks library's per-thread cache of the mask noise and attachment
    SDF at each Voronoi cell centre, which many query points share. Keyed by the
    centre's float bits, so results are bitwise identical to uncached evaluation. CPU
    only; a library without `set_attachment_cache` is left as is."""
    setter = getattr(element.dll, "set_attachment_cache", None)
    if element.device == "cpu" and setter is not None:
        setter(1)
