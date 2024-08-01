# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import numpy as np

from .text import Text


def apply(obj, selection=None, bbox=(0, 1, 0, 1), emission=0, **kwargs):
    Text(np.random.randint(1e5), False, emission).apply(obj, selection, bbox, **kwargs)
