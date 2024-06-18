# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from infinigen.assets.materials import leather_and_fabrics

from .leather_and_fabrics import *


def apply(obj, selection=None, **kwargs):
    leather_and_fabrics.apply(obj, selection=selection, **kwargs)
