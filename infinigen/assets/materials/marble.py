# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from infinigen.assets.materials import common
from infinigen.assets.materials.table_materials import shader_marble


def apply(obj, selection=None, **kwargs):
    common.apply(obj, shader_marble, selection, **kwargs)
