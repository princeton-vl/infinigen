# Copyright (C) 2023, Princeton University.
# This source code is licensed under the GPL license found in the LICENSE file in the root directory of this
# source tree.

# Authors: Mingzhe Wang, Lingjie Mei


from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.materials.plastics.plastic_rough import shader_rough_plastic
from infinigen.assets.materials.plastics.plastic_translucent import (
    shader_translucent_plastic,
)


def apply(obj, selection=None, clear=None, **kwargs):
    is_rough = kwargs.get("rough", uniform(0, 1))
    is_translucent = kwargs.get("translucent", uniform(0, 1))
    if clear is None:
        clear = uniform() < 0.2
    shader_func = (
        shader_rough_plastic
        if is_rough > is_translucent
        else shader_translucent_plastic
    )
    common.apply(obj, shader_func, selection, clear=clear, **kwargs)
