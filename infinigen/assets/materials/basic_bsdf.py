# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils

import numpy as np
from numpy.random import uniform, normal

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba
from infinigen.core import surface

def shader_basic_bsdf(nw):

    color = nw.new_node(Nodes.RGB)
    color.outputs[0].default_value = hsv2rgba(uniform(0.05, 0.95, 3))

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={
            'Base Color': color,
            'Roughness': np.clip(normal(0.6, 0.3), 0.05, 0.95),
            'Metallic': uniform(0, 1) if uniform() < 0.3 else 0,
            'Subsurface': 0 if uniform() < 0.8 else uniform(0, 0.2)
        },
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

    return principled_bsdf

def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_basic_bsdf, reuse=False)