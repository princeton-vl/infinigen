import numpy as np
from numpy.random import uniform
import gin
from util.random import random_general as rg
from nodes.node_wrangler import Nodes
from nodes import color
from surfaces import surface

type = None


@gin.configurable
def shader_atmosphere(nw, enable_scatter=True, density=("uniform", 0, 0.006), anisotropy=0.5, **kwargs):
    nw.force_input_consistency()

    principled_volume = nw.new_node(Nodes.PrincipledVolume,
        input_kwargs={
            'Color': color.color_category('fog'),
            'Density': rg(density),
            'Anisotropy': rg(anisotropy)
        })
    
    return (None, principled_volume)

def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_atmosphere, selection=selection)
