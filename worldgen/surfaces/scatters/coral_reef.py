import numpy as np

from assets.corals.generate import CoralFactory, TableCoralFactory
from nodes.node_wrangler import NodeWrangler
from nodes import node_utils
from placement.camera import ng_dist2camera


    if horizontal:
        return apply_horizontal(obj, scale, density, n, selection)
    else:
        return apply_all(obj, scale, density, n, selection)


def apply_all(obj, scale=1, density=5., n=12, selection=None):
    n_species = np.random.randint(5, 10)


    return scatter_obj, corals


def apply_horizontal(obj, scale=1, density=5., n=4, selection=None):
    n_species = np.random.randint(2, 3)
    factories = [TableCoralFactory(np.random.randint(1e5)) for _ in range(n_species)]
                                              weights=np.random.uniform(0.8, 1, len(factories)), n=n,
                                              verbose=True)

    def scaling(nw):
        basic = nw.uniform(1. * scale, 2. * scale)
        camera_based = nw.build_float_curve(nw.new_node(ng_dist2camera().name), [(0, .2), (4, 1)])
        return nw.vector_math('MINIMUM', basic, camera_based)


    return scatter_obj, corals
