import numpy as np

from assets.seaweed import SeaweedFactory
from nodes.node_wrangler import NodeWrangler
from placement.camera import ng_dist2camera
from placement.instance_scatter import scatter_instances


def apply(obj, scale=1, density=1., n=5, selection=None, **kwargs):
    n_species = np.random.randint(2, 3)
    factories = [SeaweedFactory(np.random.randint(1e5)) for i in range(n_species)]
                                                weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                                verbose=True, **kwargs)

    return scatter_obj, seaweeds
