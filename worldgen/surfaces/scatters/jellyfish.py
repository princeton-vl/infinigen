import numpy as np

from assets.jellyfish import JellyfishFactory
from nodes.node_wrangler import NodeWrangler


def apply(obj, scale=1, density=1., n=10, selection=None):
    n_species = np.random.randint(2, 3)
    factories = list(JellyfishFactory(np.random.randint(1e5)) for i in range(n_species))
                                                 weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                                 verbose=True)

    def ground_offset(nw: NodeWrangler):

        base_obj=obj, collection=jellyfish,
        scale=scale, scale_rand=U(0.2, 0.9),
        normal_fac=0.0,
        rotation_offset=lambda nw: nw.uniform(3*(-r,), 3*(r,)), reset_children=False,
    return scatter_obj, jellyfish
