import numpy as np

from assets.urchin import UrchinFactory
from nodes.node_wrangler import NodeWrangler
from placement.camera import ng_dist2camera


    n_species = np.random.randint(2, 3)
    factories = list(UrchinFactory(np.random.randint(1e5)) for i in range(n_species))
                                              weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                              verbose=True)
    def ground_offset(nw: NodeWrangler):
        return nw.uniform(.4 * scale, .8 * scale)

        base_obj=obj, collection=urchin,

    return scatter_obj, urchin
