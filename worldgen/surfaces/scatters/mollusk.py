import numpy as np

from assets.mollusk import MolluskFactory
from assets.utils.misc import CountInstance
from assets.utils.decorate import toggle_hide
from util import blender as butil
from nodes import node_utils
from surfaces import surface


    with CountInstance('mollusk'):
        n_species = np.random.randint(4, 6)
        factories = list(MolluskFactory(np.random.randint(1e5)) for _ in range(n_species))
                                                    weights=np.random.uniform(0.5, 1, len(factories)), n=n,
                                                    verbose=True)

        def scaling(nw):
            return nw.uniform([.4 * scale] * 3, [.8 * scale] * 3, data_type='FLOAT_VECTOR')


        return scatter_obj, mollusk
