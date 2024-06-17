# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from infinigen.assets.trees import TreeFactory, BushFactory
from infinigen.assets.creatures import CarnivoreFactory
from infinigen.assets.cactus import CactusFactory
from infinigen.assets.rocks.boulder import BoulderFactory

class CachedBoulderFactory(BoulderFactory):
    pass

class CachedCactusFactory(CactusFactory):
    pass

class CachedCreatureFactory(CarnivoreFactory):
    pass

class CachedBushFactory(BushFactory):
    pass

class CachedTreeFactory(TreeFactory):
    pass