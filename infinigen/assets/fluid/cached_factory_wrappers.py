# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from infinigen.assets.objects.cactus import CactusFactory
from infinigen.assets.objects.creatures import CarnivoreFactory
from infinigen.assets.objects.rocks.boulder import BoulderFactory
from infinigen.assets.objects.trees import BushFactory, TreeFactory


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
