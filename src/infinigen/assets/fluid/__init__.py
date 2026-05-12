# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

from .asset_cache import FireCachingSystem
from .cached_factory_wrappers import (
    CachedBoulderFactory,
    CachedBushFactory,
    CachedCactusFactory,
    CachedCreatureFactory,
    CachedTreeFactory,
)
from .flip_fluid import make_beach, make_river, make_still_water, make_tilted_river
from .fluid import set_fire_to_assets
from .fluid_scenecomp_additions import cached_fire_scenecomp_options
