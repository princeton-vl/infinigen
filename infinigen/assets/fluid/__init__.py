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
