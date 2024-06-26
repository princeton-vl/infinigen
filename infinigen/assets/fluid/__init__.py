from .fluid_scenecomp_additions import cached_fire_scenecomp_options
from .fluid import set_fire_to_assets
from .asset_cache import FireCachingSystem
from .cached_factory_wrappers import (
    CachedBoulderFactory, 
    CachedBushFactory,
    CachedCactusFactory, 
    CachedCreatureFactory, 
    CachedTreeFactory
)
from .flip_fluid import (
    make_river,
    make_still_water,
    make_tilted_river,
    make_beach
)