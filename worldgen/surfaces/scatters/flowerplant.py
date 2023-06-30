from placement.instance_scatter import scatter_instances
from placement.factory import AssetFactory, make_asset_collection
from assets.grassland.flowerplant import FlowerPlantFactory
from surfaces.scatters.utils.wind import wind

def apply(obj, selection=None, density=1.0):
    flowerplant_col = make_asset_collection(FlowerPlantFactory(np.random.randint(1e5)), n=12, verbose=True)
    
    density = np.clip(density / avg_vol, 0, 200)
    scatter_obj = scatter_instances(
        base_obj=obj, collection=flowerplant_col,
        scale=1.5, scale_rand=0.7, scale_rand_axi=0.2,
        density=float(density),
        ground_offset=0, normal_fac=0.3,
        rotation_offset=wind(strength=20),
        selection=selection, taper_scale=True
    )
