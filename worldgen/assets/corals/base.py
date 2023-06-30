import bpy

from placement.factory import AssetFactory


class BaseCoralFactory(AssetFactory):
    
    tentacle_distance = .05
    default_scale = [.8] * 3
    noise_strength = .02
    tentacle_prob = .5
    bump_prob = .3
    density = 500

    def __init__(self, factory_seed, coarse=False):
        super(BaseCoralFactory, self).__init__(factory_seed, coarse)
        self.points_fn = lambda nw, points: points

    def create_asset(self, **params) -> bpy.types.Object:
        raise NotImplemented
