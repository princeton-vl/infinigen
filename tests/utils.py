
from pathlib import Path
import importlib

import gin
import bpy

from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core import init

def setup_gin(configs=None, overrides=None):
    
    gin.clear_config()
    init.apply_gin_configs(
        configs_folder='infinigen_examples/configs',
        configs=configs,
        overrides=overrides,
        skip_unknown=True
    )
    surface.registry.initialize_from_gin()


def import_item(name):
    *path_parts, name = name.split('.')
    with gin.unlock_config():
        
        try:
            return importlib.import_module('.' + name, '.'.join(path_parts))
        except ModuleNotFoundError:
            mod = importlib.import_module('.'.join(path_parts))
            return getattr(mod, name)

def load_txt_list(path):
    res = (Path(__file__).parent/path).read_text().splitlines()
    res = [f for f in res if not f.startswith('#')]
    return res

def check_factory_runs(fac_class, seed1=0, seed2=0, distance_m=50):
    butil.clear_scene()
    fac = fac_class(seed1)
    asset = fac.spawn_asset(seed2, distance=distance_m)
    assert isinstance(asset, bpy.types.Object)