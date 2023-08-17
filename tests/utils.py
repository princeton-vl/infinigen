
from pathlib import Path
import importlib

import gin
import bpy

from infinigen.core import surface
from infinigen.core.util import blender as butil

def setup_gin():
    
    gin.clear_config()

    gin.parse_config_files_and_bindings(
        config_files=['examples/configs/base.gin'],
        bindings=None,
        skip_unknown=True
    )

    surface.registry.initialize_from_gin()


def get_def_from_folder(name, folder):
    root = Path(__file__).parent.parent
    for file in (root/folder).iterdir():
        with gin.unlock_config():
            module_parent = str(file.parent.relative_to(root)).replace('/', '.')
            module = importlib.import_module(f'{module_parent}.{file.stem}')
        if hasattr(module, name):
            return getattr(module, name)

    raise ModuleNotFoundError(f'Could not find any factory with {name=}, make sure it is imported by a direct descendent of infinigen.assets')

def load_txt_list(path):
    res = (Path(__file__).parent/path).read_text().splitlines()
    res = [f for f in res if not f.startswith('#')]
    return res

def check_factory_runs(fac_class, seed1=0, seed2=0, distance_m=50):
    butil.clear_scene()
    fac = fac_class(seed1)
    asset = fac.spawn_asset(seed2, distance=distance_m)
    assert isinstance(asset, bpy.types.Object)