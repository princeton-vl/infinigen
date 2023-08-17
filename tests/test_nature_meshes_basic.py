from pathlib import Path

import pytest
import bpy
import gin

from infinigen.assets import * # so gin can find them
from infinigen.core.util import blender as butil

from utils import (
    setup_gin, 
    get_def_from_folder, 
    load_txt_list, 
    check_factory_runs
)

setup_gin()

@pytest.mark.parametrize('factory_name', load_txt_list('test_nature_meshes_basic.txt'))
def test_factory_runs(factory_name, **kwargs):
    fac_class = get_def_from_folder(factory_name, 'infinigen/assets')
    check_factory_runs(fac_class, **kwargs)