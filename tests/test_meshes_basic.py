from pathlib import Path

import pytest
import bpy
import gin

from infinigen.core.util import blender as butil

from utils import (
    setup_gin, 
    import_item, 
    load_txt_list, 
    check_factory_runs
)

setup_gin()

@pytest.mark.ci
@pytest.mark.parametrize('pathspec', load_txt_list('test_meshes_basic.txt'))
def test_factory_runs(pathspec, **kwargs):
    fac_class = import_item(pathspec)
    check_factory_runs(fac_class, **kwargs)