# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import pytest

from infinigen.core.init import configure_blender
from infinigen.core.util import blender as butil
from infinigen.core.util.test_utils import import_item, load_txt_list, setup_gin


def check_scatter_runs(pathspec):
    butil.clear_scene()
    base_cube = butil.spawn_cube()

    scatter = import_item(pathspec)
    scatter.apply(base_cube)


@pytest.mark.parametrize("pathspec", load_txt_list("tests/assets/list_scatters.txt"))
def test_scatter_runs(pathspec, **kwargs):
    configure_blender()
    setup_gin("infinigen_examples/configs_nature", ["base_nature.gin"])
    check_scatter_runs(pathspec)
