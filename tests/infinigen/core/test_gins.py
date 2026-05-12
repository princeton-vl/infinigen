# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


import pytest

import infinigen
from infinigen.core.util.test_utils import setup_gin

nature_folder = "infinigen_examples/configs_nature"
nature_gins = [p.name for p in (infinigen.repo_root() / nature_folder).glob("**/*.gin")]


@pytest.mark.parametrize("extra_gin", sorted(nature_gins))
def test_gins_load_nature(extra_gin):
    # gin must successfully load the config without crashing
    # common failures are misspellings of config fields, renamed functions, etc
    setup_gin(nature_folder, configs=[extra_gin])


indoor_folder = "infinigen_examples/configs_indoor"
indoor_gins = [p.name for p in (infinigen.repo_root() / indoor_folder).glob("**/*.gin")]


@pytest.mark.parametrize("extra_gin", sorted(indoor_gins))
def test_gins_load_indoor(extra_gin):
    # gin must successfully load the config without crashing
    # common failures are misspellings of config fields, renamed functions, etc
    setup_gin([indoor_folder, nature_folder], configs=[extra_gin])
