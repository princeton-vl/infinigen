# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path
from types import SimpleNamespace
import logging
import importlib

import pytest
import bpy
import gin

from infinigen_examples import generate_nature
from infinigen.core import execute_tasks
from infinigen.core.placement import camera
from infinigen.core import init

from infinigen_examples.util.test_utils import setup_gin

nature_folder = 'infinigen_examples/configs_nature'
nature_gins = [p.name for p in (init.repo_root()/nature_folder).glob('**/*.gin')]

@pytest.mark.parametrize('extra_gin', sorted(nature_gins))
def test_gins_load_nature(extra_gin):
    # gin must successfully load the config without crashing
    # common failures are misspellings of config fields, renamed functions, etc
    setup_gin(nature_folder, configs=[extra_gin])

indoor_folder = 'infinigen_examples/configs_indoor' 
indoor_gins = [p.name for p in (init.repo_root()/indoor_folder).glob('**/*.gin')]

@pytest.mark.parametrize('extra_gin', sorted(indoor_gins))
def test_gins_load_indoor(extra_gin):
    # gin must successfully load the config without crashing
    # common failures are misspellings of config fields, renamed functions, etc
    setup_gin(indoor_folder, configs=[extra_gin])
