# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

from pathlib import Path

import infinigen.assets.sim_objects.blueprints as blueprints


def blueprint_path_completion(blueprint_path, root=None):
    """
    Returns the absolute path to an assets blueprint.
    logic borrowed and adapted from https://github.com/ARISE-Initiative/robosuite
    """
    if blueprint_path.startswith("/"):
        full_path = blueprint_path
    else:
        if root is None:
            root = blueprints
        full_path = Path(root.__path__[0]) / blueprint_path
    return Path(full_path) if isinstance(full_path, str) else full_path
