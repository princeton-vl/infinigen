# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import logging
from pathlib import Path

__version__ = "1.11.2"


def repo_root():
    return Path(__file__).parent.parent
