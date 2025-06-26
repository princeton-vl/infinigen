# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Original code: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/mjcf_utils.py
# Modified by Abhishek Joshi

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
