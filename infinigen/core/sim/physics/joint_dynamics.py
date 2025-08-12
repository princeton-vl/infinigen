# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author
from typing import Dict


def get_joint_properties(joint_name: str, joint_params: Dict) -> Dict:
    if joint_name not in joint_params:
        return {"stiffness": 0, "damping": 0, "friction": 0}
    res = {
        "stiffness": joint_params[joint_name].get("stiffness", 0.0),
        "damping": joint_params[joint_name].get("damping", 0.0),
        "friction": joint_params[joint_name].get("friction", 0.0),
    }

    return res
