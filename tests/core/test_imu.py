# Copyright (c) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Dylan Li: primary author

import numpy as np

from infinigen.core.placement.animation_policy import keyframe
from infinigen.core.util import blender as butil
from infinigen.core.util.imu import save_imu_tum_files


def test_imu(tmp_path):
    obj = butil.spawn_cube()
    obj.name = "cube"

    keyframe(obj, loc=(0, 0, 0), rot=(0, 0, 0), t=1, interp="BEZIER")
    keyframe(obj, loc=(10, 0, 0), rot=(1, 0, 0), t=10, interp="BEZIER")

    save_imu_tum_files(tmp_path, [obj], 1, 10)

    imu_path = tmp_path / "cube_imu.txt"
    with open(imu_path) as f:
        imu_text = f.read()
        data = np.array([line.split(" ") for line in imu_text.split("\n")[1:]])
        assert data.shape == (10, 7)

    tum_path = tmp_path / "cube_tum.txt"
    with open(tum_path) as f:
        tum_text = f.read()
        data = np.array(
            [line.split(" ") for line in tum_text.split("\n")[1:]], dtype=float
        )
        assert data.shape == (10, 8)
        assert (
            data[0][0] == 1.0
            and data[0][1] == 0.0
            and data[0][2] == 0.0
            and data[0][3] == 0.0
        )
        assert (
            data[-1][0] == 10.0
            and data[-1][1] == 10.0
            and data[-1][2] == 0.0
            and data[-1][3] == 0.0
        )
