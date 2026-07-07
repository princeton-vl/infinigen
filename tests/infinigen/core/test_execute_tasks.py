# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path

import bpy

from infinigen.core import execute_tasks
from infinigen.core.placement import camera
from infinigen.core.util.test_utils import setup_gin


def test_compose_cube():
    setup_gin("infinigen_examples/configs_nature", configs=["base_nature.gin"])

    def compose_cube(output_folder, scene_seed, **params):
        camera.spawn_camera_rigs()
        bpy.ops.mesh.primitive_cube_add()

    output = Path("/tmp/test_compose_cube")
    output.mkdir(exist_ok=True)

    execute_tasks.execute_tasks(
        compose_cube,
        populate_scene_func=None,
        input_folder=None,
        output_folder=output,
        task="coarse populate",
        scene_seed=0,
        frame_range=[0, 100],
        camera_id=(0, 0),
    )
