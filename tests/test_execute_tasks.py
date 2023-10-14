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

from utils import setup_gin

'''
@pytest.mark.order('last')
def test_noassets_noterrain():
    args = SimpleNamespace(
        input_folder=None, 
        output_folder='/tmp/test_noassets_noterrain',
        seed="0",
        task='coarse',
        configs=['desert.gin', 'simple.gin', 'no_assets.gin'],
        overrides=['compose_scene.generate_resolution = (480, 270)'],
        task_uniqname='coarse',
        loglevel=logging.DEBUG
    )
    generate_nature.main(args)
'''

@pytest.mark.ci
def test_compose_cube():

    setup_gin()

    def compose_cube(output_folder, scene_seed, **params):
        camera_rigs = camera.spawn_camera_rigs()
        bpy.ops.mesh.primitive_cube_add()

    output = Path('/tmp/test_compose_cube')
    output.mkdir(exist_ok=True)

    execute_tasks.execute_tasks(
        compose_cube,
        input_folder=None,
        output_folder=output,
        task='coarse populate',
        scene_seed=0,
        frame_range=[0, 100],
        camera_id=(0, 0)
    )

