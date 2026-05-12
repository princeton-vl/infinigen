# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick

import subprocess
from pathlib import Path

import gin
import pytest

import infinigen
from infinigen.core.init import apply_gin_configs
from infinigen.datagen import manage_jobs

conf = infinigen.repo_root() / "infinigen/datagen/configs"
assert conf.exists()

compute_platforms = [x.name for x in (conf / "compute_platform").glob("*.gin")]
data_schema = [x.name for x in (conf / "data_schema").glob("*.gin")]


@pytest.mark.parametrize("compute_platform", sorted(compute_platforms))
def test_load_gin_compute_platform(compute_platform):
    gin.clear_config()

    apply_gin_configs(
        config_folders=Path("infinigen/datagen/configs"),
        configs=[compute_platform, "monocular.gin"],
        overrides=[],
        mandatory_folders=manage_jobs.mandatory_exclusive_configs,
        mutually_exclusive_folders=manage_jobs.mandatory_exclusive_configs,
    )


@pytest.mark.parametrize("data_schema", sorted(data_schema))
def test_load_gin_data_schema(data_schema):
    gin.clear_config()

    apply_gin_configs(
        config_folders=Path("infinigen/datagen/configs"),
        configs=["local_256GB.gin", data_schema],
        overrides=[],
        mandatory_folders=manage_jobs.mandatory_exclusive_configs,
        mutually_exclusive_folders=manage_jobs.mandatory_exclusive_configs,
    )


def test_dryrun_hello_world(tmp_path):
    cmd = (
        f"python -m infinigen.datagen.manage_jobs --output_folder {tmp_path}/hello_world --num_scenes 1 --specific_seed 0 "
        "--configs desert.gin simple.gin --pipeline_configs local_16GB.gin monocular.gin blender_gt.gin "
        "--pipeline_overrides LocalScheduleHandler.use_gpu=False"
    )

    cmd += " --overrides execute_tasks.dryrun=True"
    res = subprocess.run(cmd, shell=True, check=True)
    assert res.returncode == 0


def test_dryrun_hello_room(tmp_path):
    cmd = (
        f"python -m infinigen.datagen.manage_jobs --output_folder {tmp_path}/hello_room --num_scenes 1 --specific_seed 0 "
        "--pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin "
        "--configs singleroom.gin "
        "--pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' LocalScheduleHandler.use_gpu=False "
        "--overrides compose_indoors.restrict_single_supported_roomtype=True"
    )

    cmd += " execute_tasks.dryrun=True"
    res = subprocess.run(cmd, shell=True, check=True)
    assert res.returncode == 0
