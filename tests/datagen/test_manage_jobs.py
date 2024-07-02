from pathlib import Path

import pytest
import gin

import infinigen
from infinigen.datagen import manage_jobs
from infinigen.core.init import apply_gin_configs

conf = infinigen.repo_root() / "infinigen/datagen/configs"
assert conf.exists()

compute_platforms = [x.name for x in (conf / "compute_platform").glob("*.gin")]
data_schema = [x.name for x in (conf / "data_schema").glob("*.gin")]


@pytest.mark.parametrize("compute_platform", sorted(compute_platforms))
def test_load_gin_compute_platform(compute_platform):
    gin.clear_config()

    apply_gin_configs(
        configs_folder=Path("infinigen/datagen/configs"),
        configs=[compute_platform, "monocular.gin"],
        overrides=[],
        mandatory_folders=manage_jobs.mandatory_exclusive_configs,
        mutually_exclusive_folders=manage_jobs.mandatory_exclusive_configs,
    )


@pytest.mark.parametrize("data_schema", sorted(data_schema))
def test_load_gin_data_schema(data_schema):
    gin.clear_config()

    apply_gin_configs(
        configs_folder=Path("infinigen/datagen/configs"),
        configs=["local_256GB.gin", data_schema],
        overrides=[],
        mandatory_folders=manage_jobs.mandatory_exclusive_configs,
        mutually_exclusive_folders=manage_jobs.mandatory_exclusive_configs,
    )
