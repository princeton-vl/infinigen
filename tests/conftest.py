import bpy
import gin
import pytest


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    gin.clear_config()
    bpy.ops.wm.read_factory_settings(use_empty=True)
