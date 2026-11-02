# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import os

import bpy
import pytest

import infinigen
from infinigen.core.init import configure_cycles_devices
from infinigen.perf import options


@pytest.fixture
def cycles(monkeypatch, tmp_path):
    bpy.context.scene.render.engine = "CYCLES"
    monkeypatch.delenv("OPTIX_CACHE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield tmp_path
    infinigen.use_optimized(False)


def test_nothing_selected_leaves_the_cache_to_the_driver(cycles):
    infinigen.use_optimized(False)
    configure_cycles_devices()
    assert "OPTIX_CACHE_PATH" not in os.environ


def test_runtime_keeps_the_cache_in_a_per_user_directory(cycles):
    infinigen.use_optimized(["runtime"])
    configure_cycles_devices()
    expected = cycles / "infinigen" / "optix"
    assert os.environ["OPTIX_CACHE_PATH"] == str(expected)
    assert expected.is_dir()


def test_a_set_cache_path_is_kept_and_the_cache_can_be_off(cycles, monkeypatch):
    infinigen.use_optimized(["runtime"])
    monkeypatch.setenv("OPTIX_CACHE_PATH", "/somewhere/else")
    configure_cycles_devices()
    assert os.environ["OPTIX_CACHE_PATH"] == "/somewhere/else"

    monkeypatch.delenv("OPTIX_CACHE_PATH")
    # CPU renders never set it
    configure_cycles_devices(use_gpu=False)
    assert "OPTIX_CACHE_PATH" not in os.environ
    # optix_kernel_cache.enabled = False
    monkeypatch.setattr(options, "optix_kernel_cache", lambda: None)
    configure_cycles_devices()
    assert "OPTIX_CACHE_PATH" not in os.environ
