# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""The scene comparison must generate the same scene on both sides."""

from infinigen.tools.perf_validation.run import _generate_command


def _configs(cmd):
    assert cmd.count("--configs") == 1  # generate_nature keeps only the last
    i = cmd.index("--configs") + 1
    configs = []
    while i < len(cmd) and not cmd[i].startswith("-"):
        configs.append(cmd[i])
        i += 1
    return configs


def test_every_side_keeps_the_scene_configs():
    for names in (None, "all", "trees,leaves"):
        cmd = _generate_command("cave", "out", ["coarse"], names)
        assert _configs(cmd)[:3] == ["cave", "simple", "dev"]


def test_all_selects_through_the_performance_config():
    assert _configs(_generate_command("cave", "out", ["coarse"], "all"))[-1] == (
        "performance"
    )
    cmd = _generate_command("cave", "out", ["coarse"], "trees,leaves")
    assert cmd[cmd.index("-p") + 1] == "optimized.packages=['trees', 'leaves']"


def test_scene_rows_compare_the_generated_assets():
    from infinigen.tools.perf_validation.geometry import describe

    a = {"asset_triangles": {"assets:TreeFactory(1)": 100, "assets:Rock(2)": 50}}
    b = {"asset_triangles": {"assets:TreeFactory(1)": 70, "assets:Rock(2)": 50}}
    line = describe(a, b, ".", ".")
    assert line.startswith("1 of 2 assets differ; assets 150 -> 120 triangles")
    assert "TreeFactory(1) 100 -> 70" in line
    assert describe(a, a, ".", ".").startswith("0 of 2 assets differ")
