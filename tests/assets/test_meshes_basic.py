# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path

import bpy
import pytest

from infinigen.core import tagging
from infinigen.core.init import configure_blender
from infinigen.core.util import blender as butil
from infinigen.core.util.test_utils import import_item, load_txt_list, setup_gin


def check_factory_runs(fac_class, seed1=0, seed2=0, distance_m=50):
    butil.clear_scene()
    fac = fac_class(seed1)
    asset = fac.spawn_asset(seed2, distance=distance_m)

    if not isinstance(asset, bpy.types.Object):
        raise ValueError(f"{asset.name=} had {type(asset)=}")

    if tuple(asset.location) != (0, 0, 0):
        raise ValueError(f"{asset.location=}")
    if tuple(asset.rotation_euler) != (0, 0, 0):
        raise ValueError(f"{asset.rotation_euler=}")
    if tuple(asset.scale) != (1, 1, 1):
        raise ValueError(f"{asset.scale=}")

    # currently, assets may have objects as '.children'.
    # This will eventually be removed except well-documented special cases
    for o in butil.iter_object_tree(asset):
        for i, slot in enumerate(o.material_slots):
            if slot.material is None:
                raise ValueError(f"In {asset.name=} {slot=} had {slot.material=}")

        for mod in asset.modifiers:
            if mod.type != "NODES" and mod.type != "SUBSURF":
                # currently we allow unapplied non-modifiers for things like time-based deformation on
                # seaweed etc. NODES and SUBSURF should still always be applied.
                continue
            raise ValueError(
                f"In {asset.name=} {o.name=} had unapplied modifier {mod.name=} {mod.type=} "
            )

        if o.type != "MESH":
            continue

        if o.data is None:
            raise ValueError(f"In {asset.name=} {o.name=} had {o.data=}")

        if len(o.data.vertices) <= 2:
            raise ValueError(
                f"{asset.name=} had {len(o.data.vertices)} vertices, usually indicates failed operation"
            )

        if tagging.COMBINED_ATTR_NAME in o.data.attributes:
            attr = o.data.attributes[tagging.COMBINED_ATTR_NAME]
            if attr.domain != "FACE":
                raise ValueError(
                    f"In {asset.name=} had {attr.domain=} for {attr.name=}. Should be FACE"
                )

        # some objects like the older LeafFactory
        # if len(o.data.polygons) < 2:
        #    raise ValueError(f'{asset.name=} had {len(o.data.polygons)} polygons, usually indicates failed operation')

        for attr in o.data.attributes:
            if attr.name.startswith(tagging.PREFIX):
                raise ValueError(
                    f"In {asset.name}, {o.name=} had un-merged tag-system tag {attr.name=}, need to call {tagging.tag_system.relabel_obj}"
                )


@pytest.mark.nature
@pytest.mark.parametrize(
    "pathspec", load_txt_list(Path(__file__).parent / "list_nature_meshes.txt")
)
def test_nature_factory_runs(pathspec, **kwargs):
    setup_gin("infinigen_examples/configs_nature", configs=["base_nature.gin"])
    configure_blender()
    fac_class = import_item(pathspec)
    check_factory_runs(fac_class, **kwargs)


@pytest.mark.parametrize(
    "pathspec", load_txt_list(Path(__file__).parent / "list_indoor_meshes.txt")
)
def test_indoor_factory_runs(pathspec, **kwargs):
    setup_gin(
        ["infinigen_examples/configs_indoor", "infinigen_examples/configs_nature"],
        configs=["base_indoors.gin"],
    )
    fac_class = import_item(pathspec)
    check_factory_runs(fac_class, **kwargs)
