# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import bpy

from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import usage_lookup
from infinigen.core.util import blender as butil
from infinigen_examples.constraints.home import home_asset_usage


def test_canonical_planes_real_placeholders():
    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)

    pholder_facs = usage_lookup.factories_for_usage({t.Semantics.RealPlaceholder})
    asset_facs = usage_lookup.factories_for_usage({t.Semantics.AssetAsPlaceholder})
    test_facs = pholder_facs.union(asset_facs)

    test_facs.intersection_update(
        usage_lookup.factories_for_usage({t.Semantics.Storage}).union(
            usage_lookup.factories_for_usage({t.Semantics.Seating})
        )
    )

    for fac in test_facs:
        butil.clear_scene()

        if fac in pholder_facs:
            obj = fac(0).spawn_placeholder(0, loc=(0, 0, 0), rot=(0, 0, 0))
        elif fac in asset_facs:
            obj = fac(0).spawn_asset(0, loc=(0, 0, 0), rot=(0, 0, 0))
        else:
            raise ValueError()

        with butil.ViewportMode(obj, mode="EDIT"):
            butil.select(obj)
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.quads_convert_to_tris(
                quad_method="BEAUTY", ngon_method="BEAUTY"
            )

        tagging.tag_canonical_surfaces(obj)

        obj_tags = tagging.union_object_tags(obj)

        for tag in [t.Subpart.Back, t.Subpart.Bottom]:  # , t.Subpart.SupportSurface]:
            mask = tagging.tagged_face_mask(obj, {tag})
            if mask.sum() == 0:
                obj_tags = tagging.union_object_tags(obj)
                raise ValueError(
                    f"{obj.name=} has nothing tagged for {tag=}. {obj_tags=}"
                )
