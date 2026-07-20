# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path

import procfunc as pf

from infinigen2.exporters.render_error_check.displacement import (
    assert_displacement_coords_safe,
)
from infinigen2.exporters.render_error_check.finite_geometry import (
    assert_geometry_finite,
)
from infinigen2.exporters.render_error_check.material_attributes import (
    assert_material_attributes_present,
)
from infinigen2.exporters.render_error_check.material_nodes import (
    assert_material_nodes_valid,
)
from infinigen2.exporters.render_error_check.object_transform import (
    assert_transforms_nonsingular,
)
from infinigen2.exporters.render_error_check.object_visibility import (
    assert_render_objects_visible,
)
from infinigen2.exporters.render_error_check.shader_complexity import (
    assert_shader_complexity_ok,
)
from infinigen2.exporters.render_error_check.uv_coords import assert_uv_coords_satisfied
from infinigen2.exporters.util.blender_render import DisplacementMode
from infinigen2.exporters.util.format import ExportType


@pf.tracer.primitive
def render_validity_check(
    objects: list[pf.MeshObject],
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
) -> dict[ExportType, list[Path]]:
    assert_displacement_coords_safe(objects, displacement_mode)
    assert_shader_complexity_ok(objects)
    assert_uv_coords_satisfied(objects)
    assert_material_nodes_valid(objects)
    assert_geometry_finite(objects)
    assert_transforms_nonsingular(objects)
    assert_material_attributes_present(objects)
    assert_render_objects_visible(objects)
    return {}
