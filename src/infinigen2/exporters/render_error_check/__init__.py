# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from infinigen2.exporters.render_error_check.adaptive_sampling import (
    AdaptiveSamplingError,
    assert_adaptive_sampling_converged,
    configure_sample_count_output,
)
from infinigen2.exporters.render_error_check.check_frames_valid import (
    FrameCheckError,
    assert_frames_not_black,
)
from infinigen2.exporters.render_error_check.cycles_errors import (
    CyclesShaderError,
    detect_cycles_errors,
)
from infinigen2.exporters.render_error_check.displacement import (
    DisplacementCoordError,
    assert_displacement_coords_safe,
    unsafe_displacement_materials,
)
from infinigen2.exporters.render_error_check.finite_geometry import (
    NonFiniteGeometryError,
    assert_geometry_finite,
    nonfinite_vertex_counts,
)
from infinigen2.exporters.render_error_check.material_attributes import (
    MissingAttributeError,
    assert_material_attributes_present,
    missing_attribute_issues,
)
from infinigen2.exporters.render_error_check.material_nodes import (
    MaterialNodeError,
    assert_material_nodes_valid,
    normal_input_used,
    unlinked_texture_vector,
)
from infinigen2.exporters.render_error_check.mesh_quality import (
    DegenerateGeometryError,
    MeshDefects,
    assert_geometry_nondegenerate,
    coincident_face_counts,
    degenerate_geometry_defects,
    mesh_defects,
)
from infinigen2.exporters.render_error_check.object_transform import (
    SingularTransformError,
    assert_transforms_nonsingular,
    singular_transform_objects,
)
from infinigen2.exporters.render_error_check.object_visibility import (
    HiddenRenderObjectError,
    assert_render_objects_visible,
    hidden_render_objects,
)
from infinigen2.exporters.render_error_check.pre_render_validity import (
    render_validity_check,
)
from infinigen2.exporters.render_error_check.shader_complexity import (
    SHADER_NODE_COUNT_FAIL,
    ShaderTooComplexError,
    assert_shader_complexity_ok,
    count_material_nodes,
)
from infinigen2.exporters.render_error_check.uv_coords import (
    UVCoordError,
    UVLayerInfo,
    assert_uv_coords_satisfied,
    check_material_uv_coords,
)

__all__ = [
    "AdaptiveSamplingError",
    "CyclesShaderError",
    "DegenerateGeometryError",
    "DisplacementCoordError",
    "FrameCheckError",
    "HiddenRenderObjectError",
    "MeshDefects",
    "MaterialNodeError",
    "MissingAttributeError",
    "NonFiniteGeometryError",
    "ShaderTooComplexError",
    "SingularTransformError",
    "UVCoordError",
    "check_material_uv_coords",
    "coincident_face_counts",
    "count_material_nodes",
    "degenerate_geometry_defects",
    "detect_cycles_errors",
    "hidden_render_objects",
    "mesh_defects",
    "missing_attribute_issues",
    "nonfinite_vertex_counts",
    "normal_input_used",
    "render_validity_check",
    "singular_transform_objects",
    "unlinked_texture_vector",
    "unsafe_displacement_materials",
]
