# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Leaf meshes simplified straight from dense rasters of the default leaves' fields.

The default decimates 97% of its fine grid; the raster keeps the grid's density, as
sampling near the final face count aliases narrow veins and the outline.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

# the default leaf collection decimates by 0.97
DEFAULT_COLLECTION_RETENTION = 0.03
_TARGET_FACE_SCALE: ContextVar[float] = ContextVar(
    "infinigen_leaf_target_face_scale", default=1.0
)
_TARGET_FACE_SIZE: ContextVar[float | None] = ContextVar(
    "infinigen_leaf_target_face_size", default=None
)
_ASSET_SCALE: ContextVar[float] = ContextVar("infinigen_leaf_asset_scale", default=1.0)

# as_pointer() of the meshes built here, kept out of the objects' custom properties
_FINAL_MESH_POINTERS: set[int] = set()

# raster sizes at the default leaves' subdivision density
DEFAULT_RASTER_SHAPES = {
    "broadleaf": (1025, 1025),
    "v2": (1025, 1025),
    "maple": (2049, 2049),
    "ginko": (513, 513),
}

# Fraction of the raster's triangles that the default's decimation keeps. Maple's
# raster has twice the samples per axis of its default mesh, hence a quarter of 0.03.
LEGACY_DENSE_TRIANGLE_RATIOS = {
    "broadleaf": 0.03,
    "v2": 0.03,
    "maple": 0.0075,
    "ginko": 0.03,
}


def raster_shape(asset_type: str) -> tuple[int, int]:
    """Dense (height, width) sampling resolution for `asset_type`."""
    return DEFAULT_RASTER_SHAPES[asset_type]


def _scale_default_face_budget(face_count: int) -> int:
    return round(face_count * _TARGET_FACE_SCALE.get())


@contextmanager
def leaf_collection_decimation(decimate_rate: float, *, face_size: float | None = None):
    """Scale the leaf face budget, set for the default's 0.97, to `decimate_rate`
    instead of decimating again. `face_size`, when given, takes precedence and asks
    for a world-space triangle edge length, as tree `face_size` does."""
    decimate_rate = float(decimate_rate)
    if not 0.0 <= decimate_rate <= 1.0:
        raise ValueError(f"decimate_rate must be in [0, 1], got {decimate_rate}")
    if face_size is not None:
        face_size = float(face_size)
        if not 0.0 < face_size < float("inf"):
            raise ValueError(f"face_size must be finite and positive, got {face_size}")
    retention = 1.0 - decimate_rate
    scale_token = _TARGET_FACE_SCALE.set(retention / DEFAULT_COLLECTION_RETENTION)
    size_token = _TARGET_FACE_SIZE.set(face_size)
    try:
        yield
    finally:
        _TARGET_FACE_SIZE.reset(size_token)
        _TARGET_FACE_SCALE.reset(scale_token)


def requested_leaf_face_size() -> float | None:
    """World-space face size requested by `leaf_collection_decimation`, if any."""
    return _TARGET_FACE_SIZE.get()


@contextmanager
def leaf_asset_scale(scale: float):
    """Set the local-to-world uniform scale that turns `face_size` into a budget."""
    scale = float(scale)
    if not 0.0 < scale < float("inf"):
        raise ValueError(f"leaf asset scale must be finite and positive, got {scale}")
    token = _ASSET_SCALE.set(scale)
    try:
        yield
    finally:
        _ASSET_SCALE.reset(token)


def _face_size_budget(positions, complete, original_face_count: int) -> int | None:
    """Face budget for world-space edge size `s`, about `2 * area / s**2` (two
    triangles per grid cell). The area is the base plane's, since displacement only
    moves where the simplifier spends the budget."""
    face_size = _TARGET_FACE_SIZE.get()
    if face_size is None:
        return None

    import numpy as np

    height, width = positions.shape[:2]
    horizontal = (positions[0, -1] - positions[0, 0]) / max(1, width - 1)
    vertical = (positions[-1, 0] - positions[0, 0]) / max(1, height - 1)
    local_cell_area = float(np.linalg.norm(np.cross(horizontal, vertical)))
    local_area = local_cell_area * int(np.count_nonzero(complete))
    world_area = local_area * _ASSET_SCALE.get() ** 2
    budget = round(2.0 * world_area / face_size**2)
    return min(original_face_count, max(1, budget))


@lru_cache(maxsize=1)
def _kernels_available() -> bool:
    """True when the kernels built by INFINIGEN_INSTALL_PERF import. Without them the
    route would fail part way through a leaf, so leaves use the default meshing."""
    try:
        from infinigen.perf.assets.utils import _native_fields  # noqa: F401
        from infinigen.perf.assets.utils.heightfield import (  # noqa: F401
            _adaptive_triangulation,
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "native leaf meshing kernels unavailable; using the default leaf "
            "path. Reinstall with INFINIGEN_INSTALL_PERF=True to enable them."
        )
        return False
    return True


def enabled() -> bool:
    """Whether leaves are built through the height-field route."""
    return _kernels_available()


def mark_final_mesh(obj) -> None:
    """Mark `obj` as built here, so the leaf collection does not decimate it."""
    if enabled():
        _FINAL_MESH_POINTERS.add(obj.as_pointer())


def is_final_mesh(obj) -> bool:
    return obj.as_pointer() in _FINAL_MESH_POINTERS


def convert_active_to_mesh(obj):
    """Convert `obj`'s evaluated geometry to a mesh, keeping the material slots that
    a Mesh Grid node's output drops."""

    materials = list(obj.data.materials) if enabled() else None
    converted = _convert_evaluated_copy(obj)
    if materials is not None:
        for material in materials:
            if material is not None and material.name not in converted.data.materials:
                converted.data.materials.append(material)
    return converted


def _convert_evaluated_copy(obj):
    """`bpy.ops.object.convert(target="MESH")` that copies the depsgraph's evaluated
    mesh when `obj` is active and the only selected editable object, instead of
    evaluating the modifiers again; other selections use the operator."""
    import bpy

    from infinigen.perf.core.util.isolation import IsolatedScene

    active = bpy.context.active_object
    selected = [o.name for o in bpy.context.selected_editable_objects]
    if active is None or active != obj or obj.type != "MESH" or selected != [obj.name]:
        bpy.ops.object.convert(target="MESH")
        return bpy.context.object
    IsolatedScene.ensure_evaluable(obj)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    new_mesh = bpy.data.meshes.new_from_object(
        obj.evaluated_get(depsgraph), depsgraph=depsgraph
    )
    old_mesh = obj.data
    name = old_mesh.name
    obj.modifiers.clear()
    obj.data = new_mesh
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    new_mesh.name = name
    return obj


def convert_raster_fields_to_mesh(
    obj,
    asset_type: str,
    fields: dict,
    *,
    simplify_features: dict[str, float] | None = None,
):
    """Replace `obj`'s mesh with one simplified from the field rasters, with
    `simplify_features` mapping raster field names to their error weights."""
    import numpy as np

    if asset_type != "broadleaf":
        raise ValueError(
            f"shader raster conversion is not implemented for {asset_type}"
        )
    mask = np.asarray(fields["mask"] >= 0.5, dtype=np.uint8)
    rendered_x = np.asarray(fields["coordinate_x"], dtype=np.float32)
    rendered_y = np.asarray(fields["coordinate_y"], dtype=np.float32)
    # Background pixels are zero. Fit the affine pixel-to-coordinate map on
    # adjacent valid samples and extend it through the background, so XY and UV
    # interpolate correctly at the outline. The extrema of the occupied pixels
    # would be wrong, since the render crop pads the asset.
    occupied = mask != 0
    horizontal_pairs = occupied[:, 1:] & occupied[:, :-1]
    vertical_pairs = occupied[1:, :] & occupied[:-1, :]
    x_step = float(
        np.median((rendered_x[:, 1:] - rendered_x[:, :-1])[horizontal_pairs])
    )
    y_step = float(np.median((rendered_y[1:, :] - rendered_y[:-1, :])[vertical_pairs]))
    rows, columns = np.indices(mask.shape, dtype=np.float32)
    x_origin = float(np.median(rendered_x[occupied] - columns[occupied] * x_step))
    y_origin = float(np.median(rendered_y[occupied] - rows[occupied] * y_step))
    x_values = x_origin + np.arange(mask.shape[1], dtype=np.float32) * x_step
    y_values = y_origin + np.arange(mask.shape[0], dtype=np.float32) * y_step
    coordinate_x = np.broadcast_to(x_values, mask.shape)
    coordinate_y = np.broadcast_to(y_values[:, None], mask.shape)
    y_shift = -float(coordinate_y[mask != 0].min())
    positions = np.empty(mask.shape + (3,), dtype=np.float32)
    positions[..., 0] = coordinate_x
    positions[..., 1] = coordinate_y + y_shift
    positions[..., 2] = fields["z"]
    coordinate = np.empty(mask.shape + (3,), dtype=np.float32)
    coordinate[..., 0] = coordinate_x
    coordinate[..., 1] = coordinate_y
    coordinate[..., 2] = 0.0
    attributes = {
        "offset": np.asarray(fields["offset"], dtype=np.float32),
        "coordinate": coordinate,
        "subvein offset": np.asarray(fields["subvein_offset"], dtype=np.float32),
        "vein value": np.asarray(fields["vein_value"], dtype=np.float32),
        "x modulated": np.asarray(fields["x_modulated"], dtype=np.float32),
    }
    uv = np.empty(mask.shape + (2,), dtype=np.float32)
    uv[..., 0] = 0.5 * (coordinate_x + 1.0)
    uv[..., 1] = 0.5 * (coordinate_y + 1.0)
    return _convert_geometry_simplified(
        obj,
        mask,
        positions,
        attributes,
        list(obj.data.materials),
        asset_type,
        uv_raster=uv,
        simplify_features={
            name: (fields[name], weight)
            for name, weight in (simplify_features or {}).items()
        },
    )


def _convert_geometry_simplified(
    obj,
    mask,
    positions,
    attributes,
    materials,
    asset_type: str,
    *,
    uv_raster=None,
    simplify_features: dict[str, tuple[object, float]] | None = None,
):
    """Give `obj` a mesh of the masked raster, simplified before it becomes a Blender
    mesh. UVs and `attributes` are carried along without affecting it; only
    `simplify_features` add to its error."""
    import bpy
    import numpy as np

    from infinigen.perf.assets.utils.heightfield import (
        precondition_flat_grid_regions,
        simplify_geometry,
        simplify_geometry_with_fields,
    )

    selected = mask != 0
    vertex_ids = np.full(mask.shape, -1, dtype=np.int32)
    vertex_ids[selected] = np.arange(np.count_nonzero(selected), dtype=np.int32)
    complete = (
        selected[:-1, :-1] & selected[:-1, 1:] & selected[1:, 1:] & selected[1:, :-1]
    )
    row, column = np.nonzero(complete)
    quads = np.column_stack(
        (
            vertex_ids[row, column],
            vertex_ids[row, column + 1],
            vertex_ids[row + 1, column + 1],
            vertex_ids[row + 1, column],
        )
    )
    dense_vertices = np.ascontiguousarray(positions[selected], dtype=np.float32)
    original_face_count = 2 * len(quads)
    face_size_budget = _face_size_budget(positions, complete, original_face_count)
    final_face_budget = (
        face_size_budget
        if face_size_budget is not None
        else max(
            1,
            _scale_default_face_budget(
                round(original_face_count * LEGACY_DENSE_TRIANGLE_RATIOS[asset_type])
            ),
        )
    )
    metric_scale = (1.0, 1.0, 1.0)
    dense_faces, _, _ = precondition_flat_grid_regions(
        dense_vertices,
        vertex_ids,
        complete,
        tolerance=2e-3,
        minimum_block=4,
        metric_scale=metric_scale,
    )
    retained = np.unique(dense_faces)
    retained_remap = np.full(len(dense_vertices), -1, dtype=np.int32)
    retained_remap[retained] = np.arange(len(retained), dtype=np.int32)
    dense_faces = np.ascontiguousarray(retained_remap[dense_faces], dtype=np.uint32)
    dense_vertices = np.ascontiguousarray(dense_vertices[retained], dtype=np.float32)
    feature_columns = []
    feature_weights = []
    for name, (raster, weight) in (simplify_features or {}).items():
        values = np.asarray(raster, dtype=np.float32)
        if values.shape[:2] != mask.shape:
            raise ValueError(f"simplification feature {name!r} has incompatible shape")
        selected_values = values[selected][retained]
        if selected_values.ndim == 1:
            selected_values = selected_values[:, None]
        else:
            selected_values = selected_values.reshape(len(dense_vertices), -1)
        feature_columns.append(selected_values)
        feature_weights.extend([float(weight)] * selected_values.shape[1])
    feature_array = (
        np.ascontiguousarray(np.column_stack(feature_columns), dtype=np.float32)
        if feature_columns
        else None
    )
    feature_weight_array = (
        np.asarray(feature_weights, dtype=np.float32) if feature_columns else None
    )

    if uv_raster is None:
        dense_uv = np.column_stack(np.nonzero(selected))[:, ::-1].astype(np.float32)
        dense_uv[:, 0] /= mask.shape[1] - 1
        dense_uv[:, 1] /= mask.shape[0] - 1
    else:
        dense_uv = np.asarray(uv_raster[selected], dtype=np.float32)
    dense_uv = dense_uv[retained]

    transport_columns = []
    transport_layout = {}
    for name, raster in attributes.items():
        values = np.asarray(raster[selected], dtype=np.float32)[retained]
        if values.ndim == 1:
            values = values[:, None]
        else:
            values = values.reshape(len(dense_vertices), -1)
        start = sum(column.shape[1] for column in transport_columns)
        transport_columns.append(values)
        transport_layout[name] = (start, start + values.shape[1])
    uv_start = sum(column.shape[1] for column in transport_columns)
    transport_columns.append(dense_uv)
    passive_fields = np.ascontiguousarray(
        np.column_stack(transport_columns), dtype=np.float32
    )
    common_kwargs = {
        "face_budget": final_face_budget,
        "lock_border": False,
        "target_error": 1e30,
        "regularize": "none",
        "metric_scale": metric_scale,
    }
    if feature_array is None:
        vertices, faces, source_indices, transported_fields, result_error = (
            simplify_geometry_with_fields(
                dense_vertices,
                dense_faces,
                passive_fields,
                **common_kwargs,
            )
        )
    else:
        vertices, faces, source_indices, result_error = simplify_geometry(
            dense_vertices,
            dense_faces,
            **common_kwargs,
            update_positions=False,
            features=feature_array,
            feature_weights=feature_weight_array,
        )
        transported_fields = passive_fields[source_indices]

    # Correct the fields against the raster at the transported chart points. The
    # chart is affine in the raster indices; fit that map rather than assume a
    # square canvas.
    dense_chart = dense_uv
    chart = transported_fields[:, uv_start : uv_start + 2]
    row_ids, column_ids = np.nonzero(selected)
    row_ids = row_ids[retained]
    column_ids = column_ids[retained]
    design = np.column_stack((dense_chart, np.ones(len(dense_chart))))
    mapping = np.linalg.lstsq(
        design,
        np.column_stack(
            (column_ids / (mask.shape[1] - 1), row_ids / (mask.shape[0] - 1))
        ),
        rcond=None,
    )[0]
    chart = np.column_stack((chart, np.ones(len(chart)))) @ mapping
    source_columns = []
    for name, (start, stop) in transport_layout.items():
        source_columns.append(
            np.asarray(attributes[name]).reshape(mask.shape + (stop - start,))
        )
    from infinigen.perf.assets.utils.heightfield.reconstruction import (
        correct_field_residuals,
    )

    transported_fields[:, :uv_start], _ = correct_field_residuals(
        np.concatenate(source_columns, axis=2),
        selected,
        chart,
        faces,
        transported_fields[:, :uv_start],
    )

    mesh = bpy.data.meshes.new(f"{obj.data.name}_meshopt")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for name, (start, stop) in transport_layout.items():
        values = transported_fields[:, start:stop]
        if stop - start == 1:
            attribute = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
            attribute.data.foreach_set("value", values[:, 0])
        else:
            attribute = mesh.attributes.new(
                name=name, type="FLOAT_VECTOR", domain="POINT"
            )
            attribute.data.foreach_set("vector", values[:, :3].ravel())

    coordinate = transported_fields[:, uv_start : uv_start + 2]
    uv_layer = mesh.uv_layers.new(name="UVMap")
    vertex_indices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", vertex_indices)
    uv_layer.data.foreach_set("uv", coordinate[vertex_indices].ravel())
    for material in materials:
        if material is not None:
            mesh.materials.append(material)

    old_mesh = obj.data
    obj.data = mesh
    for modifier in list(obj.modifiers):
        obj.modifiers.remove(modifier)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    if "sharp_face" not in mesh.attributes:
        mesh.attributes.new(name="sharp_face", type="BOOLEAN", domain="FACE")
    return obj
