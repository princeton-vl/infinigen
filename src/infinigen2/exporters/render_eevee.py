# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan: Eevee rendering implementation

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Unpack

import bpy
import procfunc as pf
from procfunc.util.log import Suppress

from infinigen2.exporters.util.blender_render import (
    DisplacementMode,
    configure_compositor_viewlayer_output,
    configure_material_index_table,
    configure_object_index_table,
    override_shading_for_gt,
    postprocess_renderpass_paths,
)
from infinigen2.exporters.util.eevee_aov import (
    configure_materials_for_aovs,
    setup_eevee_aovs,
)
from infinigen2.exporters.util.format import (
    EEVEE_UNSUPPORTED_TYPES,
    ExportType,
    RenderPass,
    ensure_path_placeholders,
)
from infinigen2.util.camera_projection import adjust_camera_sensor

__all__ = [
    "configure_eevee_params",
    "render_eevee",
    "render_eevee_ground_truth",
]

logger = logging.getLogger(__name__)


@dataclass
class _RenderEeveeParams:
    taa_render_samples: int = 64
    use_taa_reprojection: bool = True
    film_exposure: float = 1.0
    taa_samples: int = 16
    use_gtao: bool = False
    gtao_distance: float = 0.2
    gtao_quality: float = 0.25
    use_shadows: bool = True
    shadow_pool_size: int = "512"
    shadow_ray_count: int = 1
    shadow_resolution_scale: float = 1.0
    shadow_step_count: int = 6
    use_raytracing: bool = False
    ray_tracing_method: str = "SCREEN"  # PROBE or SCREEN
    use_fast_gi: bool = False
    fast_gi_method: str = "GLOBAL_ILLUMINATION"  # or AMBIENT_OCCLUSION_ONLY
    fast_gi_ray_count: int = 2
    fast_gi_resolution: str = "2"
    fast_gi_step_count: int = 8
    fast_gi_quality: float = 0.25
    fast_gi_bias: float = 0.05
    fast_gi_distance: float = 0.0
    fast_gi_thickness_far: float = 0.785398
    fast_gi_thickness_near: float = 0.25
    use_overscan: bool = False
    overscan_size: float = 3.0
    use_volume_custom_range: bool = False
    volumetric_start: float = 0.1
    volumetric_end: float = 100.0
    volumetric_samples: int = 64
    volumetric_shadow_samples: int = 16
    volumetric_tile_size: str = "8"
    volumetric_sample_rand: float = 0.8
    volumetric_light_clamp: float = 0.0
    volumetric_ray_depth: int = 16
    use_volumetric_shadows: bool = False


def configure_eevee_params(
    params: _RenderEeveeParams,
):
    eevee = bpy.context.scene.eevee
    for k, v in params.__dict__.items():
        if hasattr(eevee, k):
            setattr(eevee, k, v)


@pf.tracer.primitive
def render_eevee(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject,
    output_folder: Path,
    render_passes: list[RenderPass],
    frame_start: int,
    frame_end: int,
    resolution: tuple[int, int] = (1280, 720),
    frame_rate: int = 24,
    view_layer: pf.ViewLayer = None,
    depth_of_field_fstop: float | None = None,
    motion_blur_shutter: float | None = None,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
    allow_gt_types: bool = False,
    **parameters: Unpack[_RenderEeveeParams],
) -> dict[ExportType, list[Path]]:
    """Render using Eevee with multiple passes"""

    render_passes = [ensure_path_placeholders(rp) for rp in render_passes]
    unsupported = [rp for rp in render_passes if rp.type in EEVEE_UNSUPPORTED_TYPES]
    if unsupported:
        raise ValueError(f"Eevee exporter does not support {unsupported=}")

    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.fps = frame_rate
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.render.use_persistent_data = frame_end > frame_start
    bpy.context.scene.camera = camera.item()

    adjust_camera_sensor(camera)
    configure_eevee_params(_RenderEeveeParams(**parameters))

    use_dof = depth_of_field_fstop is not None
    camera.item().data.dof.use_dof = use_dof
    if use_dof:
        camera.item().data.dof.aperture_fstop = depth_of_field_fstop

    if motion_blur_shutter is not None:
        bpy.context.scene.render.motion_blur_shutter = motion_blur_shutter

    if displacement_mode not in [
        DisplacementMode.NONE,
        DisplacementMode.REALIZE_MESH,
    ]:
        for material in bpy.data.materials:
            material.displacement_method = displacement_mode.value

    bpy.context.scene.render.use_persistent_data = frame_end > frame_start

    if not render_passes:
        return {}

    result = {}
    pass_types = {rp.type for rp in render_passes}

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    tmp_folder = output_folder / "tmp_"
    tmp_folder.mkdir(exist_ok=True, parents=True)
    bpy.context.scene.render.filepath = str(tmp_folder) + "/"

    camera_name = camera.item().name
    camera_folder = output_folder / camera_name
    camera_folder.mkdir(exist_ok=True, parents=True)

    if ExportType.OBJECT_INDEX in pass_types:
        table = configure_object_index_table()
        object_index_path = camera_folder / "object-index-table.json"
        with object_index_path.open("w") as f:
            json.dump(table, f, indent=4)
        result[ExportType.OBJECT_INDEX_TABLE] = [object_index_path]

    if ExportType.MATERIAL_INDEX in pass_types:
        table = configure_material_index_table()
        material_index_path = camera_folder / "material-index-table.json"
        with material_index_path.open("w") as f:
            json.dump(table, f, indent=4)
        result[ExportType.MATERIAL_INDEX_TABLE] = [material_index_path]

    if view_layer is None:
        view_layer = bpy.context.scene.view_layers["ViewLayer"]

    # Setup AOVs for passes that need them in Eevee
    aov_mapping = setup_eevee_aovs(view_layer, render_passes)
    if aov_mapping:
        configure_materials_for_aovs(aov_mapping)

    blender_result_paths = configure_compositor_viewlayer_output(
        render_passes, tmp_folder, camera_name, view_layer, aov_mapping
    )

    # Cancel the render only after configuring it, since render with no passes is an intended way to cause config but no render
    # TODO fix by adding a separate function for configuration only
    if len(render_passes) == 0:
        return {}

    context = Suppress() if logger.getEffectiveLevel() > logging.INFO else nullcontext()
    with context:
        bpy.ops.render.render(animation=True)

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    for pass_config in render_passes:
        paths = postprocess_renderpass_paths(
            from_template=Path(blender_result_paths[pass_config.type]),
            to_template=(output_folder / pass_config.path),
            frame_start=frame_start,
            frame_end=frame_end,
            pass_config=pass_config,
            camera=camera,
        )
        result[pass_config.type] = paths

    return result


@pf.tracer.primitive
def render_eevee_ground_truth(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject,
    output_folder: Path,
    render_passes: list[RenderPass],
    frame_start: int,
    frame_end: int,
    resolution: tuple[int, int] = (1280, 720),
    frame_rate: int = 24,
    view_layer: pf.ViewLayer = None,
    depth_of_field_fstop: float | None = None,
    motion_blur_shutter: float | None = None,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
    **parameters: Unpack[_RenderEeveeParams],
) -> dict[ExportType, list[Path]]:
    with override_shading_for_gt(objects):
        return render_eevee(
            objects=objects,
            camera=camera,
            output_folder=output_folder,
            render_passes=render_passes,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
            frame_rate=frame_rate,
            view_layer=view_layer,
            depth_of_field_fstop=depth_of_field_fstop,
            motion_blur_shutter=motion_blur_shutter,
            displacement_mode=displacement_mode,
            allow_gt_types=True,
            **parameters,
        )
