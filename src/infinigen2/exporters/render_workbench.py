# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson, Hei Law, Alexander Raistrick: original Infinigen v1 rendering pipeline (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/core/rendering/render.py)
# - Alexander Raistrick: port to v2

import logging
from contextlib import nullcontext
from pathlib import Path

import bpy
import procfunc as pf
from procfunc.util.log import Suppress

from infinigen2.exporters.camera_pose import save_camera_poses
from infinigen2.exporters.util.blender_render import DisplacementMode
from infinigen2.exporters.util.format import (
    ExportType,
    RenderPass,
    ensure_path_placeholders,
)
from infinigen2.util.camera_projection import adjust_camera_sensor

__all__ = [
    "RENDER_WORKBENCH_PASS_TYPES",
    "render_workbench",
]

logger = logging.getLogger(__name__)

RENDER_WORKBENCH_PASS_TYPES = frozenset(
    {
        ExportType.CAMERA,
        ExportType.IMAGE,
    }
)


@pf.tracer.primitive
def render_workbench(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject,
    output_folder: Path,
    render_passes: list[RenderPass],
    frame_start: int = 1,
    frame_end: int = 1,
    resolution: tuple[int, int] = (1280, 720),
    frame_rate: int = 24,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
    render_skip_existing: bool = False,
    lighting: str = "MATCAP",
    color_type: str = "RANDOM",
    studio_light: str = "studio.sl",
    matcap_light: str = "basic_1.exr",
    samples: int = 16,
) -> dict[ExportType, list[Path]]:
    render_passes = [ensure_path_placeholders(rp) for rp in render_passes]

    unsupported = [
        rp for rp in render_passes if rp.type not in RENDER_WORKBENCH_PASS_TYPES
    ]
    if unsupported:
        logger.warning(
            f"render_workbench ignoring unsupported passes: {[rp.type for rp in unsupported]}"
        )
        render_passes = [
            rp for rp in render_passes if rp.type in RENDER_WORKBENCH_PASS_TYPES
        ]

    camera_pass = next(
        (rp for rp in render_passes if rp.type == ExportType.CAMERA), None
    )
    render_passes = [rp for rp in render_passes if rp.type != ExportType.CAMERA]

    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.render.use_overwrite = not render_skip_existing
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.fps = frame_rate
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.camera = camera.item()

    adjust_camera_sensor(camera)

    # Workbench shading settings
    shading = bpy.context.scene.display.shading
    shading.light = lighting
    shading.color_type = color_type
    if lighting == "STUDIO":
        shading.studio_light = studio_light
    elif lighting == "MATCAP":
        shading.studio_light = matcap_light
    bpy.context.scene.display.render_aa = "16" if samples >= 16 else "8"

    if displacement_mode not in [
        DisplacementMode.NONE,
        DisplacementMode.REALIZE_MESH,
    ]:
        for material in bpy.data.materials:
            material.displacement_method = displacement_mode.value

    result = {}
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if camera_pass is not None:
        result.update(
            save_camera_poses(
                camera=camera,
                output_folder=output_folder,
                frame_start=frame_start,
                frame_end=frame_end,
                path_template=camera_pass.path,
            )
        )

    camera_name = camera.item().name
    camera_folder = output_folder / camera_name
    camera_folder.mkdir(exist_ok=True, parents=True)

    # Set render output directly to the final IMAGE path template
    image_pass = next((rp for rp in render_passes if rp.type == ExportType.IMAGE), None)
    if image_pass is not None:
        # Resolve the template so Blender writes directly to the final path
        resolved_dir = output_folder / str(image_pass.path.parent).replace(
            "%c", camera_name
        )
        resolved_dir.mkdir(parents=True, exist_ok=True)
        filename_prefix = image_pass.path.stem.replace("%f", "")
        filepath = (
            str(resolved_dir / filename_prefix)
            if filename_prefix
            else str(resolved_dir) + "/"
        )
        bpy.context.scene.render.filepath = filepath
        bpy.context.scene.render.image_settings.color_mode = "RGB"
        bpy.context.scene.render.image_settings.file_format = "PNG"

    if not render_passes:
        return result

    context = Suppress() if logger.getEffectiveLevel() > logging.INFO else nullcontext()
    with context:
        bpy.ops.render.render(animation=True)

    # Blender writes files like <prefix>0001.png — collect them
    for pass_config in render_passes:
        paths = []
        for i in range(frame_start, frame_end + 1):
            to_path = output_folder / str(pass_config.path).replace(
                "%c", camera_name
            ).replace("%f", f"{i:04d}")
            if not to_path.exists():
                raise FileNotFoundError(f"Expected rendered file {to_path}")
            paths.append(to_path)
        result[pass_config.type] = paths

    return result
