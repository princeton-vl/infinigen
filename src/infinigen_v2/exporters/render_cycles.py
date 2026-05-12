import os

# ruff: noqa: E402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import logging
import shutil
from pathlib import Path

import bpy
import procfunc as pf

from infinigen_v2.exporters.camera_pose import save_camera_poses
from infinigen_v2.exporters.util.blender_render import (
    DisplacementMode,
    configure_compositor_viewlayer_output,
    configure_material_index_table,
    configure_object_index_table,
    override_shading_for_gt,
    postprocess_renderpass_paths,
)
from infinigen_v2.exporters.util.format import (
    ExportType,
    RenderPass,
    ensure_path_placeholders,
)
from infinigen_v2.exporters.util.render_error_check import (
    check_scene_shader_complexity,
    detect_cycles_errors,
)
from infinigen_v2.util.camera_projection import (
    adjust_camera_sensor,
)

logger = logging.getLogger(__name__)

RENDER_CYCLES_PASS_TYPES = frozenset(
    {
        ExportType.CAMERA,
        ExportType.IMAGE,
        ExportType.IMAGE_DENOISED,
        ExportType.IMAGE_HDR,
        ExportType.IMAGE_DENOISED_HDR,
        ExportType.MATERIAL_INDEX,
        ExportType.DIFFUSE_COLOR,
        ExportType.EMISSION,
        ExportType.ENVIRONMENT,
        ExportType.GLOSSY_DIRECT,
        ExportType.GLOSSY_INDIRECT,
        ExportType.TRANSMISSION_DIRECT,
        ExportType.TRANSMISSION_INDIRECT,
        ExportType.AMBIENT_OCCLUSION,
        ExportType.DIFFUSE_DIRECT,
        ExportType.DIFFUSE_INDIRECT,
    }
)

RENDER_CYCLES_GT_PASS_TYPES = frozenset(
    {
        ExportType.DEPTH,
        ExportType.SURFACE_NORMAL,
        ExportType.SURFACE_NORMAL_WORLD,
        ExportType.OPTICAL_FLOW,
        ExportType.OBJECT_INDEX,
        ExportType.OCCLUSION_BOUNDARIES,
    }
)

CYCLES_GPUTYPES_PREFERENCE = [
    # key must be a valid cycles device_type
    # ordering indicate preference - earlier device types will be used over later if both are available
    #  - e.g most OPTIX gpus will also show up as a CUDA gpu, but we will prefer to use OPTIX due to this list's ordering
    "OPTIX",
    "CUDA",
    "METAL",  # untested
    "HIP",  # untested
    "ONEAPI",  # untested
    "CPU",
]


def configure_cycles_devices(
    device_type: str = "BEST_AVAILABLE",
):
    if device_type == "CPU":
        logger.info(f"Job will use CPU-only due to {device_type=}")
        bpy.context.scene.cycles.device = "CPU"
        return

    assert bpy.context.scene.render.engine == "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences

    # Necessary to "remind" cycles that the devices exist? Not sure. Without this no devices are found.
    for dt in prefs.get_device_types(bpy.context):
        prefs.get_devices_for_type(dt[0])

    assert len(prefs.devices) != 0, prefs.devices

    types = list(d.type for d in prefs.devices)

    types = sorted(types, key=CYCLES_GPUTYPES_PREFERENCE.index)
    logger.info(f"Available devices have {types=}")

    if device_type == "BEST_AVAILABLE":
        use_device_type = types[0]
    else:
        if device_type not in types:
            raise ValueError(
                f"User specifically requested {device_type=} but it was not found in {types=}"
            )
        use_device_type = device_type

    if use_device_type == "CPU":
        logger.warning(f"Job will use CPU-only, only found {types=}")
        bpy.context.scene.cycles.device = "CPU"
        return

    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = use_device_type
    use_devices = [d for d in prefs.devices if d.type == use_device_type]

    n_devices = len(use_devices)
    logger.info(f"Cycles will use {use_device_type=}, {n_devices=}")

    for d in prefs.devices:
        d.use = False
    for d in use_devices:
        d.use = True

    return use_devices


def configure_cycles_performance(
    max_samples: int,
    min_samples: int,
    samples_adaptive_threshold: float,
    film_exposure: float,
    volume_step_rate: float,
    volume_preview_step_rate: float,
    volume_max_steps: int,
    volume_bounces: int,
    max_bounces: int,
    diffuse_bounces: int,
    glossy_bounces: int,
    sample_clamp_indirect: float,
):
    cycles = bpy.context.scene.cycles
    cycles.samples = max_samples
    cycles.adaptive_min_samples = min_samples
    cycles.adaptive_threshold = samples_adaptive_threshold
    cycles.film_exposure = film_exposure
    cycles.volume_step_rate = volume_step_rate
    cycles.volume_preview_step_rate = volume_preview_step_rate
    cycles.volume_max_steps = volume_max_steps
    cycles.volume_bounces = volume_bounces
    cycles.max_bounces = max_bounces
    cycles.diffuse_bounces = diffuse_bounces
    cycles.glossy_bounces = glossy_bounces
    cycles.sample_clamp_indirect = sample_clamp_indirect

    system = bpy.context.preferences.system
    system.max_shader_compilation_subprocesses = pf.context.globals.num_cpu_cores


def _render_cycles_impl(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject,
    output_folder: Path,
    render_passes: list[RenderPass],
    frame_start: int = 1,
    frame_end: int = 1,
    resolution: tuple[int, int] = (1280, 720),
    frame_rate: int = 24,
    device_type: str = "BEST_AVAILABLE",
    view_layer: pf.ViewLayer = None,
    depth_of_field_fstop: float | None = None,
    motion_blur_shutter: float | None = None,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
    render_skip_existing: bool = False,
    min_samples: int = 32,
    max_samples: int = 256,
    samples_adaptive_threshold: float = 0.01,
    film_exposure: float = 1.0,
    volume_step_rate: float = 0.1,
    volume_preview_step_rate: float = 0.1,
    volume_max_steps: int = 32,
    volume_bounces: int = 4,
    max_bounces: int = 5,
    diffuse_bounces: int = 4,
    glossy_bounces: int = 4,
    sample_clamp_indirect: float = 10.0,
    render_output_subdir: str | None = None,
) -> dict[ExportType, list[Path]]:
    render_passes = [ensure_path_placeholders(rp) for rp in render_passes]

    camera_pass = next(
        (rp for rp in render_passes if rp.type == ExportType.CAMERA), None
    )
    render_passes = [rp for rp in render_passes if rp.type != ExportType.CAMERA]

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.use_overwrite = not render_skip_existing
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.fps = frame_rate
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.render.use_persistent_data = frame_end > frame_start
    bpy.context.scene.camera = camera.item()

    adjust_camera_sensor(camera)

    result = {}
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
    configure_cycles_devices(device_type)

    configure_cycles_performance(
        max_samples=max_samples,
        min_samples=min_samples,
        samples_adaptive_threshold=samples_adaptive_threshold,
        film_exposure=film_exposure,
        volume_step_rate=volume_step_rate,
        volume_preview_step_rate=volume_preview_step_rate,
        volume_max_steps=volume_max_steps,
        volume_bounces=volume_bounces,
        max_bounces=max_bounces,
        diffuse_bounces=diffuse_bounces,
        glossy_bounces=glossy_bounces,
        sample_clamp_indirect=sample_clamp_indirect,
    )

    # DEPTH OF FIELD
    use_dof = depth_of_field_fstop is not None
    camera.item().data.dof.use_dof = use_dof
    if use_dof:
        camera.item().data.dof.aperture_fstop = depth_of_field_fstop

    # MOTION BLUR
    if motion_blur_shutter is not None:
        bpy.context.scene.render.motion_blur_shutter = motion_blur_shutter

    # FLAT SHADING - TODO this should be temporary and reconsidered later
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.shade_flat()

    # DISPLACEMENT METHOD
    if displacement_mode not in [
        DisplacementMode.NONE,
        DisplacementMode.REALIZE_MESH,
    ]:
        for material in bpy.data.materials:
            material.displacement_method = displacement_mode.value

    # DENOISING
    denoised_passes = {ExportType.IMAGE_DENOISED, ExportType.IMAGE_DENOISED_HDR}
    use_denoising = any(rp.type in denoised_passes for rp in render_passes)
    bpy.context.scene.cycles.use_denoising = use_denoising
    if use_denoising:
        try:
            bpy.context.scene.cycles.denoiser = "OPTIX"
        except Exception as e:
            logger.warning(f"Cannot use OPTIX denoiser {e}")
        logger.info(f"Using {bpy.context.scene.cycles.denoiser=}")
    else:
        logger.info(
            f"No denoising passes requested, will use {bpy.context.scene.cycles.use_denoising=}"
        )

    pass_types = {rp.type for rp in render_passes}
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    camera_name = camera.item().name
    camera_folder = output_folder / camera_name
    camera_folder.mkdir(exist_ok=True, parents=True)
    render_filepath_folder = (
        camera_folder / render_output_subdir
        if render_output_subdir is not None
        else camera_folder
    )
    render_filepath_folder.mkdir(exist_ok=True, parents=True)

    # Use render.filepath as a filename prefix so Blender writes the auto-save directly
    # to the final path (e.g. "rgb-denoised_" → "rgb-denoised_0001.png").
    # Priority: IMAGE_DENOISED (denoised) > IMAGE (noisy). The chosen pass is written
    # via the auto-save; the other rgb pass (if present) goes through the compositor.
    autorender_pass = next(
        (
            rp
            for rp in render_passes
            if rp.type == ExportType.IMAGE_DENOISED and use_denoising
        ),
        next((rp for rp in render_passes if rp.type == ExportType.IMAGE), None),
    )
    if autorender_pass is not None and render_output_subdir is None:
        filename_prefix = autorender_pass.path.stem.replace("%f", "")
        prefix_path = output_folder / autorender_pass.path.parent / filename_prefix
        prefix_path = Path(str(prefix_path).replace("%c", camera_name))
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        # Trailing "/" tells Blender path is a directory, not a filename prefix
        bpy.context.scene.render.filepath = str(prefix_path) + (
            "" if filename_prefix else "/"
        )
        bpy.context.scene.render.image_settings.color_mode = "RGB"
    else:
        autorender_pass = None
        bpy.context.scene.render.filepath = str(render_filepath_folder) + "/"

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

    # Enable view layer passes FIRST before configuring compositor
    if view_layer is None:
        view_layer = bpy.context.scene.view_layers["ViewLayer"]

    blender_result_paths = configure_compositor_viewlayer_output(
        render_passes,
        output_folder,
        camera_name,
        view_layer,
        autorender_pass_type=autorender_pass.type
        if autorender_pass is not None
        else None,
    )

    # Cancel the render only after configuring it, since render with no passes is an intended way to cause config but no render
    # TODO fix by adding a separate function for configuration only
    if len(render_passes) == 0:
        return result

    check_scene_shader_complexity()
    replay = logger.getEffectiveLevel() <= logging.INFO
    with detect_cycles_errors(replay=replay):
        bpy.ops.render.render(animation=True)

    if render_output_subdir is not None and render_filepath_folder.exists():
        shutil.rmtree(render_filepath_folder)

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
def render_cycles(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject,
    output_folder: Path,
    render_passes: list[RenderPass],
    frame_start: int = 1,
    frame_end: int = 1,
    resolution: tuple[int, int] = (1280, 720),
    frame_rate: int = 24,
    device_type: str = "BEST_AVAILABLE",
    view_layer: pf.ViewLayer = None,
    depth_of_field_fstop: float | None = None,
    motion_blur_shutter: float | None = None,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
    render_skip_existing: bool = False,
    min_samples: int = 32,
    max_samples: int = 256,
    samples_adaptive_threshold: float = 0.01,
    film_exposure: float = 1.0,
    volume_step_rate: float = 0.1,
    volume_preview_step_rate: float = 0.1,
    volume_max_steps: int = 32,
    volume_bounces: int = 4,
    max_bounces: int = 5,
    diffuse_bounces: int = 4,
    glossy_bounces: int = 4,
    sample_clamp_indirect: float = 10.0,
) -> dict[ExportType, list[Path]]:
    unsupported = [
        rp for rp in render_passes if rp.type not in RENDER_CYCLES_PASS_TYPES
    ]
    if unsupported:
        logger.warning(
            f"render_cycles ignoring unsupported passes: {[rp.type for rp in unsupported]}"
        )
        render_passes = [
            rp for rp in render_passes if rp.type in RENDER_CYCLES_PASS_TYPES
        ]
    return _render_cycles_impl(
        objects=objects,
        camera=camera,
        output_folder=output_folder,
        render_passes=render_passes,
        frame_start=frame_start,
        frame_end=frame_end,
        resolution=resolution,
        frame_rate=frame_rate,
        device_type=device_type,
        view_layer=view_layer,
        depth_of_field_fstop=depth_of_field_fstop,
        motion_blur_shutter=motion_blur_shutter,
        displacement_mode=displacement_mode,
        render_skip_existing=render_skip_existing,
        min_samples=min_samples,
        max_samples=max_samples,
        samples_adaptive_threshold=samples_adaptive_threshold,
        film_exposure=film_exposure,
        volume_step_rate=volume_step_rate,
        volume_preview_step_rate=volume_preview_step_rate,
        volume_max_steps=volume_max_steps,
        volume_bounces=volume_bounces,
        max_bounces=max_bounces,
        diffuse_bounces=diffuse_bounces,
        glossy_bounces=glossy_bounces,
        sample_clamp_indirect=sample_clamp_indirect,
    )


@pf.tracer.primitive
def render_cycles_ground_truth(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject,
    output_folder: Path,
    render_passes: list[RenderPass],
    frame_start: int = 1,
    frame_end: int = 1,
    resolution: tuple[int, int] = (1280, 720),
    frame_rate: int = 24,
    device_type: str = "BEST_AVAILABLE",
    view_layer: pf.ViewLayer = None,
    depth_of_field_fstop: float | None = None,
    motion_blur_shutter: float | None = None,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
    render_skip_existing: bool = False,
    min_samples: int = 32,
    max_samples: int = 256,
    samples_adaptive_threshold: float = 0.01,
    film_exposure: float = 1.0,
    volume_step_rate: float = 0.1,
    volume_preview_step_rate: float = 0.1,
    volume_max_steps: int = 32,
    volume_bounces: int = 4,
    max_bounces: int = 5,
    diffuse_bounces: int = 4,
    glossy_bounces: int = 4,
    sample_clamp_indirect: float = 10.0,
) -> dict[ExportType, list[Path]]:
    unsupported = [
        rp for rp in render_passes if rp.type not in RENDER_CYCLES_GT_PASS_TYPES
    ]
    if unsupported:
        logger.warning(
            f"render_cycles_ground_truth ignoring unsupported passes: {[rp.type for rp in unsupported]}"
        )
        render_passes = [
            rp for rp in render_passes if rp.type in RENDER_CYCLES_GT_PASS_TYPES
        ]
    with override_shading_for_gt(objects):
        return _render_cycles_impl(
            objects=objects,
            camera=camera,
            output_folder=output_folder,
            render_passes=render_passes,
            render_output_subdir="tmp_gt",
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
            frame_rate=frame_rate,
            device_type=device_type,
            view_layer=view_layer,
            depth_of_field_fstop=depth_of_field_fstop,
            motion_blur_shutter=motion_blur_shutter,
            displacement_mode=displacement_mode,
            render_skip_existing=render_skip_existing,
            min_samples=min_samples,
            max_samples=max_samples,
            samples_adaptive_threshold=samples_adaptive_threshold,
            film_exposure=film_exposure,
            volume_step_rate=volume_step_rate,
            volume_preview_step_rate=volume_preview_step_rate,
            volume_max_steps=volume_max_steps,
            volume_bounces=volume_bounces,
            max_bounces=max_bounces,
            diffuse_bounces=diffuse_bounces,
            glossy_bounces=glossy_bounces,
            sample_clamp_indirect=sample_clamp_indirect,
        )


__all__ = [
    "render_cycles",
    "render_cycles_ground_truth",
    "RENDER_CYCLES_PASS_TYPES",
    "RENDER_CYCLES_GT_PASS_TYPES",
]
