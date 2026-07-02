import logging
from contextlib import contextmanager
from pathlib import Path

import bpy
import procfunc as pf

from infinigen2.exporters.render_cycles import DenoiseMode, _render_cycles_impl
from infinigen2.exporters.util.blender_render import (
    DisplacementMode,
    disconnect_output_links,
    material_output_targets,
    relink_output_links,
)
from infinigen2.exporters.util.format import ExportType, RenderPass

logger = logging.getLogger(__name__)

CLAY_EXPOSURE_SCALE = 0.3

RENDER_CYCLES_CLAY_PASS_TYPES = frozenset(
    {
        ExportType.CAMERA,
        ExportType.IMAGE,
        ExportType.IMAGE_DENOISED,
        ExportType.AMBIENT_OCCLUSION,
    }
)


@contextmanager
def override_shading_clay(objects: list[pf.MeshObject], keep_displacement: bool = True):
    """Swap every surface for a neutral white diffuse, keeping the scene's own
    lighting, then revert. keep_displacement leaves the Displacement socket wired so
    bump/displacement geometry survives; dropping it (flat clay) stops the default
    BUMP method perturbing the otherwise-flat surface."""
    targets = material_output_targets(objects)
    sockets = ["Volume", "Surface"]
    if not keep_displacement:
        sockets.append("Displacement")
    removed = disconnect_output_links(targets, sockets)
    added = []
    for nt, output in targets:
        if output is None:
            continue
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        added.append((nt, diffuse))
        nt.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    try:
        yield
    finally:
        for nt, node in added:
            nt.nodes.remove(node)
        relink_output_links(removed)


@contextmanager
def clay_fill_light(enabled: bool, camera: pf.CameraObject):
    if not enabled:
        yield
        return
    light_data = bpy.data.lights.new(name="clay_fill", type="POINT")
    light_data.energy = 700.0
    light_data.shadow_soft_size = 1.0
    light_obj = bpy.data.objects.new(name="clay_fill", object_data=light_data)
    # Parent to the camera so the headlight tracks orbit/pan motion across frames.
    light_obj.parent = camera.item()
    light_obj.location = (0.0, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(light_obj)
    try:
        yield
    finally:
        bpy.data.objects.remove(light_obj, do_unlink=True)
        bpy.data.lights.remove(light_data, do_unlink=True)


@pf.tracer.primitive
def render_cycles_clay(
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
    max_bounces: int = 4,
    diffuse_bounces: int = 4,
    glossy_bounces: int = 4,
    sample_clamp_indirect: float = 10.0,
    sample_clamp_direct: float = 10.0,
    transmission_bounces: int = 2,
    fill_light: bool = False,
    denoise_mode: DenoiseMode = DenoiseMode.BEST,
) -> dict[ExportType, list[Path]]:
    """White-clay render: override surfaces with neutral diffuse (keeping the scene
    lighting), optionally with a camera-parented fill headlight for pan views the
    scene's own lights miss. Not a registered exporter - used by render_clay_pan_video."""
    unsupported = [
        rp for rp in render_passes if rp.type not in RENDER_CYCLES_CLAY_PASS_TYPES
    ]
    if unsupported:
        logger.warning(
            f"render_cycles_clay ignoring unsupported passes: {[rp.type for rp in unsupported]}"
        )
        render_passes = [
            rp for rp in render_passes if rp.type in RENDER_CYCLES_CLAY_PASS_TYPES
        ]
    # White clay albedo is far brighter than scene-average materials, so the
    # mainrender exposure blows it out. Scale down to expose the clay well.
    film_exposure = film_exposure * CLAY_EXPOSURE_SCALE
    keep_displacement = displacement_mode != DisplacementMode.NONE
    with (
        override_shading_clay(objects, keep_displacement=keep_displacement),
        clay_fill_light(fill_light, camera),
    ):
        return _render_cycles_impl(
            objects=objects,
            camera=camera,
            output_folder=output_folder,
            render_passes=render_passes,
            render_output_subdir="tmp_clay",
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
            sample_clamp_direct=sample_clamp_direct,
            transmission_bounces=transmission_bounces,
            denoise_mode=denoise_mode,
        )
