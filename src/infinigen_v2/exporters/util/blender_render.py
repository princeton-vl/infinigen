import enum
import itertools
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
import OpenEXR
import procfunc as pf

from infinigen_v2.exporters.util.export_utils import load_exr
from infinigen_v2.exporters.util.format import ExportType, RenderPass
from infinigen_v2.util.camera_projection import get_3x4_RT_matrix_from_blender

logger = logging.getLogger(__name__)

BLENDER_FRAME_NUMBER_PLACEHOLDER = "####"


class DisplacementMode(enum.Enum):
    NONE = "NONE"
    BUMP = "BUMP"
    DISPLACEMENT = "DISPLACEMENT"
    DISPLACEMENT_AND_BUMP = "BOTH"
    REALIZE_MESH = "REALIZE_MESH"


@dataclass
class ViewLayerPass:
    pass_name: str
    socket_name: str
    result_suffix: str


BLENDER_VIEWLAYER_PASS_CONFIG = {
    # RGB
    ExportType.IMAGE: ViewLayerPass("use_pass_combined", "noisy_image", ".png"),
    ExportType.IMAGE_DENOISED: ViewLayerPass("use_pass_combined", "image", ".png"),
    ExportType.IMAGE_HDR: ViewLayerPass("use_pass_combined", "noisy_image", ".exr"),
    ExportType.IMAGE_DENOISED_HDR: ViewLayerPass("use_pass_combined", "image", ".exr"),
    # 3D
    ExportType.DEPTH: ViewLayerPass("use_pass_z", "depth", ".exr"),
    ExportType.SURFACE_NORMAL: ViewLayerPass("use_pass_normal", "normal", ".exr"),
    ExportType.SURFACE_NORMAL_WORLD: ViewLayerPass("use_pass_normal", "normal", ".exr"),
    ExportType.OPTICAL_FLOW: ViewLayerPass("use_pass_vector", "vector", ".exr"),
    # segmentation
    ExportType.OBJECT_INDEX: ViewLayerPass("use_pass_object_index", "indexOB", ".exr"),
    ExportType.MATERIAL_INDEX: ViewLayerPass(
        "use_pass_material_index", "indexMA", ".exr"
    ),
    ExportType.ATTRIBUTE_MASK: None,
    # lighting splits
    ExportType.DIFFUSE_COLOR: ViewLayerPass(
        "use_pass_diffuse_color", "diffcol", ".png"
    ),
    ExportType.DIFFUSE_DIRECT: ViewLayerPass(
        "use_pass_diffuse_direct", "diffdir", ".png"
    ),
    ExportType.DIFFUSE_INDIRECT: ViewLayerPass(
        "use_pass_diffuse_indirect", "diffind", ".png"
    ),
    ExportType.GLOSSY_DIRECT: ViewLayerPass(
        "use_pass_glossy_direct", "glossdir", ".png"
    ),
    ExportType.GLOSSY_INDIRECT: ViewLayerPass(
        "use_pass_glossy_indirect", "glossind", ".png"
    ),
    ExportType.TRANSMISSION_DIRECT: ViewLayerPass(
        "use_pass_transmission_direct", "transdir", ".png"
    ),
    ExportType.TRANSMISSION_INDIRECT: ViewLayerPass(
        "use_pass_transmission_indirect", "transind", ".png"
    ),
    ExportType.VOLUME_DIRECT: ViewLayerPass(
        "use_pass_volume_direct", "volumedir", ".png"
    ),
    ExportType.EMISSION: ViewLayerPass("use_pass_emit", "emit", ".png"),
    ExportType.ENVIRONMENT: ViewLayerPass("use_pass_environment", "env", ".png"),
    ExportType.AMBIENT_OCCLUSION: ViewLayerPass(
        "use_pass_ambient_occlusion", "ao", ".png"
    ),
}


def _resolve_viewlayer_pass(
    render_pass: RenderPass,
    view_layer: bpy.types.ViewLayer,
    aov_mapping: dict[ExportType, str] | None,
) -> tuple[ViewLayerPass, str]:
    """Enable the view layer pass attribute and return (viewlayer_pass, socket_name)."""
    if aov_mapping and render_pass.type in aov_mapping:
        socket_name = aov_mapping[render_pass.type]
        viewlayer_pass = ViewLayerPass(
            pass_name=None,
            socket_name=socket_name,
            result_suffix=BLENDER_VIEWLAYER_PASS_CONFIG.get(
                render_pass.type, ViewLayerPass(None, None, ".exr")
            ).result_suffix,
        )
        return viewlayer_pass, socket_name

    if render_pass.type not in BLENDER_VIEWLAYER_PASS_CONFIG:
        raise ValueError(
            f"{configure_compositor_viewlayer_output.__name__} does not support {render_pass.type}, "
            f"options are {list(BLENDER_VIEWLAYER_PASS_CONFIG.keys())}"
        )

    viewlayer_pass = BLENDER_VIEWLAYER_PASS_CONFIG[render_pass.type]
    pass_name = viewlayer_pass.pass_name

    if hasattr(view_layer, pass_name):
        setattr(view_layer, pass_name, True)
    elif hasattr(view_layer, "eevee") and hasattr(view_layer.eevee, pass_name):
        setattr(view_layer.eevee, pass_name, True)
    else:
        available = sorted(n for n in dir(view_layer) if n.startswith("use_pass_"))
        raise ValueError(
            f"View layer does not have pass {pass_name} for {render_pass.type=}\n"
            f"{available=}"
        )

    return viewlayer_pass, viewlayer_pass.socket_name


def _make_compositor_slot(
    render_pass: RenderPass,
    render_layers,
    viewlayer_pass: ViewLayerPass,
    socket_name: str,
    camera_name: str,
) -> tuple[str, object, str]:
    """Return (unique_slot_name, processed_output, pass_path) for a compositor File Output slot."""
    if socket_name == "noisy_image" and not (
        bpy.context.scene.render.engine == "CYCLES"
        and bpy.context.scene.cycles.use_denoising
    ):
        socket_name = "image"

    render_socket = getattr(render_layers, socket_name)

    match render_pass.type:
        case ExportType.OPTICAL_FLOW:
            separate_color = pf.nodes.func.separate_color(render_socket)
            # Based on debug: blue channel has X data, alpha has Y data
            processed_output = pf.nodes.compositor.combine_color(
                red=separate_color.blue,  # X velocity from blue
                green=separate_color.alpha,  # Y velocity from alpha
                blue=0.0,
                alpha=0.0,
            )
        case ExportType.SURFACE_NORMAL | ExportType.SURFACE_NORMAL_WORLD:
            processed_output = pf.nodes.compositor.mix_rgb(
                image_0=render_socket,
                image_1=(0.0, 0.0, 0.0, 0.0),
                blend_type="ADD",
            )
        case _:
            processed_output = render_socket

    pass_path = str(render_pass.path.with_suffix(""))
    pass_path = pass_path.replace("%f", BLENDER_FRAME_NUMBER_PLACEHOLDER)
    pass_path = pass_path.replace("%c", camera_name)

    # Ensure unique slot names to allow multiple passes using the same underlying socket
    unique_slot_name = socket_name
    if render_pass.type in {
        ExportType.SURFACE_NORMAL,
        ExportType.SURFACE_NORMAL_WORLD,
        ExportType.IMAGE,
        ExportType.IMAGE_DENOISED,
        ExportType.IMAGE_HDR,
        ExportType.IMAGE_DENOISED_HDR,
    }:
        unique_slot_name = f"{socket_name}_{render_pass.type.value}"

    return unique_slot_name, processed_output, pass_path


def configure_compositor_viewlayer_output(
    render_passes: list[RenderPass],
    frames_folder: Path,
    camera_name: str,
    view_layer: bpy.types.ViewLayer,
    aov_mapping: dict[ExportType, str] | None = None,
    autorender_pass_type: ExportType | None = None,
) -> dict[ExportType, list[Path]]:
    render_layers = pf.nodes.compositor.render_layers()

    png_file_slots = {}
    exr_file_slots = {}
    result_paths = {}

    for render_pass in render_passes:
        viewlayer_pass, socket_name = _resolve_viewlayer_pass(
            render_pass, view_layer, aov_mapping
        )

        result_path = frames_folder / render_pass.path
        result_path = Path(str(result_path).replace("%c", camera_name))
        result_path = result_path.with_suffix(viewlayer_pass.result_suffix)
        result_paths[render_pass.type] = result_path

        # This pass is written directly by Blender's auto-save via render.filepath prefix;
        # no compositor File Output slot needed.
        if render_pass.type == autorender_pass_type:
            continue

        unique_slot_name, processed_output, pass_path = _make_compositor_slot(
            render_pass, render_layers, viewlayer_pass, socket_name, camera_name
        )

        match render_pass.path.suffix:
            case ".exr" | ".npy":
                assert unique_slot_name not in exr_file_slots
                exr_file_slots[unique_slot_name] = (processed_output, pass_path)
            case ".png":
                assert unique_slot_name not in png_file_slots
                png_file_slots[unique_slot_name] = (processed_output, pass_path)
            case _:
                raise ValueError(f"Unhandled {render_pass.path.suffix=}")

    outputs = {}

    if png_file_slots:
        outputs["image"] = pf.nodes.compositor.output_file(
            base_path=str(frames_folder),
            format=dict(
                file_format="PNG",
                color_mode="RGB",
            ),
            slot_paths={k: v[1] for k, v in png_file_slots.items()},
            **{k: v[0] for k, v in png_file_slots.items()},
        )

    if exr_file_slots:
        outputs["image_exr"] = pf.nodes.compositor.output_file(
            base_path=str(frames_folder),
            format=dict(
                file_format="OPEN_EXR",
                color_mode="RGB",
            ),
            slot_paths={k: v[1] for k, v in exr_file_slots.items()},
            **{k: v[0] for k, v in exr_file_slots.items()},
        )

    if outputs:
        pf.nodes.to_compositor(outputs)
    else:
        # No compositor outputs needed — disable compositor so render.filepath auto-save fires normally.
        bpy.context.scene.use_nodes = False

    return result_paths


def configure_object_index_table():
    obj_order = ["none"]
    for i, obj in enumerate(bpy.data.objects):
        obj.pass_index = i + 1
        obj_order.append(obj.name)
    return obj_order


def configure_material_index_table():
    mat_order = ["none"]
    for i, mat in enumerate(bpy.data.materials):
        mat.pass_index = i + 1
        mat_order.append(mat.name)
    return mat_order


def load_single_channel(p):
    file = OpenEXR.InputFile(str(p))
    channel, channel_type = next(iter(file.header()["channels"].items()))
    match str(channel_type.type):
        case "FLOAT":
            np_type = np.float32
        case _:
            np_type = np.uint8
    data = np.frombuffer(file.channel(channel, channel_type.type), np_type)
    dw = file.header()["dataWindow"]
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    return data.reshape(sz)


def postprocess_renderpass_frame(
    render_pass: RenderPass,
    from_path: Path,
    to_path: Path,
    camera: pf.CameraObject | None = None,
) -> Path:
    if (
        from_path == to_path
        # currently there is no possible need to re-load a png, so we can skip it to save disk ops
        and from_path.suffix in {".png", ".exr"}
    ):
        if not from_path.exists():
            raise FileNotFoundError(f"Expected {from_path} to already exist")
        return to_path

    dt = render_pass.data_type

    logger.info(
        f"Processing {render_pass.type} from {from_path} to {to_path} with {dt=}"
    )

    if not from_path.exists():
        raise FileNotFoundError(f"From path {from_path} does not exist")
    to_path.parent.mkdir(parents=True, exist_ok=True)

    match render_pass.type, from_path.suffix, to_path.suffix, dt:
        case _, ".png", ".png", np.uint8:
            shutil.copy(from_path, to_path)

        case _, ".exr", ".exr", _:
            shutil.copy(from_path, to_path)

        case ExportType.DEPTH, ".exr", ".npy", (np.float16 | np.float32):
            data = load_single_channel(from_path)
            assert data.dtype == np.float32, data.dtype
            np.save(to_path, data.astype(dt))

        case ExportType.SURFACE_NORMAL, ".exr", ".npy", (np.float16 | np.float32):
            data = load_exr(from_path)

            assert camera is not None, "Camera is required for surface normal world"
            # Transform surface normals from world space to camera space using CV convention

            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_np = np.array(RT)
            R_world2cv = RT_np[:3, :3]

            # Reshape normals for matrix multiplication: (H, W, 3) -> (H*W, 3)
            original_shape = data.shape
            normals_flat = data.reshape(-1, 3)

            # Transform normals from world space to camera space
            # Since normals are direction vectors, only rotation is applied (no translation)
            normals_cam = (R_world2cv @ normals_flat.T).T

            norms = np.linalg.norm(normals_cam, axis=1, keepdims=True)
            valid_mask = norms > 1e-6
            normals_cam[valid_mask.flatten()] /= norms[valid_mask.flatten()]

            data = normals_cam.reshape(original_shape)

            np.save(to_path, data.astype(dt))

        case ExportType.SURFACE_NORMAL_WORLD, ".exr", ".npy", (np.float16 | np.float32):
            data = load_exr(from_path)
            norms = np.linalg.norm(data, axis=2, keepdims=True)
            valid = norms > 1e-6
            data = np.divide(data, norms, out=np.zeros_like(data), where=valid)
            np.save(to_path, data.astype(dt))

        case ExportType.OPTICAL_FLOW, ".exr", ".npy", (np.float16 | np.float32):
            data = load_exr(from_path)
            data[..., 0] *= -1
            np.save(to_path, data[..., :2].astype(dt))

        case (
            ExportType.OBJECT_INDEX
            | ExportType.MATERIAL_INDEX
            | ExportType.INSTANCE_SEGMENTATION,
            ".exr",
            ".npy",
            (np.uint8 | np.uint16 | np.uint32),
        ):
            data = load_single_channel(from_path)
            # Round before converting to integer to handle float precision issues
            data = np.round(data).astype(dt)
            np.save(to_path, data)

        case (
            ExportType.OBJECT_INDEX
            | ExportType.MATERIAL_INDEX
            | ExportType.INSTANCE_SEGMENTATION,
            ".exr",
            ".npz",
            (np.uint8 | np.uint16 | np.uint32 | np.uint64),
        ):
            raise NotImplementedError(
                "compress-masks style npz lookuptable is not yet supported"
            )

        case _:
            raise ValueError(
                f"Unhandled {render_pass=}, cycles exports {from_path.suffix} "
                f"and there is no defined conversion to {render_pass.path.suffix} {render_pass.data_type}"
            )

    if not to_path.exists():
        raise ValueError(f"Failed to produce {to_path=} for {render_pass.type}")

    if from_path != to_path and from_path.exists():
        from_path.unlink()
        logger.debug(f"Cleaned up intermediate {from_path}")

    return to_path


def _resolve_template(template: Path, camera_name: str, frame: int) -> Path:
    s = str(template)
    s = s.replace("%c", camera_name)
    s = s.replace("%f", f"{frame:04d}")
    return Path(s)


def postprocess_renderpass_paths(
    from_template: Path,
    to_template: Path,
    pass_config: RenderPass,
    camera: pf.CameraObject,
    frame_start: int,
    frame_end: int,
) -> list[Path]:
    frame_paths_new = []
    camera_name = camera.item().name

    for i in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(i)
        bpy.context.view_layer.update()
        from_path = _resolve_template(from_template, camera_name, i)
        to_path = _resolve_template(to_template, camera_name, i)

        postprocessed_path = postprocess_renderpass_frame(
            pass_config,
            from_path=from_path,
            to_path=to_path,
            camera=camera,
        )
        frame_paths_new.append(postprocessed_path)

    return frame_paths_new


@contextmanager
def override_shading_for_gt(objects: list[pf.MeshObject]):
    restore_links = []

    # Material transparency and volumes can cause issues with GT.
    # We remove them here, and restore them later.

    targets = [
        (slot.material.node_tree, slot.material.node_tree.nodes.get("Material Output"))
        for obj in objects
        for slot in obj.item().material_slots
        if slot is not None and slot.material is not None and slot.material.use_nodes
    ]

    if bpy.context.scene.world is not None and bpy.context.scene.world.use_nodes:
        world_tree = bpy.context.scene.world.node_tree
        targets += [(world_tree, world_tree.nodes.get("World Output"))]

    for target_nodetree, output in targets:
        if output is None:
            continue
        for link in list(
            itertools.chain(
                output.inputs["Volume"].links, output.inputs["Surface"].links
            )
        ):
            restore_links.append((target_nodetree, link.from_socket, link.to_socket))
            target_nodetree.links.remove(link)

    try:
        yield
    finally:
        for nt, from_socket, to_socket in restore_links:
            nt.links.new(from_socket, to_socket)
