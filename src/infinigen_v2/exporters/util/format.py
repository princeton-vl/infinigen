import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np

"""
Defines infinigen's conventions for what GT types exist, regardless of the exporter/renderer used.
"""


class ExportType(enum.Enum):
    CAMERA = "camera"
    IMAGE = "rgb"
    IMAGE_DENOISED = "rgb-denoised"
    IMAGE_HDR = "rgb-hdr"
    IMAGE_DENOISED_HDR = "rgb-denoised-hdr"

    BLENDER_FILE = "scene"
    CAM_IMU_TUM_TRAJ = "cam-trajectory-imu-tum"
    OBJ_IMU_TUM_TRAJ = "obj-trajectory-imu-tum"

    MESH = "mesh"

    OBJECTS_FILE = "objects"

    OPTICAL_FLOW = "optical-flow"
    DEPTH_CHANGE = "depth-change"
    FLOW_3D = "flow-3d"
    FLOW_MASK_MATCHED = "flow-mask-matched"
    FLOW_MASK_CYCLECONSISTENCY = "flow-mask-cycle-consistency"
    INV_FLOW = "inv-flow-3d"
    POINT_TRAJECTORIES = "point-trajectories-3d"
    INV_POINT_TRAJECTORIES = "inv-point-trajectories-3d"

    DEPTH = "depth"
    SURFACE_NORMAL = "surface-normal"
    SURFACE_NORMAL_WORLD = "surface-normal-world"
    OCCLUSION_BOUNDARIES = "occlusion"

    OBJECT_INDEX = "semantic-segmentation"
    INSTANCE_SEGMENTATION = "instance-segmentation"
    MATERIAL_INDEX = "material-segmentation"
    OBJECT_INDEX_TABLE = "object-index-table"
    MATERIAL_INDEX_TABLE = "material-index-table"
    ATTRIBUTE_MASK = "attribute-mask"

    VISUALIZATIONS = "visualizations"

    DIFFUSE_COLOR = "diffuse-color"

    AMBIENT_OCCLUSION = "ambient-occlusion"

    # lighting splits
    DIFFUSE_DIRECT = "diffuse-direct"
    DIFFUSE_INDIRECT = "diffuse-indirect"
    GLOSSY_DIRECT = "glossy-direct"
    GLOSSY_INDIRECT = "glossy-indirect"
    TRANSMISSION_DIRECT = "transmission-direct"
    TRANSMISSION_INDIRECT = "transmission-indirect"
    VOLUME_DIRECT = "volume-direct"
    EMISSION = "emission"
    ENVIRONMENT = "environment"


@dataclass
class RenderPass:
    type: ExportType
    path: Path
    data_type: np.dtype

    def __post_init__(self):
        self.path = Path(self.path)


class Compression(enum.Enum):
    NONE = "none"
    TARBALL = "tarball"
    VIDEO = "video"


NON_FRAME_TYPES = {ExportType.CAMERA, ExportType.MESH, ExportType.CAM_IMU_TUM_TRAJ}
NON_CAMERA_TYPES = {
    ExportType.SURFACE_NORMAL_WORLD,
    ExportType.MESH,
    ExportType.CAM_IMU_TUM_TRAJ,
}


def ensure_path_placeholders(render_pass: RenderPass) -> RenderPass:
    """Add %f and %c placeholders to path if missing."""
    path = render_pass.path
    s = str(path)
    if render_pass.type not in NON_FRAME_TYPES and "%f" not in s:
        path = path.with_name(path.stem + "_%f" + path.suffix)
        s = str(path)
    if render_pass.type not in NON_CAMERA_TYPES and "%c" not in s:
        path = Path("%c") / path
    return RenderPass(render_pass.type, path, render_pass.data_type)


EEVEE_UNSUPPORTED_TYPES = {
    ExportType.CAMERA,
    ExportType.DIFFUSE_INDIRECT,
    ExportType.GLOSSY_INDIRECT,
    ExportType.TRANSMISSION_INDIRECT,
    ExportType.MESH,
    ExportType.CAM_IMU_TUM_TRAJ,
}


MAINRENDER_PASS_DEFAULTS: dict[ExportType, RenderPass] = {
    ExportType.IMAGE: RenderPass(
        ExportType.IMAGE, Path("%c/%f.png"), np.dtype(np.uint8)
    ),
    ExportType.IMAGE_DENOISED: RenderPass(
        ExportType.IMAGE_DENOISED, Path("%c/image-denoised_%f.png"), np.dtype(np.uint8)
    ),
    ExportType.MATERIAL_INDEX: RenderPass(
        ExportType.MATERIAL_INDEX, Path("%c/material-index_%f.npy"), np.dtype(np.uint32)
    ),
    ExportType.DIFFUSE_COLOR: RenderPass(
        ExportType.DIFFUSE_COLOR, Path("%c/diffuse-color_%f.png"), np.dtype(np.uint8)
    ),
    ExportType.EMISSION: RenderPass(
        ExportType.EMISSION, Path("%c/emission_%f.png"), np.dtype(np.uint8)
    ),
    ExportType.ENVIRONMENT: RenderPass(
        ExportType.ENVIRONMENT, Path("%c/environment_%f.png"), np.dtype(np.uint8)
    ),
    ExportType.GLOSSY_DIRECT: RenderPass(
        ExportType.GLOSSY_DIRECT, Path("%c/glossy-direct_%f.png"), np.dtype(np.uint8)
    ),
    ExportType.GLOSSY_INDIRECT: RenderPass(
        ExportType.GLOSSY_INDIRECT,
        Path("%c/glossy-indirect_%f.png"),
        np.dtype(np.uint8),
    ),
    ExportType.TRANSMISSION_DIRECT: RenderPass(
        ExportType.TRANSMISSION_DIRECT,
        Path("%c/transmission-direct_%f.png"),
        np.dtype(np.uint8),
    ),
    ExportType.TRANSMISSION_INDIRECT: RenderPass(
        ExportType.TRANSMISSION_INDIRECT,
        Path("%c/transmission-indirect_%f.png"),
        np.dtype(np.uint8),
    ),
    ExportType.AMBIENT_OCCLUSION: RenderPass(
        ExportType.AMBIENT_OCCLUSION,
        Path("%c/ambient-occlusion_%f.png"),
        np.dtype(np.uint8),
    ),
    ExportType.DIFFUSE_DIRECT: RenderPass(
        ExportType.DIFFUSE_DIRECT, Path("%c/diffuse-direct_%f.png"), np.dtype(np.uint8)
    ),
    ExportType.DIFFUSE_INDIRECT: RenderPass(
        ExportType.DIFFUSE_INDIRECT,
        Path("%c/diffuse-indirect_%f.png"),
        np.dtype(np.uint8),
    ),
}

GT_PASS_DEFAULTS: dict[ExportType, RenderPass] = {
    ExportType.DEPTH: RenderPass(
        ExportType.DEPTH, Path("%c/depth_%f.npy"), np.dtype(np.float32)
    ),
    ExportType.SURFACE_NORMAL: RenderPass(
        ExportType.SURFACE_NORMAL,
        Path("%c/surface-normal_%f.npy"),
        np.dtype(np.float32),
    ),
    ExportType.OPTICAL_FLOW: RenderPass(
        ExportType.OPTICAL_FLOW, Path("%c/optical-flow_%f.npy"), np.dtype(np.float32)
    ),
    ExportType.OBJECT_INDEX: RenderPass(
        ExportType.OBJECT_INDEX, Path("%c/object_%f.npy"), np.dtype(np.uint32)
    ),
    ExportType.SURFACE_NORMAL_WORLD: RenderPass(
        ExportType.SURFACE_NORMAL_WORLD,
        Path("%c/surface-normal-world_%f.npy"),
        np.dtype(np.float32),
    ),
}

SCENE_PASS_DEFAULTS: dict[ExportType, RenderPass] = {
    ExportType.CAM_IMU_TUM_TRAJ: RenderPass(
        ExportType.CAM_IMU_TUM_TRAJ, Path("cam-imu-tum/"), np.dtype(np.float32)
    ),
    ExportType.OBJ_IMU_TUM_TRAJ: RenderPass(
        ExportType.OBJ_IMU_TUM_TRAJ, Path("obj-imu-tum/"), np.dtype(np.float32)
    ),
    ExportType.CAMERA: RenderPass(
        ExportType.CAMERA, Path("%c/camera_%f.npz"), np.dtype(np.float32)
    ),
}
