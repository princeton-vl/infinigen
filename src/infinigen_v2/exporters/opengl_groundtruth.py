from dataclasses import dataclass
from typing import Unpack

import bpy

from infinigen_v2.exporters.format import ExportType


@dataclass
class OpenGLGroundTruthParams:
    render_passes: list[ExportType]
    resolution: tuple[int, int]


def default_opengl_groundtruth(
    resolution: tuple[int, int],
) -> OpenGLGroundTruthParams:
    render_passes = [
        format.ExportType.DEPTH,
        format.ExportType.SURFACE_NORMAL,
        format.ExportType.INSTANCE_SEGMENTATION,
        format.ExportType.FLOW_3D,
        format.ExportType.FLOW_MASK_MATCHED,
        format.ExportType.OCCLUSION_BOUNDARIES,
    ]

    return OpenGLGroundTruthParams(
        render_passes=render_passes,
        resolution=resolution,
    )


def render_opengl_groundtruth(
    objects: bpy.types.Collection,
    camera: bpy.types.Camera,
    **params: Unpack[OpenGLGroundTruthParams],
):
    pass
