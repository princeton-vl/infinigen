# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson: original Infinigen v1 ground-truth visualization (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/core/rendering/post_render.py)
# - Alexander Raistrick: port to v2

import colorsys
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import OpenEXR
from imageio import imwrite
from matplotlib import pyplot as plt

from infinigen2.exporters.util import flow_vis
from infinigen2.exporters.util.format import ExportType

__all__ = [
    "load_data",
    "load_single_channel",
    "mask_to_transparency_checkerboard",
    "visualize_any_frametype",
    "visualize_bw",
    "visualize_depth",
    "visualize_flow",
    "visualize_gt",
    "visualize_normals",
    "visualize_seg_mask",
    "visualize_uniq_inst",
]

logger = logging.getLogger(__name__)

# Enable OpenEXR support for OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_data(input_path: Path) -> np.ndarray:
    """Load data from either OpenEXR or NPY file."""
    input_path = str(input_path)
    if input_path.endswith(".npy"):
        return np.load(input_path)
    elif input_path.endswith(".exr"):
        return cv2.imread(input_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Must be .npy or .exr")


def load_single_channel(input_path: Path) -> np.ndarray:
    """Load single channel data from either OpenEXR or NPY file."""
    input_path = str(input_path)
    if input_path.endswith(".npy"):
        return np.load(input_path)
    elif input_path.endswith(".exr"):
        file = OpenEXR.InputFile(input_path)
        channel, channel_type = next(iter(file.header()["channels"].items()))
        data = np.frombuffer(file.channel(channel, channel_type.type), np.float32)
        dw = file.header()["dataWindow"]
        sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        return data.reshape(sz)
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Must be .npy or .exr")


def mask_to_transparency_checkerboard(
    img: np.ndarray, mask: np.ndarray, checker_size: int = 16
) -> np.ndarray:
    h, w = img.shape[:2]
    y_indices, x_indices = np.ogrid[:h, :w]
    checkerboard = ((y_indices // checker_size) + (x_indices // checker_size)) % 2
    checker_color = np.where(checkerboard[..., None], 0.8, 0.6)  # Light/dark grey

    if img.dtype == np.uint8:
        checker_color = (checker_color * 255).astype(np.uint8)

    img = img.copy()
    img = np.where(mask[..., None], checker_color, img)
    return img


def visualize_flow(
    input_path: Path,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    flow = load_data(input_path)
    flow_uv = flow[..., :2]

    if vmin is not None and vmax is not None:
        # Normalize flow to [0, 1] range
        flow_uv = (flow_uv - vmin) / (vmax - vmin)

    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)

    flow_color = mask_to_transparency_checkerboard(
        flow_color, np.isnan(flow_uv).any(axis=2)
    )

    imwrite(output_path, flow_color)


def visualize_normals(input_path: Path, output_path: Path) -> None:
    normals = load_data(input_path)[..., [2, 0, 1]] * np.array([-1.0, 1.0, 1.0])
    assert normals.max() < 1 + 1e-4
    assert normals.min() > -1 - 1e-4
    norm = np.linalg.norm(normals, axis=2)
    color = np.round((normals + 1) * (255 / 2)).astype(np.uint8)
    color = mask_to_transparency_checkerboard(color, norm < 1e-4)
    imwrite(output_path, color)


def visualize_depth(
    input_path: Path,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scale_vmin: float = 1.0,
) -> None:
    depth = load_single_channel(input_path)
    depth = 1 / depth

    depth_notnan = depth.copy()
    depth_notnan[np.isnan(depth_notnan)] = 0
    if vmin is None:
        vmin = depth.min() * scale_vmin
    if vmax is None:
        vmax = depth.max()

    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth_notnan))
    depth = mask_to_transparency_checkerboard(depth, np.isnan(depth_notnan))
    imwrite(output_path, np.ascontiguousarray(depth[..., :3] * 255, dtype=np.uint8))


def visualize_seg_mask(
    input_path: Path, output_path: Path, color_seed: int = 0
) -> None:
    mask = load_single_channel(input_path).astype(np.int64)
    H, W = mask.shape
    data = mask.reshape((H * W, -1))
    uniq, indices = np.unique(data, return_inverse=True, axis=0)
    random_states = [
        np.random.RandomState(e[:2].astype(np.uint32) + color_seed) for e in uniq
    ]
    unique_colors = (
        np.asarray(
            [
                colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)
                for s in random_states
            ]
        )
        * 255
    ).astype(np.uint8)
    imwrite(output_path, unique_colors[indices].reshape((H, W, 3)))


def visualize_uniq_inst(
    input_path: Path, output_path: Path, color_seed: int = 0
) -> None:
    uniq_inst = load_data(input_path).view(np.int32)
    H, W = uniq_inst.shape[:2]
    data = uniq_inst.reshape((H * W, -1))
    uniq, indices = np.unique(data, return_inverse=True, axis=0)
    random_states = [
        np.random.RandomState(e[:2].astype(np.uint32) + color_seed) for e in uniq
    ]
    unique_colors = (
        np.asarray(
            [
                colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)
                for s in random_states
            ]
        )
        * 255
    ).astype(np.uint8)
    imwrite(output_path, unique_colors[indices].reshape((H, W, 3)))


def visualize_bw(input_path: Path, output_path: Path) -> None:
    data = load_single_channel(input_path)
    imwrite(output_path, data)


VISUALIZATION_FUNCS: Dict[ExportType, Callable] = {
    ExportType.OPTICAL_FLOW: visualize_flow,
    ExportType.SURFACE_NORMAL: visualize_normals,
    ExportType.DEPTH: visualize_depth,
    ExportType.OBJECT_INDEX: visualize_seg_mask,
    ExportType.INSTANCE_SEGMENTATION: visualize_uniq_inst,
    ExportType.MATERIAL_INDEX: visualize_seg_mask,
    ExportType.FLOW_MASK_MATCHED: visualize_bw,
    ExportType.FLOW_MASK_CYCLECONSISTENCY: visualize_bw,
    ExportType.OCCLUSION_BOUNDARIES: visualize_bw,
}


def visualize_any_frametype(
    export_type: ExportType, frames: list[Path], output_folder: Path
) -> list[Path]:
    """Visualize a sequence of frames with consistent normalization.

    Args:
        export_type: Type of data to visualize
        frames: List of input frame paths
        output_path: Directory to save visualizations

    Returns:
        List of output paths for the visualizations
    """

    output_folder.mkdir(parents=True, exist_ok=True)

    # For depth and flow, compute global statistics
    kwargs = {}
    if export_type == ExportType.DEPTH and len(frames) > 1:
        vmin = float("inf")
        vmax = float("-inf")
        for frame in frames:
            data = load_single_channel(frame)
            inv = 1.0 / data
            valid = np.isfinite(inv) & (inv > 0)
            if valid.any():
                vmin = min(vmin, float(inv[valid].min()))
                vmax = max(vmax, float(inv[valid].max()))
        kwargs = {"vmin": vmin, "vmax": vmax}
    # OPTICAL_FLOW: no vmin/vmax — flow_to_color normalizes by max magnitude internally,
    # and pre-shifting by vmin would destroy zero-centering, making one direction appear black.

    out_paths = []
    for _i, frame in enumerate(frames):
        if frame.suffix == ".png":
            raise ValueError(f"Visualization already exists for {frame}")
        out_path = frame.with_suffix(".png")
        VISUALIZATION_FUNCS[export_type](frame, out_path, **kwargs)
        out_paths.append(out_path)

    return out_paths


def visualize_gt(
    exports: dict[ExportType, list[Path]] | list[dict[ExportType, list[Path]]],
    output_folder: Path,
) -> dict[ExportType, list[Path]]:
    if isinstance(exports, list):
        merged: dict[ExportType, list[Path]] = {}
        for d in exports:
            for k, v in d.items():
                merged.setdefault(k, []).extend(v)
        exports = merged

    all_vis_paths = []
    for export_type, frames in exports.items():
        if export_type in VISUALIZATION_FUNCS:
            all_vis_paths.extend(
                visualize_any_frametype(export_type, frames, output_folder)
            )
    return {ExportType.VISUALIZATIONS: all_vis_paths}
