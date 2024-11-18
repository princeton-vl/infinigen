# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import argparse
import logging
import os

import OpenEXR

# ruff: noqa: E402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.

import colorsys
from pathlib import Path

import cv2
import numpy as np
from imageio import imwrite
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def load_exr(path):
    assert Path(path).exists() and Path(path).suffix == ".exr", path
    return cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


load_flow = load_exr


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


def load_depth(p):
    return load_single_channel(p)

def load_normals(p):
    return load_exr(p)[..., [2, 0, 1]] * np.array([-1.0, 1.0, 1.0])


def load_seg_mask(p):
    return load_single_channel(p).astype(np.int64)


def load_uniq_inst(p):
    return load_exr(p).view(np.int32)


def colorize_flow(optical_flow):
    try:
        import flow_vis
    except ImportError:
        logger.warning(
            "Flow visualization requires the 'flow_vis' package. Please install via `pip install .[vis]."
        )
        return None

    flow_uv = optical_flow[..., :2]
    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_color


def colorize_normals(surface_normals):
    assert surface_normals.max() < 1 + 1e-4
    assert surface_normals.min() > -1 - 1e-4
    norm = np.linalg.norm(surface_normals, axis=2)
    color = np.round((surface_normals + 1) * (255 / 2)).astype(np.uint8)
    color[norm < 1e-4] = 0
    return color


def colorize_depth(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[..., :3] * 255, dtype=np.uint8)


def colorize_int_array(data, color_seed=0):
    H, W, *_ = data.shape
    data = data.reshape((H * W, -1))
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
    return unique_colors[indices].reshape((H, W, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_path", type=Path, default=None)
    parser.add_argument("--depth_path", type=Path, default=None)
    parser.add_argument("--seg_path", type=Path, default=None)
    parser.add_argument("--uniq_inst_path", type=Path, default=None)
    parser.add_argument("--normals_path", type=Path, default=None)
    args = parser.parse_args()

    if args.flow_path is not None:
        try:
            flow_color = colorize_flow(load_flow(args.flow_path))
            if flow_color is not None:
                output_path = args.flow_path.with_suffix(".png")
                imwrite(output_path, flow_color)
                print(f"Wrote {output_path}")
        except ModuleNotFoundError:
            print(
                "Flow visualization requires the 'flow_vis' package. Install it with 'pip install flow_vis'"
            )
            pass

    if args.normals_path is not None:
        normal_color = colorize_normals(load_normals(args.normals_path))
        output_path = args.normals_path.with_suffix(".png")
        imwrite(output_path, normal_color)
        print(f"Wrote {output_path}")

    if args.depth_path is not None:
        depth_color = colorize_depth(load_depth(args.depth_path))
        output_path = args.depth_path.with_suffix(".png")
        imwrite(output_path, depth_color)
        print(f"Wrote {output_path}")

    if args.uniq_inst_path is not None:
        mask_color = colorize_int_array(load_uniq_inst(args.uniq_inst_path))
        output_path = args.uniq_inst_path.with_suffix(".png")
        imwrite(output_path, mask_color)
        print(f"Wrote {output_path}")

    if args.seg_path is not None:
        mask_color = colorize_int_array(load_seg_mask(args.seg_path))
        output_path = args.seg_path.with_suffix(".png")
        imwrite(output_path, mask_color)
        print(f"Wrote {output_path}")
