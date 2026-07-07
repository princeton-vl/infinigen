# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from suffixes import get_suffix, parse_suffix


def get_mask(depth, flow, dst_depth):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    target = np.stack((y, x, depth), -1)
    target[:, :, 0] += flow[:, :, 1] * 2
    target[:, :, 1] += flow[:, :, 0] * 2
    target[:, :, 2] += flow[:, :, 2]

    interpolator = RegularGridInterpolator(
        (np.arange(H), np.arange(W)),
        dst_depth,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    mask = np.zeros((H, W), dtype=bool)
    interpolated_values = interpolator(target[:, :, :2].reshape((-1, 2)))
    mask |= (target[:, :, 2] >= 0) & (
        target[:, :, 2] <= interpolated_values.reshape((H, W))
    )
    target[:, :, 0] += 1
    interpolated_values = interpolator(target[:, :, :2].reshape((-1, 2)))
    mask |= (target[:, :, 2] >= 0) & (
        target[:, :, 2] <= interpolated_values.reshape((H, W))
    )
    target[:, :, 0] -= 2
    interpolated_values = interpolator(target[:, :, :2].reshape((-1, 2)))
    mask |= (target[:, :, 2] >= 0) & (
        target[:, :, 2] <= interpolated_values.reshape((H, W))
    )
    target[:, :, 0] += 1

    target[:, :, 1] += 1
    interpolated_values = interpolator(target[:, :, :2].reshape((-1, 2)))
    mask |= (target[:, :, 2] >= 0) & (
        target[:, :, 2] <= interpolated_values.reshape((H, W))
    )
    target[:, :, 1] -= 2
    interpolated_values = interpolator(target[:, :, :2].reshape((-1, 2)))
    mask |= (target[:, :, 2] >= 0) & (
        target[:, :, 2] <= interpolated_values.reshape((H, W))
    )

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_frames_dir", type=Path)
    parser.add_argument("point_traj_source_frame", type=int)
    args = parser.parse_args()
    assert args.target_frames_dir.exists()
    assert args.target_frames_dir.name.startswith("frames_")

    for file_path in args.target_frames_dir.glob("*.npy"):
        info = parse_suffix(file_path.name)
        data_type = file_path.name.split("_")[0]
        if not file_path.name.endswith(".npy"):
            continue
        if data_type == "Flow3D":
            depth_info = dict(info)
            depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
            depth_info["frame"] += 1
            dst_depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
        elif data_type == "PointTraj3D":
            depth_info = dict(info)
            depth_info["frame"] = args.point_traj_source_frame
            depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
            depth_info["frame"] = info["frame"]
            dst_depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
        else:
            continue
        mask = get_mask(depth, np.load(file_path), dst_depth)
        np.save(
            file_path.parent / (data_type + "Mask" + file_path.name[len(data_type) :]),
            mask,
        )
        cv2.imwrite(
            str(
                file_path.parent
                / (data_type + "Mask" + file_path.name[len(data_type) : -4] + ".png")
            ),
            mask.astype(np.uint8) * 255,
        )
