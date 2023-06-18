# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson
# Date Signed: May 2 2023

import time
import warnings
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # This must be done BEFORE import cv2. 
import cv2
import colorsys
import flow_vis # run pip install flow_vis
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from imageio import imwrite

def flow_to_colorwheel(flow_path):
    assert flow_path.exists() and flow_path.suffix == ".exr"
    optical_flow = cv2.imread(str(flow_path),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    flow_uv = optical_flow[...,:2]
    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_color

def depth_to_jet(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[...,:3] * 255, dtype=np.uint8)

def mask_to_color(mask_path, color_seed=None):
    assert mask_path.exists() and mask_path.suffix == ".exr"
    exr = cv2.imread(str(mask_path),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,2].astype(np.int64)
    H,W = exr.shape
    top = exr.max()+1
    if color_seed is None:
        color_seed = int(time.time()*1000)%1000
    perm = np.random.RandomState(color_seed).permutation(top)
    output_image = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(top):
        clr = (np.asarray(colorsys.hsv_to_rgb(i / top, 0.9, 0.8))*255).astype(np.uint8)
        output_image[exr == perm[i]] = clr
    return output_image

def exr_depth_to_jet(depth_path, scale_vmin=1.0):
    assert depth_path.exists() and depth_path.suffix == ".exr"
    depth = cv2.imread(str(depth_path),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
    return depth_to_jet(depth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_path', type=Path, default=None)
    parser.add_argument('--depth_path', type=Path, default=None)
    parser.add_argument('--seg_path', type=Path, default=None)
    args = parser.parse_args()

    if args.flow_path is not None:
        flow_color = flow_to_colorwheel(args.flow_path)
        output_path = args.flow_path.with_suffix('.png')
        imwrite(output_path, flow_color)
        print(f"Wrote {output_path}")

    if args.depth_path is not None:
        depth_color = exr_depth_to_jet(args.depth_path)
        output_path = args.depth_path.with_suffix('.png')
        imwrite(output_path, depth_color)
        print(f"Wrote {output_path}")

    if args.seg_path is not None:
        mask_color = mask_to_color(args.seg_path)
        output_path = args.seg_path.with_suffix('.png')
        imwrite(output_path, mask_color)
        print(f"Wrote {output_path}")

    