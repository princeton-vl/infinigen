# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


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

def load_exr(path):
    assert Path(path).exists() and Path(path).suffix == ".exr"
    return cv2.imread(str(path),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

load_flow = load_exr
load_depth = lambda p: load_exr(p)[..., 0]
load_normals = lambda p: load_exr(p)[...,[2,0,1]] * np.array([-1.,1.,1.])
load_seg_mask = lambda p: load_exr(p)[...,2].astype(np.int64)

def colorize_flow(optical_flow):
    flow_uv = optical_flow[...,:2]
    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_color

def colorize_normals(surface_normals):
    assert surface_normals.max() < 1+1e-4
    assert surface_normals.min() > -1-1e-4
    norm = np.linalg.norm(surface_normals, axis=2)
    color = np.round((surface_normals + 1) * (255/2)).astype(np.uint8)
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
    return np.ascontiguousarray(depth[...,:3] * 255, dtype=np.uint8)

def colorize_seg_mask(exr, color_seed=None):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_path', type=Path, default=None)
    parser.add_argument('--depth_path', type=Path, default=None)
    parser.add_argument('--seg_path', type=Path, default=None)
    parser.add_argument('--normals_path', type=Path, default=None)
    args = parser.parse_args()

    if args.flow_path is not None:
        flow_color = colorize_flow(load_flow(args.flow_path))
        output_path = args.flow_path.with_suffix('.png')
        imwrite(output_path, flow_color)
        print(f"Wrote {output_path}")

    if args.normals_path is not None:
        normal_color = colorize_normals(load_normals(args.normals_path))
        output_path = args.normals_path.with_suffix('.png')
        imwrite(output_path, normal_color)
        print(f"Wrote {output_path}")

    if args.depth_path is not None:
        depth_color = colorize_depth(load_depth(args.depth_path))
        output_path = args.depth_path.with_suffix('.png')
        imwrite(output_path, depth_color)
        print(f"Wrote {output_path}")

    if args.seg_path is not None:
        mask_color = colorize_seg_mask(load_seg_mask(args.seg_path))
        output_path = args.seg_path.with_suffix('.png')
        imwrite(output_path, mask_color)
        print(f"Wrote {output_path}")

    