# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numba as nb
import numpy as np
from einops import pack, rearrange, repeat
from imageio.v3 import imread, imwrite
from numba.types import bool_

from ..compress_masks import recover
from ..dataset_loader import get_frame_path

"""
Usage: python -m tools.ground_truth.segmentation_lookup <scene-folder> <frame-index> [--query <query>] [--boxes]
Output:
- testbed
    - A.png # Original image
    - B.png # Original image + mask/2D-bounding-boxes for the provided query
"""

@nb.njit
def should_highlight_pixel(arr2d, set1d):
    """Compute boolean mask for items in arr2d that are also in set1d"""
    H, W = arr2d.shape
    output = np.zeros((H, W), dtype=bool_)
    for i in range(H):
        for j in range(W):
            for n in range(set1d.size):
                output[i,j] = output[i,j] or (arr2d[i,j] == set1d[n])
    return output

@nb.njit
def compute_boxes(indices, binary_tag_mask):
    """Compute 2d bounding boxes for highlighted pixels"""
    H, W = binary_tag_mask.shape
    num_u = indices.max() + 1
    x_min = np.full(num_u, W-1, dtype=np.int32)
    y_min = np.full(num_u, H-1, dtype=np.int32)
    x_max = np.full(num_u, -1, dtype=np.int32)
    y_max = np.full(num_u, -1, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            idx = indices[y, x]
            tag_is_present = binary_tag_mask[y, x]
            if tag_is_present:
                x_min[idx] = min(x_min[idx], x)
                x_max[idx] = max(x_max[idx], x)
                y_min[idx] = min(y_min[idx], y)
                y_max[idx] = max(y_max[idx], y)
    return np.stack((x_min, y_min, x_max, y_max), axis=-1)

# Deterministic, but probably slow. Good enough for visualization.
def arr2color(e):
    s = np.random.RandomState(np.array(e, dtype=np.uint32))
    return (np.asarray(colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)) * 255).astype(np.uint8)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--boxes', action='store_true')
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    # Load images & masks
    object_segmentation_mask = recover(np.load(get_frame_path(args.folder, 0, args.frame, 'ObjectSegmentation_npz')))
    instance_segmentation_mask = recover(np.load(get_frame_path(args.folder, 0, args.frame, 'InstanceSegmentation_npz')))
    image = imread(get_frame_path(args.folder, 0, args.frame, "Image_png"))
    object_json = json.loads(get_frame_path(args.folder, 0, args.frame, 'Objects_json').read_text())
    H, W = object_segmentation_mask.shape
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

    # Identify objects visible in the image
    unique_object_idxs = set(np.unique(object_segmentation_mask))
    present_objects = [obj for obj in object_json if (obj['object_index'] in unique_object_idxs)]

    # Complain if the query isn't valid/present
    unique_names = sorted({q['name'] for q in present_objects})
    if args.query is None:
        print('`--query` not specified. Choices are:')
        for qn in unique_names:
            print(f"- {qn}")
        sys.exit(0)
    elif not any((args.query.lower() in name.lower()) for name in unique_names):
        print(f'"{args.query}" doesn\'t match any object names in this image. Choices are:')
        for qn in unique_names:
            print(f"- {qn}")
        sys.exit(0)

    # Mask the pixels with any relevant object
    objects_to_highlight = [obj for obj in present_objects if (args.query.lower() in obj['name'].lower())]
    highlighted_pixels = should_highlight_pixel(object_segmentation_mask, np.array([o['object_index'] for o in objects_to_highlight]))
    assert highlighted_pixels.dtype == bool

    # Assign unique colors to each object instance
    combined_mask, _ = pack([object_segmentation_mask, instance_segmentation_mask], 'h w *')
    combined_mask = rearrange(combined_mask, 'h w d -> (h w) d')
    uniq_instances, indices = np.unique(combined_mask, return_inverse=True, axis=0) # this line is the bottleneck
    unique_colors = np.stack([arr2color(row) for row in uniq_instances])

    if args.boxes:
        bbox = compute_boxes(indices.reshape((H, W)), highlighted_pixels)
        m = bbox[:,3] >= 0 # Ignore objects which weren't queried
        bbox = bbox[m]
        unique_colors = unique_colors[m]
        uniq_instances = uniq_instances[m]
        canvas = np.copy(image)
        for (x_min, y_min, x_max, y_max), color, idx, ui in zip(bbox, unique_colors, np.arange(m.size)[m], uniq_instances):
            points = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
            for i in range(4):
                canvas = cv2.line(canvas, points[i], points[(i+1)%4], color=color.tolist(), thickness=2)
    else:
        colors_for_instances = unique_colors[indices].reshape((H, W, 3))
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for obj in objects_to_highlight:
            m = repeat(object_segmentation_mask == obj['object_index'], 'H W -> H W 3')
            canvas[m] = colors_for_instances[m]

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "B.png", canvas)
    print(f'Wrote {args.output / "B.png"}')
