# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import colorsys
import json
import sys
from itertools import chain
from pathlib import Path

import cv2
import numba as nb
import numpy as np
from einops import repeat
from imageio.v3 import imread, imwrite
from numba.types import bool_


@nb.njit
def compute_binary_tag_mask(tag_mask, relevant_tag_numbers):
    H, W = tag_mask.shape
    output = np.zeros((H, W), dtype=bool_)
    for i in range(H):
        for j in range(W):
            for n in range(relevant_tag_numbers.size):
                output[i,j] = output[i,j] or (tag_mask[i,j] == relevant_tag_numbers[n])
    return output

@nb.njit
def compute_boxes(indices, binary_tag_mask):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--query', type=str)
    parser.add_argument('--boxes', action='store_true')
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    folder_data = json.loads((args.folder / "summary.json").read_text())

    tag_mask = np.load(args.folder / folder_data["TagSegmentation"]['npy']["00"]["00"][f"{args.frame:04d}"])

    tag_lookup = json.loads((args.folder / folder_data["Mask Tags"][f"{args.frame:04d}"]).read_text())
    tag_lookup_rev = {v:k for k,v in tag_lookup.items()}
    tags_in_this_image = set(chain.from_iterable(tag_lookup_rev[e].split('.') for e in np.unique(tag_mask) if e > 0))

    if args.query is None:
        print('`--query` not specified. Choices are:')
        for q in tags_in_this_image:
            print(f"- {q}")
        sys.exit(0)
    elif args.query not in tags_in_this_image:
        print(f'"{args.query}" doesn\'t match any tag in this image. Choices are:')
        for q in tags_in_this_image:
            print(f"- {q}")
        sys.exit(0)

    relevant_tag_numbers = np.asarray([i for i,s in tag_lookup_rev.items() if args.query in s.split('.')])
    binary_tag_mask = compute_binary_tag_mask(tag_mask, relevant_tag_numbers)
    assert binary_tag_mask.dtype == bool

    instance_segmentation_mask = np.load(args.folder / folder_data["InstanceSegmentation"]['npy']["00"]["00"][f"{args.frame:04d}"])
    object_segmentation_mask = np.load(args.folder / folder_data["ObjectSegmentation"]['npy']["00"]["00"][f"{args.frame:04d}"])
    combined_mask = np.stack((object_segmentation_mask, instance_segmentation_mask), axis=-1).reshape((-1, 2))
    H, W = tag_mask.shape

    image = imread(args.folder / folder_data["Image"]['png']["00"]["00"][f"{args.frame:04d}"])
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

    uniq, indices = np.unique(combined_mask, return_inverse=True, axis=0) # bottleneck
    random_states = [np.random.RandomState(e[:2].astype(np.uint32)) for e in uniq]
    unique_colors = (np.asarray([colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1) for s in random_states]) * 255).astype(np.uint8)

    if args.boxes:
        bbox = compute_boxes(indices.reshape((H, W)), binary_tag_mask)
        m = bbox[:,3] >= 0
        bbox = bbox[m]
        unique_colors[m]
        uniq = uniq[m]
        canvas = np.copy(image)
        for (x_min, y_min, x_max, y_max), color in zip(bbox, unique_colors):
            points = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
            for i in range(4):
                canvas = cv2.line(canvas, points[i], points[(i+1)%4], color=color.tolist(), thickness=2)
    else:
        colors_for_instances = unique_colors[indices].reshape((H, W, 3))
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for k, v in tag_lookup.items():
            if args.query in k.split('.'):
                m = repeat(tag_mask == v, 'H W -> H W 3')
                canvas[m] = colors_for_instances[m]

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "B.png", canvas)
    print(f'Wrote {args.output / "B.png"}')
