# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from einops import pack, rearrange
from imageio.v3 import imread, imwrite
from numpy.linalg import inv
from tqdm import tqdm

from ..compress_masks import recover
from ..dataset_loader import get_frame_path

"""
Usage: python -m tools.ground_truth.bounding_boxes_3d <scene-folder> <frame-index> [--query <query>]
Output:
- testbed
    - A.png # Original image
    - B.png # Original image + 3D-bounding-boxes for the provided query
"""

def transform(T, p):
    assert T.shape == (4,4)
    p = T[:3,:3] @ p.T
    return (p + T[:3, [3]]).T

def calc_bbox_pts(min_pt, max_pt):
    min_x, min_y, min_z = min_pt
    max_x, max_y, max_z = max_pt
    points = np.asarray([ # 8 x 2
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [max_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [max_x, max_y, max_z],
    ])

    faces = np.asarray([ # 6 x 4
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ])
    faces = faces[:,[0,1,3,2]]

    return points, faces

# Deterministic, but probably slow. Good enough for visualization.
def arr2color(e):
    s = np.random.RandomState(np.array(e, dtype=np.uint32))
    return (np.asarray(colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)) * 255).astype(np.uint8)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    object_segmentation_mask = recover(np.load(get_frame_path(args.folder, 0, args.frame, 'ObjectSegmentation_npz')))
    instance_segmentation_mask = recover(np.load(get_frame_path(args.folder, 0, args.frame, 'InstanceSegmentation_npz')))
    image = imread(get_frame_path(args.folder, 0, args.frame, "Image_png"))
    object_json = json.loads(get_frame_path(args.folder, 0, args.frame, 'Objects_json').read_text())
    camview = np.load(get_frame_path(args.folder, 0, args.frame, 'camview_npz'))

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

    H, W, _ = image.shape
    camera_pose = camview['T']
    K = camview['K']

    # Assign unique colors to each object instance
    combined_mask, _ = pack([object_segmentation_mask, instance_segmentation_mask], 'h w *')
    combined_mask = rearrange(combined_mask, 'h w d -> (h w) d')
    visible_instances = np.unique(combined_mask, axis=0) # this line is a bottleneck
    visible_instances = {tuple(row) for row in visible_instances}

    boxes_to_draw = []
    for obj in tqdm(present_objects, desc='Identifying boxes to draw'):
        if args.query.lower() in obj['name'].lower():
            for instance_id, model_mat in zip(obj['instance_ids'], np.asarray(obj['model_matrices'])):
                if ((obj['object_index'],) + tuple(instance_id)) in visible_instances:
                    boxes_to_draw.append(dict(model_mat=model_mat, min=obj['min'], max=obj['max'], color=arr2color(instance_id).tolist()))

    canvas = np.copy(image)
    for bbox in tqdm(boxes_to_draw, desc='Drawing boxes'):
        if bbox['min'] is None: # Object has no volume (e.g. a light/camera)
            continue
        min_pt = np.asarray(bbox['min'])
        max_pt = np.asarray(bbox['max'])
        size = np.linalg.norm(max_pt - min_pt)
        bbox_points, faces = calc_bbox_pts(min_pt, max_pt)
        bbox_points_wc = transform(bbox['model_mat'], bbox_points)
        bbox_points_cc = transform(inv(camera_pose), bbox_points_wc)
        bbox_points_h = (K @ bbox_points_cc.T).T
        bbox_points_uv = (bbox_points_h[:,:2] / bbox_points_h[:,[2]]).astype(int)
        if bbox_points_h[:,2].min() < 0: # bbox goes behind the camera
            continue
        points_in_faces_uv = bbox_points_uv[faces.flatten()].reshape((6, 4, 2))
        sign = np.cross(points_in_faces_uv[:, 1] - points_in_faces_uv[:, 0], points_in_faces_uv[:, 2] - points_in_faces_uv[:, 0])
        sign = sign * np.array([-1, 1, 1, -1, -1, 1])

        for is_visible, indices in zip(sign < 0, faces):
            if is_visible:
                for i in range(4):
                    canvas = cv2.line(canvas, bbox_points_uv[indices[i]], bbox_points_uv[indices[(i+1)%4]], color=bbox['color'], thickness=1)

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "B.png", canvas)
    print(f'Wrote {args.output / "B.png"}')
