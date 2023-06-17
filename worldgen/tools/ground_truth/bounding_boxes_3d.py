import argparse
import json
from itertools import chain
from pathlib import Path

import colorsys
import cv2
import numpy as np
from imageio.v3 import imread, imwrite
from numpy.linalg import inv


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

if __name__ == "__main__":

    import os,sys
    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--query', type=str)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    folder_data = json.loads((args.folder / "summary.json").read_text())

    depth_paths = folder_data["Depth_"]['npy']["00"]["00"]
    image_paths = folder_data["Image"]['png']["00"]["00"]
    Ks = folder_data["Camera Intrinsics"]['npy']["00"]["00"]
    Ts = folder_data["Camera Pose"]['npy']["00"]["00"]

    frame = f"{args.frame:04d}"

    image = imread(args.folder / image_paths[frame])
    H, W, _ = image.shape
    camera_pose = np.load(args.folder / Ts[frame])
    K = np.load(args.folder / Ks[frame])

    tag_mask = np.load(args.folder / folder_data["TagSegmentation_"]['npy']["00"]["00"][f"{args.frame:04d}"])
    tag_mask = cv2.resize(tag_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    tag_lookup = json.loads((args.folder / folder_data["Mask Tags"][f"{args.frame:04d}"]).read_text())
    tag_lookup_rev = {v:k for k,v in tag_lookup.items()}
    tags_in_this_image = set(chain.from_iterable(tag_lookup_rev[e].split('.') for e in np.unique(tag_mask) if e > 0))

    bboxes_path = args.folder / folder_data["BoundingBoxes_"]["json"]["00"]["00"][frame]
    bounding_boxes = json.loads((args.folder / folder_data["BoundingBoxes_"]["json"]["00"]["00"][frame]).read_text())
    unique_tag_numbers = set(np.unique(tag_mask))

    canvas = np.copy(image)

    if args.query is None:
        print('`--query` not specified. Choices are:')
        for q in tags_in_this_image:
            print(f"- {q}")
    elif args.query not in tags_in_this_image:
        print(f'"{args.query}" doesn\'t match any tag in this image. Choices are:')
        for q in tags_in_this_image:
            print(f"- {q}")
    else:
        for k, v in tag_lookup.items():
            if args.query in k.split('.'):
                for bbox in bounding_boxes:
                    tags = bbox['tags']
                    if bbox['min'] is None:
                        continue
                    min_pt = np.asarray(bbox['min'])
                    max_pt = np.asarray(bbox['max'])
                    size = np.linalg.norm(max_pt - min_pt)
                    if (v in tags) and (v in unique_tag_numbers):
                        bbox_points, faces = calc_bbox_pts(min_pt, max_pt)
                        for instance_id, model_mat in bbox['model matrices'].items():
                            bbox_points_wc = transform(np.asarray(model_mat), bbox_points)
                            bbox_points_cc = transform(inv(camera_pose), bbox_points_wc)
                            bbox_points_h = (K @ bbox_points_cc.T).T
                            bbox_points_uv = (bbox_points_h[:,:2] / bbox_points_h[:,[2]]).astype(int)
                            if bbox_points_h[:,2].min() < 0:
                                continue
                            points_in_faces_uv = bbox_points_uv[faces.flatten()].reshape((6, 4, 2))
                            sign = np.cross(points_in_faces_uv[:, 1] - points_in_faces_uv[:, 0], points_in_faces_uv[:, 2] - points_in_faces_uv[:, 0])
                            sign = sign * np.array([-1, 1, 1, -1, -1, 1])

                            gen = np.random.RandomState([bbox['object index'], int(instance_id)])
                            color = (np.asarray(colorsys.hsv_to_rgb(gen.uniform(0, 1), gen.uniform(0.1, 1), 1)) * 255).astype(np.uint8).tolist()
                            for is_visible, indices in zip(sign < 0, faces):
                                if is_visible:
                                    for i in range(4):
                                        canvas = cv2.line(canvas, bbox_points_uv[indices[i]], bbox_points_uv[indices[(i+1)%4]], color=color, thickness=2)


        args.output.mkdir(exist_ok=True)
        imwrite(args.output / "A.png", image)
        print(f'Wrote {args.output / "A.png"}')
        imwrite(args.output / "B.png", canvas)
        print(f'Wrote {args.output / "B.png"}')