# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os
import numpy as np
import cv2

root_folder = "outputs_scratch/fig1_v6"
scene_types = [
    "forest",
    "river",
    "canyon",
    "under_water",
    "cave",
    "coast",
    "desert",
    "mountain",
    "plain",
]
titles=scene_types

level0_layout = (3, 3)
sublevel_mode = "below"
sublevel_layout = (2, 6)
margin = 0
H, W = (1080, 1920) # resized resolution
with_txt = True

if sublevel_mode == "below":
    subfigure_W = (W + margin) // sublevel_layout[1] - margin
    subfigure_H = subfigure_W * H // W
    block_H = H + (margin + subfigure_H) * sublevel_layout[0] + margin
    block_W = W + margin
else:
    subfigure_H = (H + margin) // sublevel_layout[0] - margin
    subfigure_W = subfigure_H * W // H
    block_W = W + (margin + subfigure_W) * sublevel_layout[1] + margin
    block_H = H + margin

canvas = np.zeros((block_H * level0_layout[0] - margin, block_W * level0_layout[1] - margin, 3)) + 255
for i, scene_type, title in zip(range(len(scene_types)), scene_types, titles):
    y, x = i // level0_layout[1], i % level0_layout[1]
    for j in range(sublevel_layout[0] * sublevel_layout[1] + 1):
        print(scene_type, j)
        folder = f'{scene_type}_{j}'
        path = f"{root_folder}/{folder}/frames_{folder}_resmpl0"
        if not os.path.exists(path): 
            print(f'{path} did not exist')
            continue
        image_path = [x for x in os.listdir(path) if x.startswith("Noisy") and x.endswith(".png")]
        if image_path == []: continue
        image_path = f"{path}/{image_path[0]}"
        image = cv2.imread(image_path)
        if j == 0:
            image = cv2.resize(image, (W, H))
            canvas[y * block_H: y * block_H + H, x * block_W: x * block_W + W] = image
        else:
            y_j, x_j = (j - 1) // sublevel_layout[1], (j - 1) % sublevel_layout[1]
            subfigure_W0, subfigure_H0 = subfigure_W, subfigure_H
            if sublevel_mode == "below":
                assert(sublevel_layout[1] > 1)
                if x_j == sublevel_layout[1] - 1:
                    subfigure_W0 = W - (subfigure_W + margin) * (sublevel_layout[1] - 1)
            else:
                assert(sublevel_layout[0] > 1)
                if y_j == sublevel_layout[0] - 1:
                    subfigure_H0 = H - (subfigure_H + margin) * (sublevel_layout[0] - 1)
            image = cv2.resize(image, (subfigure_W0, subfigure_H0))
            if sublevel_mode == "below":
                H_offset = H + margin + (margin + subfigure_H) * y_j
                W_offset = (margin + subfigure_W) * x_j
            else:
                H_offset = (margin + subfigure_H) * y_j
                W_offset = W + margin + (margin + subfigure_W) * x_j
            canvas[y * block_H + H_offset: y * block_H + H_offset + subfigure_H0, x * block_W + W_offset: x * block_W + W_offset + subfigure_W0] = image
    if with_txt:
        if np.sum(canvas[y * block_H + 5, x * block_W + 5]) > 255 * 1.5:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        cv2.putText(canvas, title, (x * block_W, y * block_H + 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=4)
cv2.imwrite(f"{root_folder}/figure.png", canvas)
