# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma, Lingjie Mei


from google_images_search import GoogleImagesSearch
from sklearn.mixture import GaussianMixture
import argparse
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import colorsys
from pathlib import Path


def make_palette(keyword, num_images, num_colors, overwrite=False):
    # define search params
    # option for commonly used search param are shown below for easy reference.
    # For param marked with '##':
    #   - Multiselect is currently not feasible. Choose ONE option only
    #   - This param can also be omitted from _search_params if you do not wish to define any value
    _search_params = {
        'q': keyword,
        'num': num_images,
        'fileType': 'jpg|png',
    }

    # this will search and download:
    folder = f'{os.path.split(os.path.abspath(__file__))[0]}/images/{keyword}'
    if os.path.exists(folder) and not overwrite:
        print("folder existing, skip")
    else:
        # set your environment variables: GCS_DEVELOPER_KEY, GCS_CX
        gis = GoogleImagesSearch(os.environ["GCS_DEVELOPER_KEY"], os.environ["GCS_CX"])
        gis.search(search_params=_search_params, path_to_dir=folder)

    colors = np.zeros((0, 3))
    for image_name in os.listdir(folder):
        if image_name.endswith("svg"): continue
        image = cv2.imread(f"{folder}/{image_name}")
        image = cv2.resize(image, (128, 128))
        image = image[:, :, :3]
        colors = np.concatenate((colors, image.reshape((-1, 3))))
    colors = colors[:, ::-1] / 255

    for i in range(len(colors)):
        colors[i] = colorsys.rgb_to_hsv(*colors[i])

    model = GaussianMixture(num_colors, random_state=0).fit(colors)

    weights = model.weights_.copy()
    index = np.argsort(weights)[::-1]
    weights = weights[index]
    colors_hsv = model.means_.copy()[index]
    cov = model.covariances_.copy()[index]
    colors_rgb = colors_hsv.copy()
    for i in range(num_colors):
        colors_rgb[i] = colorsys.hsv_to_rgb(*colors_rgb[i])
        cov[i] = np.linalg.cholesky(cov[i] + 1e-5 * np.eye(3))

    S = 20
    diagrams = np.zeros((2, S, num_colors, S, 3))
    x, y = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S), indexing="ij")
    for i in range(num_colors):
        diagrams[0, :, i, :, 0] = colors_rgb[i, 0]
        diagrams[0, :, i, :, 1] = colors_rgb[i, 1]
        diagrams[0, :, i, :, 2] = colors_rgb[i, 2]
        diagrams[1, :, i, :, 0] = colors_hsv[i, 0] + cov[i, 0, 0] * x + cov[i, 0, 1] * y
        diagrams[1, :, i, :, 1] = colors_hsv[i, 1] + cov[i, 1, 0] * x + cov[i, 1, 1] * y
        diagrams[1, :, i, :, 2] = colors_hsv[i, 2] + cov[i, 2, 0] * x + cov[i, 2, 1] * y
        for j in range(S):
            for k in range(S):
                diagrams[1, j, i, k] = colorsys.hsv_to_rgb(*diagrams[1, j, i, k])

    diagrams = np.clip(diagrams * 256, a_min=0, a_max=255).astype(np.int32)
    diagrams = diagrams.reshape((2 * S, num_colors * S, 3))

    Path(f'{os.path.split(os.path.abspath(__file__))[0]}/images').mkdir(parents=True, exist_ok=True)
    Path(f'{os.path.split(os.path.abspath(__file__))[0]}/json').mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(20, 5))
    plt.imshow(diagrams)
    plt.savefig(f'{os.path.split(os.path.abspath(__file__))[0]}/images/{keyword}.png')

    colors_rgb = np.clip(colors_rgb * 256, a_min=0, a_max=255).astype(np.int32)
    with open(f"{os.path.split(os.path.abspath(__file__))[0]}/json/{keyword}.json", "w") as f:
        f.write("{\n")
        f.write('    "color": {\n')
        for i, color in enumerate(colors_rgb):
            f.write(f'        "{i}": "#{color[0]:02X}{color[1]:02X}{color[2]:02X}",\n')
        f.write("    },\n")
        f.write('    "hsv": [\n')
        for color_hsv in colors_hsv:
            f.write(f'        [{color_hsv[0]}, {color_hsv[1]}, {color_hsv[2]}],\n')
        f.write("    ],\n")
        f.write('    "std": [\n')
        for std in cov:
            covs = ','.join([str(x) for x in std.reshape(-1)])
            f.write(f'        [{covs}],\n')
        f.write("    ],\n")
        f.write('    "prob": [\n')
        for i in range(num_colors):
            f.write(f'        {weights[i]},\n')
        f.write("    ]\n")
        f.write("}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', type=str)
    parser.add_argument('-i', '--num_images', default=10)
    parser.add_argument('-c', '--num_colors', default=10)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()
    make_palette(args.keyword, args.num_images, args.num_colors, args.overwrite)
