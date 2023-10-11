# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os

# ruff: noqa: E402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.
# See https://github.com/opencv/opencv/issues/21326#issuecomment-1008517425

import cv2
import numpy as np

def boundary_smooth(ar, p=0.1):
    N = ar.shape[0]
    P = int(N * p)
    ar = ar.copy()
    increasing_smoother = (1 - np.cos(np.arange(P) / (P - 1) * np.pi)) / 2
    decreasing_smoother = increasing_smoother[::-1]
    ar[:P] *= increasing_smoother.reshape((P, 1))
    ar[-P:] *= decreasing_smoother.reshape((P, 1))
    ar[:, :P] *= increasing_smoother.reshape((1, P))
    ar[:, -P:] *= decreasing_smoother.reshape((1, P))
    return ar


def smooth(arr, k):
    arr = cv2.GaussianBlur(arr, (k, k), 0)
    return arr


def read(input_heightmap_path):
    input_heightmap_path = str(input_heightmap_path)
    assert os.path.exists(input_heightmap_path), f"{input_heightmap_path} does not exists"
    heightmap = cv2.imread(input_heightmap_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).copy().astype(float)
    return heightmap


def grid_distance(source, downsample):
    M = source.shape[0]
    source = cv2.resize(source.astype(float), (downsample, downsample)) > 0.5
    dist = np.zeros_like(source, dtype=np.float32) + 1e9
    N = source.shape[0]
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    boundary = np.zeros_like(source, dtype=bool)
    boundary[:-1, :] |= ~source[1:, :]
    boundary[1:, :] |= ~source[:-1, :]
    boundary[:, :-1] |= ~source[:, 1:]
    boundary[:, 1:] |= ~source[:, :-1]
    boundary &= source
    for i in range(N):
        for j in range(N):
            if boundary[i, j]:
                dist = np.minimum(dist, (((I - i) / N) ** 2 + ((J - j) / N) ** 2) ** 0.5)
    dist[source] = 0
    dist = cv2.resize(dist, (M, M))
    return dist


def sharpen(x):
    return (np.sin((x - 0.5) / 0.5 * np.pi / 2) + 1) / 2


def get_normal(z, grid_size):
    dzdx = np.zeros_like(z)
    dzdx[1:] = z[1:] - z[:-1]
    dzdx[0] = dzdx[1]
    dzdy = np.zeros_like(z)
    dzdy[:, 1:] = z[:, 1:] - z[:, :-1]
    dzdy[:, 0] = dzdx[:, 1]
    n = np.stack((-dzdy, -dzdx, grid_size * np.ones_like(z)), -1)
    n /= np.linalg.norm(n, axis=-1).reshape((n.shape[0], n.shape[1], 1))
    return n