# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from skimage.measure import find_contours, marching_cubes
from scipy.ndimage import convolve
from infinigen.core.util import blender as butil

from infinigen.assets.utils.object import data2mesh


def mesh_grid(n, sizes):
    shapes = [int((h - l) * n) + 1 for l, h in sizes]
    return np.meshgrid(*(np.linspace(*sz, sh) for sz, sh in zip(sizes, shapes)))


def init_mesh_3d(n, sizes):
    fn = lambda x, y, z: uniform(.5, 1) * (x - uniform(-.2, .2)) ** 2 + uniform(.5, 1) * (
            y - uniform(-.2, .2)) ** 2 + uniform(.1, .2) * z ** 2 < .2 * .2
    extend = lambda f: uniform(0, 1, f.shape) < convolve(f, np.ones((3, 3, 3)))

    x, y, z = mesh_grid(n, sizes)
    f = fn(x, y, z)
    a = np.where(f, uniform(.1, .5, x.shape), 0) + uniform(0, .02, x.shape)
    b = np.where(extend(f), 1, uniform(-1, 1, x.shape)).astype(float)
    return a, b


def init_mesh_2d(n, sizes):
    fn = lambda x, y: x <= 2 / n
    x, y = mesh_grid(n, sizes)
    f = fn(x, y)
    a = np.where(f, .99, 0) + uniform(0, .01, x.shape)
    b = uniform(-1, 1, x.shape)
    return a, b


def build_laplacian(st, a, b, t, k, dt, tau, eps, alpha, gamma, teq):
    for _ in range(t):
        lap_a = convolve(a, st)
        lap_b = convolve(b, st)
        m = alpha / np.pi * np.arctan(gamma * (teq - b))
        delta_a = (eps * eps * lap_a + a * (1. - a) * (a - .5 + m)) / tau
        delta_b = lap_b + k * delta_a
        a += delta_a * dt
        b += delta_b * dt
    return a, b


def build_laplacian_3d(n=32, t=800, k=2., dt=.0005, tau=.0003, eps=.01, alpha=.9, gamma=10., teq=1.):
    stencil = np.array([[[1, 3, 1], [3, 14, 3], [1, 3, 1]], [[3, 14, 3], [14, -128, 14], [3, 14, 3]],
                           [[1, 3, 1], [3, 14, 3], [1, 3, 1]]]) / 128

    height = 1.5
    sizes = [-1, 1], [-1, 1], [0, height]

    a, b = init_mesh_3d(n, sizes)
    a, b = build_laplacian(stencil * n * n, a, b, t, k, dt, tau, eps, alpha, gamma, teq)

    a = np.pad(a, 1)
    vertices, faces, _, _ = marching_cubes(a, .5)
    vertices -= 1
    vertices /= n
    vertices[:, :-1] -= 1
    x, y, z = vertices.T
    vertices[:, :-1] *= np.expand_dims(
        np.maximum(np.abs(x), np.abs(y)) / (np.sqrt(x ** 2 + y ** 2) + 1e-6) * (1 - z / height) + z / height,
        -1)
    return data2mesh(vertices, [], faces)


def build_laplacian_2d(n=128, t=10000, k=1.5, dt=.0002, tau=.0003, eps=.01, alpha=.9, gamma=10., teq=1.):
    stencil = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]]) / 20
    sizes = [0, 1], [0, 1]

    a, b = init_mesh_2d(n, sizes)

    st = stencil * n * n / 4
    a, b = build_laplacian(st, a, b, t, k, dt, tau, eps, alpha, gamma, teq)

    a = np.pad(a, 1)
    a = np.stack([a, a], axis=-1)
    vertices, faces, _, _ = marching_cubes(a, .5)
    vertices -= 1
    vertices /= n
    mesh = data2mesh(vertices, [], faces)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    vertices_to_remove = [v for v in bm.verts if v.co[-1] > 0]
    bmesh.ops.delete(bm, geom=vertices_to_remove)
    for v in bm.verts:
        x, y, z = v.co
        v.co *= np.maximum(np.abs(x), np.abs(y)) / (np.sqrt(x ** 2 + y ** 2) + 1e-6)
    bm.to_mesh(mesh)
    return data2mesh(vertices, [], faces)
