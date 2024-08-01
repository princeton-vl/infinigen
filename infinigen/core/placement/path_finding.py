# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import itertools

import mathutils
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def camera_rotation_matrix(pointing_direction, up_vector):
    forward = pointing_direction / np.linalg.norm(pointing_direction)
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return np.column_stack((right, up, forward))


def path_finding(
    bvhtree, bounding_box, start_pose, end_pose, resolution=100000, margin=0.1
):
    volume = np.product(bounding_box[1] - bounding_box[0])
    N = np.floor(
        (bounding_box[1] - bounding_box[0]) * (resolution / volume) ** (1 / 3)
    ).astype(np.int32)
    NN = np.product(N)
    # print(f"{N=}")
    start_location, start_rotation = start_pose
    end_location, end_rotation = end_pose
    margin_d = np.ceil((resolution / volume) ** (1 / 3) * margin)
    row = []
    col = []
    data = []

    def freespace_ray_check(a, b, margin=0):
        v = b - a
        location, *_ = bvhtree.ray_cast(a, v, v.length)
        if location is not None:
            return False
        if margin != 0:
            if v[0] != 0:
                perp = mathutils.Vector([v[1], -v[0], 0])
            else:
                perp = mathutils.Vector([0, v[2], -v[1]])
            offset = v.cross(perp)
            offset *= margin / offset.length
            check_N = 10
            angle = np.pi * 2 / check_N
            for i in range(check_N):
                location, *_ = bvhtree.ray_cast(a + offset, v, v.length)
                if location is not None:
                    return False
                tar_direction = offset.cross(v)
                tar_direction *= margin / tar_direction.length
                offset = offset * np.cos(angle) + tar_direction * np.sin(angle)
        return True

    def index(i, j, k):
        return i * N[1] * N[2] + j * N[2] + k

    x, y, z = np.meshgrid(
        np.arange(N[0]), np.arange(N[1]), np.arange(N[2]), indexing="ij"
    )
    x = (
        bounding_box[0][0]
        + (bounding_box[1][0] - bounding_box[0][0]) * (x + 0.5) / N[0]
    )
    y = (
        bounding_box[0][1]
        + (bounding_box[1][1] - bounding_box[0][1]) * (y + 0.5) / N[1]
    )
    z = (
        bounding_box[0][2]
        + (bounding_box[1][2] - bounding_box[0][2]) * (z + 0.5) / N[2]
    )
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

    start_index = index(
        *np.floor(
            (np.array(start_location) - bounding_box[0])
            / (bounding_box[1] - bounding_box[0])
            * N
        ).astype(np.int32)
    )
    end_index = index(
        *np.floor(
            (np.array(end_location) - bounding_box[0])
            / (bounding_box[1] - bounding_box[0])
            * N
        ).astype(np.int32)
    )
    if end_index == start_index:
        return None

    x[start_index] = start_pose[0].x
    y[start_index] = start_pose[0].y
    z[start_index] = start_pose[0].z
    x[end_index] = end_pose[0].x
    y[end_index] = end_pose[0].y
    z[end_index] = end_pose[0].z

    penalty = 99
    for i, j, k in list(itertools.product(range(N[0]), range(N[1]), range(N[2]))):
        index_ijk = index(i, j, k)
        pos_from = mathutils.Vector([x[index_ijk], y[index_ijk], z[index_ijk]])
        for di, dj, dk in [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, -1, 0],
            [0, 1, -1],
            [1, 0, -1],
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if (
                ni >= 0
                and nj >= 0
                and nk >= 0
                and ni < N[0]
                and nj < N[1]
                and nk < N[2]
            ):
                index_nijk = index(ni, nj, nk)
                pos_to = mathutils.Vector([x[index_nijk], y[index_nijk], z[index_nijk]])
                connected = freespace_ray_check(pos_from, pos_to)
                if connected:
                    row.append(index_ijk)
                    col.append(index_nijk)
                    data.append(1 if dk == 0 else penalty)
                    row.append(index_nijk)
                    col.append(index_ijk)
                    data.append(1 if dk == 0 else penalty)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    A = csr_matrix((data, (row, col)), shape=(NN, NN))
    G = nx.from_scipy_sparse_array(A)

    boundaries = []
    n_neighbors = np.array(A.sum(axis=0))[0]

    for i in range(NN):
        if n_neighbors[i] != 8 + 10 * penalty:
            boundaries.append(i)

    lengths_dict = nx.multi_source_dijkstra_path_length(G, boundaries, weight="weight")
    lengths = np.zeros(NN) + np.inf
    for n in lengths_dict:
        lengths[n] = lengths_dict[n]

    mask1 = lengths[row] >= margin_d
    mask2 = lengths[col] >= margin_d
    row = row[mask1 & mask2]
    col = col[mask1 & mask2]
    data = data[mask1 & mask2]

    A = csr_matrix((data, (row, col)), shape=(NN, NN))
    G = nx.from_scipy_sparse_array(A)

    try:
        path = nx.shortest_path(G, start_index, end_index, weight="weight")
    except Exception:
        return None

    stack = [start_index]

    for p in path[1:]:
        back = 0
        while freespace_ray_check(
            mathutils.Vector(
                [x[stack[-1 - back]], y[stack[-1 - back]], z[stack[-1 - back]]]
            ),
            mathutils.Vector([x[p], y[p], z[p]]),
            margin=margin,
        ):
            back += 1
            if back == len(stack):
                break
        if back != 1:
            stack = stack[: 1 - back]
        stack.append(p)

    locations = []
    lengths = []
    for i, p in enumerate(stack):
        if i == 0:
            locations.append(start_pose[0])
        elif i == len(stack) - 1:
            locations.append(end_pose[0])
        else:
            locations.append(mathutils.Vector([x[p], y[p], z[p]]))
        if len(locations) >= 2:
            lengths.append((locations[-1] - locations[-2]).length)
    keyframed_poses = []

    for i in range(len(stack)):
        if i == 0:
            keyframed_poses.append((0, *start_pose))
        else:
            if i == len(stack) - 1:
                rotation_euler = end_pose[1]
            else:
                rotation_matrix = mathutils.Matrix(
                    camera_rotation_matrix(
                        np.array(locations[i] - locations[i - 1]), np.array([0, 0, 1])
                    )
                ) @ mathutils.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                rotation_euler = rotation_matrix.to_euler()
                if rotation_euler.y != 0:
                    rotation_euler.y = 0
                    rotation_euler.x += np.pi
                    rotation_euler.z += np.pi
            angle_differece = [
                abs(rotation_euler.z - 2 * np.pi - keyframed_poses[i - 1][2].z),
                abs(rotation_euler.z - keyframed_poses[i - 1][2].z),
                abs(rotation_euler.z + 2 * np.pi - keyframed_poses[i - 1][2].z),
            ]
            rotation_euler.z += (np.argmin(angle_differece) - 1) * 2 * np.pi
            keyframed_poses.append((np.sum(lengths[:i]), locations[i], rotation_euler))
    return keyframed_poses
