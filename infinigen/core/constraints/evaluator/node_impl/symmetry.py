# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan
# Acknowledgement: Rotational symmetry code draws inspiration from https://pubs.acs.org/doi/abs/10.1021/ja00046a033 by Zabrodsky et al.


from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from scipy.optimize import linear_sum_assignment

from infinigen.core.constraints.evaluator.indoor_util import blender_objs_from_names


def rotate_vector(vector, angle):
    """Rotate a 2D vector by a given angle."""
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.dot(vector, rotation_matrix.T)


def compute_centroid(objects):
    """Compute the centroid of the provided objects."""
    total_coords = np.zeros(2)
    for obj in objects:
        total_coords += np.array([obj.location.x, obj.location.y])
    return total_coords / len(objects)


def compute_location_asymmetry(objects, centroid):
    """Compute location asymmetry based on the described method."""
    num_objects = len(objects)

    P = np.zeros((num_objects, 2))
    for i, obj in enumerate(objects):
        P[i] = np.array([obj.location.x, obj.location.y]) - centroid
    # print(P)

    # 2. Rotate all P_i so that P_1 is aligned with the x axis
    angle_p1 = np.arctan2(P[0][1], P[0][0])
    for i in range(num_objects):
        P[i] = rotate_vector(P[i], -angle_p1)

    # 3. Normalize P_i by dividing by max norm
    max_norm = max(np.linalg.norm(P, axis=1))
    P /= max_norm
    # print(P)

    # 4. Compute Q as the average of the rotated P_i vectors
    Q = np.zeros(2)
    for i in range(num_objects):
        rotated_p = rotate_vector(P[i], -i * 2 * np.pi / num_objects)
        #        print("rot", P[i], rotated_p)
        Q += rotated_p
    Q /= num_objects

    # 5. and 6. Compute Q_i and find the MSD between Q_i and P_i
    total_msd = 0
    for i in range(num_objects):
        Q_i = rotate_vector(Q, i * 2 * np.pi / num_objects)
        #        print("rot2", Q_i, P[i])
        msd = np.linalg.norm(Q_i - P[i]) ** 2
        total_msd += msd

    return total_msd / num_objects


def compute_orientation_asymmetry(objects, centroid):
    """Compute orientation asymmetry of objects."""
    num_objects = len(objects)

    # 1. Get the orientation vectors
    V = np.zeros((num_objects, 2))
    for i, obj in enumerate(objects):
        # Extract orientation from object's rotation attribute
        V[i] = np.array([np.cos(obj.rotation_euler.z), np.sin(obj.rotation_euler.z)])

    # Rotate all V_i so that V_1 is aligned with the x axis
    angle_v1 = np.arctan2(V[0][1], V[0][0])
    for i in range(num_objects):
        V[i] = rotate_vector(V[i], -angle_v1)

    # Normalize V_i by dividing by max norm
    max_norm = max(np.linalg.norm(V, axis=1))
    V /= max_norm

    # 4. Compute Q as the average of the rotated V_i vectors
    Q = np.zeros(2)
    for i in range(num_objects):
        rotated_v = rotate_vector(V[i], -i * 2 * np.pi / num_objects)
        #        print("rot", P[i], rotated_p)
        Q += rotated_v
    Q /= num_objects

    # 5. and 6. Compute Q_i and find the MSD between Q_i and V_i
    total_msd = 0
    for i in range(num_objects):
        Q_i = rotate_vector(Q, i * 2 * np.pi / num_objects)
        #        print("rot2", Q_i, P[i])
        msd = np.linalg.norm(Q_i - V[i]) ** 2
        total_msd += msd

    return total_msd / num_objects


def sort_objects_clockwise(objects, centroid):
    angles = []
    for obj in objects:
        # Calculate the angle from the centroid to the object
        dx = obj.location.x - centroid[0]
        dy = obj.location.y - centroid[1]
        angle = np.arctan2(dy, dx)
        angles.append((obj, angle))

    # Sort objects based on the angles in descending order for clockwise sorting
    angles.sort(key=lambda x: x[1])

    # Extract the sorted objects
    sorted_objects = [obj for obj, angle in angles]
    return sorted_objects


def compute_total_rotation_asymmetry(a: Union[str, list[str]]) -> float:
    """Compute the total asymmetry."""
    if isinstance(a, str):
        a = [a]
    objects = blender_objs_from_names(a)
    centroid = compute_centroid(objects)
    objects = sort_objects_clockwise(objects, centroid)
    location_asymmetry = compute_location_asymmetry(objects, centroid)
    orientation_asymmetry = compute_orientation_asymmetry(objects, centroid)

    # print("location asym", location_asymmetry, "orient asym", orientation_asymmetry)

    return (location_asymmetry + orientation_asymmetry) / 2


def reflect_point(point, plane_point, plane_normal):
    # Reflect a point across an arbitrary plane defined by a point and a normal.
    to_point = point - plane_point
    distance_to_plane = to_point.dot(plane_normal)
    reflected_point = point - 2 * distance_to_plane * plane_normal
    return reflected_point


# prob doesnt work
def reflect_quaternion(q, n):
    # Decompose the quaternion into scalar and vector parts
    w = q.w
    v = Vector((q.x, q.y, q.z))

    # Reflect the vector part
    v_reflected = v - 2 * v.dot(n) * n

    # Construct the reflected quaternion
    q_reflected = Quaternion((w, v_reflected.x, v_reflected.y, v_reflected.z))

    return q_reflected


def reflect_axis_angle(axis_angle, n):
    axis = Vector((axis_angle[1], axis_angle[2], axis_angle[3]))
    angle = axis_angle[0]
    # Reflect the vector part
    v_reflected = axis - 2 * axis.dot(n) * n
    angle_reflected = -angle

    # Construct the reflected axis angle
    axis_angle_reflected = Vector(
        (angle_reflected, v_reflected.x, v_reflected.y, v_reflected.z)
    )

    return axis_angle_reflected


def reflect(obj, plane_point, plane_normal):
    obj.rotation_mode = "AXIS_ANGLE"
    reflected_position = reflect_point(obj.location, plane_point, plane_normal)
    reflected_axis_angle = reflect_axis_angle(obj.rotation_axis_angle, plane_normal)
    reflected_quaternion = Matrix.Rotation(
        reflected_axis_angle[0], 4, reflected_axis_angle[1:]
    ).to_quaternion()
    return reflected_position, reflected_quaternion


def distance(pos1, pos2):
    # Calculate Euclidean distance between two positions
    return (pos1 - pos2).length


def angle_difference(orient1, orient2):
    # Calculate the angular difference between two orientations represented as quaternions.
    orient1.normalize()
    orient2.normalize()
    dot_product = orient1.dot(orient2)
    # lose directionality information
    dot_product = abs(dot_product)
    dot_product = max(min(dot_product, 1.0), -1.0)
    angle = 2 * np.arccos(dot_product)
    return angle


def weight(obj):
    # Assign a weight based on obj size or other criteria
    bbox = obj.bound_box
    dims = [bbox[i][0] for i in range(8)]
    volume = (max(dims) - min(dims)) ** 3
    return volume


def normalization_factor(objs):
    avg_distance = np.mean(
        [
            distance(obj1.location, obj2.location)
            for obj1 in objs
            for obj2 in objs
            if obj1 != obj2
        ]
    )
    return avg_distance


def bipartite_matching(objs, reflected_objs_data):
    # Use the Hungarian algorithm to find the optimal pairing between objs and reflected_objs
    for obj in objs:
        obj.rotation_mode = "QUATERNION"
    cost_matrix = np.array(
        [
            [
                distance(obj.location, ref[0])
                + angle_difference(obj.rotation_quaternion, ref[1])
                for ref in reflected_objs_data
            ]
            for obj in objs
        ]
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(objs[i], reflected_objs_data[j]) for i, j in zip(row_ind, col_ind)]


def calculate_reflectional_asymmetry(objs, plane_point, plane_normal, visualize=False):
    if visualize:
        fig, ax = plt.subplots()
        # plot plane point and plane normal
        ax.scatter(plane_point.x, plane_point.y, c="g", label="plane point")
        ax.quiver(
            plane_point.x,
            plane_point.y,
            plane_normal.x,
            plane_normal.y,
            color="g",
            label="plane normal",
        )

    reflected_objs_data = [reflect(obj, plane_point, plane_normal) for obj in objs]

    # Use bipartite matching to find optimal pairings
    pairings = bipartite_matching(objs, reflected_objs_data)

    total_deviation = 0
    for original, reflected_data in pairings:
        positional_deviation = distance(original.location, reflected_data[0])
        original.rotation_mode = "QUATERNION"
        angular_deviation = angle_difference(
            original.rotation_quaternion, reflected_data[1]
        )

        weighted_deviation = weight(original) * (
            positional_deviation + angular_deviation
        )
        total_deviation += weighted_deviation
        if visualize:
            # plot the point and the reflected point with different colors
            ax.scatter(
                original.location.x, original.location.y, c="b", label="original point"
            )
            ax.scatter(
                reflected_data[0].x, reflected_data[0].y, c="r", label="reflected point"
            )

    # Normalize based on scene scale or other criteria
    normalized_deviation = total_deviation / normalization_factor(objs)

    symmetry_score = 1 / (1 + normalized_deviation)
    asymmetry_score = 1 - symmetry_score

    for obj in objs:
        obj.rotation_mode = "XYZ"

    if visualize:
        ax.legend()
        plt.show()

    return asymmetry_score
