# COPYRIGHT

# Original files authored by Marian Kleineberg: https://github.com/marian42/mesh_to_sdf/tree/master


# All other contributions:
# Copyright (c) Princeton University.
# Licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Authors: Zeyu Ma


from .scan import Scan, get_camera_transform_looking_at_origin

import trimesh
import logging
import numpy as np
from sklearn.neighbors import KDTree
import math
import pyrender
from .utils import sample_uniform_points_in_unit_sphere
from .utils import get_raster_points, check_voxels

logging.getLogger("trimesh").setLevel(9000)

class BadMeshException(Exception):
    pass

class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans

        self.kd_tree = KDTree(points)

    def get_random_surface_points(self, count, use_scans=True):
        if use_scans:
            indices = np.random.choice(self.points.shape[0], count)
            return self.points[indices, :]
        else:
            samples, index = trimesh.sample.sample_surface(mesh=self.mesh, count=count, face_weight=None, seed=0)
            return samples

    def get_sdf(self, query_points, use_depth_buffer=False, sample_count=11, return_gradients=False):
        if use_depth_buffer:
            distances, indices = self.kd_tree.query(query_points)
            distances = distances.astype(np.float32).reshape(-1)
            inside = ~self.is_outside(query_points)
            distances[inside] *= -1

            if return_gradients:
                gradients = query_points - self.points[indices[:, 0]]
                gradients[inside] *= -1

        else:
            distances, indices = self.kd_tree.query(query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_from_surface = query_points[:, np.newaxis, :] - closest_points
            inside = np.einsum('ijk,ijk->ij', direction_from_surface, self.normals[indices]) < 0
            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1

            if return_gradients:
                gradients = direction_from_surface[:, 0]
                gradients[inside] *= -1

        if return_gradients:
            near_surface = np.abs(distances) < math.sqrt(0.0025**2 * 3) * 3 # 3D 2-norm stdev * 3
            gradients = np.where(near_surface[:, np.newaxis], self.normals[indices[:, 0]], gradients)
            gradients /= np.linalg.norm(gradients, axis=1)[:, np.newaxis]
            return distances, gradients
        else:
            return distances

    def get_sdf_in_batches(self, query_points, use_depth_buffer=False, sample_count=11, batch_size=1000000, return_gradients=False):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points, use_depth_buffer=use_depth_buffer, sample_count=sample_count, return_gradients=return_gradients)

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf(points, use_depth_buffer=use_depth_buffer, sample_count=sample_count, return_gradients=return_gradients)
            for points in np.array_split(query_points, n_batches)
        ]
        if return_gradients:
            distances = np.concatenate([batch[0] for batch in batches])
            gradients = np.concatenate([batch[1] for batch in batches])
            return distances, gradients
        else:
            return np.concatenate(batches) # distances

    def get_voxels(self, voxel_resolution, use_depth_buffer=False, sample_count=11, pad=False, check_result=False, return_gradients=False):
        result = self.get_sdf_in_batches(get_raster_points(voxel_resolution), use_depth_buffer, sample_count, return_gradients=return_gradients)
        if not return_gradients:
            sdf = result
        else:
            sdf, gradients = result
            voxel_gradients = np.reshape(gradients, (voxel_resolution, voxel_resolution, voxel_resolution, 3))
        
        voxels = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

        if check_result and not check_voxels(voxels):
            raise BadMeshException()

        if pad:
            voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

        if return_gradients:
            if pad:
                voxel_gradients = np.pad(voxel_gradients, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge')
            return voxels, voxel_gradients
        else:
            return voxels

    def sample_sdf_near_surface(self, number_of_points=500000, use_scans=True, sign_method='normal', normal_sample_count=11, min_size=0, return_gradients=False):
        query_points = []
        surface_sample_count = int(number_of_points * 47 / 50) // 2
        surface_points = self.get_random_surface_points(surface_sample_count, use_scans=use_scans)
        query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
        query_points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))
        
        unit_sphere_sample_count = number_of_points - surface_points.shape[0] * 2
        unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
        query_points.append(unit_sphere_points)
        query_points = np.concatenate(query_points).astype(np.float32)

        if sign_method == 'normal':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count, return_gradients=return_gradients)
        elif sign_method == 'depth':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=True, return_gradients=return_gradients)
        else:
            raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
        if return_gradients:
            sdf, gradients = sdf

        if min_size > 0:
            model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
            if model_size < min_size:
                raise BadMeshException()

        if return_gradients:
            return query_points, sdf, gradients
        else:
            return query_points, sdf

    def show(self):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(self.points, normals=self.normals))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
        
    def is_outside(self, points):
        result = None
        for scan in self.scans:
            if result is None:
                result = scan.is_visible(points)
            else:
                result = np.logical_or(result, scan.is_visible(points))
        return result

def get_equidistant_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta

def create_from_scans(mesh, bounding_radius=1, scan_count=100, scan_resolution=400, calculate_normals=True):
    scans = []

    for phi, theta in get_equidistant_camera_angles(scan_count):
        camera_transform = get_camera_transform_looking_at_origin(phi, theta, camera_distance=2 * bounding_radius)
        scans.append(Scan(mesh,
            camera_transform=camera_transform,
            resolution=scan_resolution,
            calculate_normals=calculate_normals,
            fov=1.0472,
            z_near=bounding_radius * 1,
            z_far=bounding_radius * 3
        ))

    return SurfacePointCloud(mesh, 
        points=np.concatenate([scan.points for scan in scans], axis=0),
        normals=np.concatenate([scan.normals for scan in scans], axis=0) if calculate_normals else None,
        scans=scans
    )

def sample_from_mesh(mesh, sample_point_count=10000000, calculate_normals=True):
    points, face_indices = trimesh.sample.sample_surface(mesh=mesh, count=sample_point_count, face_weight=None, seed=0)
    if calculate_normals:
        normals = mesh.face_normals[face_indices]

    return SurfacePointCloud(mesh, 
        points=points,
        normals=normals if calculate_normals else None,
        scans=None
    )