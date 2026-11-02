# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen terrain mesh utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/utils/mesh.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import cv2
import numpy as np
import trimesh
from numpy import ascontiguousarray as AC

from infinigen.core.util import blender as butil
from infinigen.terrain.utils import mesh as _default
from infinigen.terrain.utils.kernelizer_util import ATTRTYPE_FIELDS, NPTYPEDIM_ATTR
from infinigen.terrain.utils.mesh import (
    NormalMode,
    object_from_VF,
    object_to_face_attributes,
    object_to_vertex_attributes,
)


def convert_face_array(face_array):
    l = face_array.shape[0]
    min_indices = np.argmin(face_array, axis=1)
    rows = np.arange(l)
    u = face_array[rows, min_indices]
    v = face_array[rows, (min_indices + 1) % 3]
    w = face_array[rows, (min_indices + 2) % 3]
    return np.stack([u, v, w], -1)


class Mesh(_default.Mesh):
    def __init__(
        self,
        normal_mode=NormalMode.Mean,
        path=None,
        heightmap=None,
        L=None,
        downsample=1,
        vertices=None,
        faces=None,
        vertex_attributes=None,
        mesh=None,
        obj=None,
        mesh_only=False,
        **kwargs,
    ):
        self.normal_mode = normal_mode
        self.face_attributes = {}
        if path is not None:
            geometry = trimesh.load(path, process=False).geometry
            key = list(geometry.keys())[0]
            _trimesh = geometry[key]
        elif heightmap is not None:
            N = heightmap.shape[0]
            heightmap = cv2.resize(heightmap, (N // downsample, N // downsample))
            N = heightmap.shape[0]
            verts = np.zeros((N, N, 3))
            for i in range(N):
                verts[i, :, 0] = (-1 + 2 * i / (N - 1)) * L / 2
            for j in range(N):
                verts[:, j, 1] = (-1 + 2 * j / (N - 1)) * L / 2
            verts[:, :, 2] = heightmap
            verts = verts.reshape((-1, 3))
            faces = np.zeros((2, N - 1, N - 1, 3), np.int32)
            for i in range(N - 1):
                faces[0, i, :, :] += [i * N, (i + 1) * N, i * N]
                faces[1, i, :, :] += [i * N, (i + 1) * N, (i + 1) * N]
            for j in range(N - 1):
                faces[0, :, j, :] += [j, j, j + 1]
                faces[1, :, j, :] += [j + 1, j, j + 1]
            faces = faces.reshape((-1, 3))
            _trimesh = trimesh.Trimesh(verts, faces)
        elif vertices is not None:
            _trimesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces.astype(np.int32),
                vertex_attributes=vertex_attributes,
                process=False,
            )
        elif mesh is not None:
            _trimesh = mesh
        elif obj is not None:
            verts_bpy = obj.data.vertices
            faces_bpy = obj.data.polygons
            # read as float32, the property's own type (Blender's buffer fast
            # path), then widen; the values equal those of a float64 read
            verts = np.zeros((len(verts_bpy) * 3), dtype=np.float32)
            verts_bpy.foreach_get("co", verts)
            verts = verts.astype(float).reshape((-1, 3))
            faces = np.zeros((len(faces_bpy) * 3), dtype=np.int32)
            faces_bpy.foreach_get("vertices", faces)
            faces = faces.reshape((-1, 3))
            _trimesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            if not mesh_only:
                vertex_attributes = object_to_vertex_attributes(obj)
                _trimesh.vertex_attributes.update(vertex_attributes)
                self.face_attributes.update(object_to_face_attributes(obj))
            for key in kwargs:
                setattr(self, key, kwargs[key])
        else:
            _trimesh = trimesh.Trimesh(
                vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int32)
            )

        self._trimesh = _trimesh

    def make_unique(self):
        transposed_array = self.vertices.T
        sorted_indices = np.lexsort(transposed_array)
        self.vertices = self.vertices[sorted_indices]
        inverse = np.empty_like(sorted_indices)  # == np.argsort(sorted_indices)
        inverse[sorted_indices] = np.arange(len(sorted_indices))
        self.faces = inverse[self.faces]
        for attr_name in self.vertex_attributes:
            self.vertex_attributes[attr_name] = self.vertex_attributes[attr_name][
                sorted_indices
            ]
        self.faces = _default.convert_face_array(self.faces)
        transposed_array = self.faces.T
        sorted_indices = np.lexsort(transposed_array)
        self.faces = self.faces[sorted_indices]

    def export_blender(self, name, collection="Collection", material=None):
        self.make_unique()
        new_object = object_from_VF(name, self.vertices, self.faces)
        for attr_name in self.vertex_attributes:
            attr_name_ls = attr_name.lstrip("_")  # this is because of trimesh bug
            dim = (
                self.vertex_attributes[attr_name].shape[1]
                if self.vertex_attributes[attr_name].ndim != 1
                else 1
            )
            type_key = NPTYPEDIM_ATTR[
                (str(self.vertex_attributes[attr_name].dtype), dim)
            ]
            new_object.data.attributes.new(
                name=attr_name_ls, type=type_key, domain="POINT"
            )
            new_object.data.attributes[attr_name_ls].data.foreach_set(
                ATTRTYPE_FIELDS[type_key],
                AC(self.vertex_attributes[attr_name].reshape(-1)),
            )
        if material is not None:
            new_object.data.materials.append(material)

        butil.put_in_collection(bpy.data.objects[name], butil.get_collection("terrain"))

        # sets sharp_face on every face, as the shade_flat operator does,
        # without an operator call
        new_object.data.shade_flat()

        return new_object

    def cat(meshes):
        meshes = list(meshes)
        vertex_count = sum(len(mesh.vertices) for mesh in meshes)
        face_count = sum(len(mesh.faces) for mesh in meshes)

        vertex_dtypes = [np.dtype(float)]
        face_dtypes = [np.dtype(int)]
        attribute_info = {}
        vertex_offset = 0
        for mesh in meshes:
            vertex_dtypes.append(mesh.vertices.dtype)
            face_dtypes.append((mesh.faces + vertex_offset).dtype)
            for name in mesh.vertex_attributes:
                value = mesh.vertex_attributes[name]
                if value.ndim == 1:
                    value = value.reshape((-1, 1))
                    mesh.vertex_attributes[name] = value
                if name not in attribute_info:
                    attribute_info[name] = {
                        "width": value.shape[1],
                        "dtypes": [value.dtype],
                    }
                else:
                    attribute_info[name]["dtypes"].append(value.dtype)
            vertex_offset += len(mesh.vertices)

        vertices = np.empty((vertex_count, 3), dtype=np.result_type(*vertex_dtypes))
        faces = np.empty((face_count, 3), dtype=np.result_type(*face_dtypes))
        vertex_attributes = {
            name: np.zeros(
                (vertex_count, info["width"]),
                dtype=np.result_type(*info["dtypes"]),
            )
            for name, info in attribute_info.items()
        }

        vertex_offset = 0
        face_offset = 0
        for mesh in meshes:
            next_vertex_offset = vertex_offset + len(mesh.vertices)
            next_face_offset = face_offset + len(mesh.faces)
            vertices[vertex_offset:next_vertex_offset] = mesh.vertices
            faces[face_offset:next_face_offset] = mesh.faces + vertex_offset
            for name, value in mesh.vertex_attributes.items():
                vertex_attributes[name][vertex_offset:next_vertex_offset] = value
            vertex_offset = next_vertex_offset
            face_offset = next_face_offset

        return _default.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_attributes=vertex_attributes,
        )
