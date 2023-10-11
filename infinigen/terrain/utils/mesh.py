# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_double, c_int32

import bpy
import cv2
import numpy as np
import trimesh
from numpy import ascontiguousarray as AC
from infinigen.core.util import blender as butil
from infinigen.core.util.logging import Timer
from infinigen.core.util.organization import Attributes

from .camera import getK
from .ctype_util import ASDOUBLE, ASINT, load_cdll, register_func
from .kernelizer_util import ATTRTYPE_DIMS, ATTRTYPE_FIELDS, NPTYPEDIM_ATTR, Vars


class NormalMode:
    Mean = "mean"
    AngleWeighted = "angle_weighted"


def object_to_vertex_attributes(obj, specified=None, skip_internal=True):
    vertex_attributes = {}
    for attr in obj.data.attributes.keys():
        if skip_internal and butil.blender_internal_attr(attr):
            continue
        if ((specified is None) or (specified is not None and attr in specified)) and obj.data.attributes[attr].domain == "POINT":
            type_key = obj.data.attributes[attr].data_type
            tmp = np.zeros(len(obj.data.vertices) * ATTRTYPE_DIMS[type_key], dtype=np.float32)
            obj.data.attributes[attr].data.foreach_get(ATTRTYPE_FIELDS[type_key], tmp)
            vertex_attributes[attr] = tmp.reshape((len(obj.data.vertices), -1))
    return vertex_attributes

def objectdata_from_VF(vertices, faces):
    new_mesh = bpy.data.meshes.new("")
    new_mesh.vertices.add(len(vertices))
    new_mesh.vertices.foreach_set("co", vertices.reshape(-1).astype(np.float32))
    new_mesh.polygons.add(len(faces))
    new_mesh.loops.add(len(faces) * 3)
    loop_total = np.ones(len(faces), np.int32) * 3
    new_mesh.polygons.foreach_set("loop_total", loop_total)
    if len(loop_total) >= 1:
        loop_start = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(loop_total[:-1])))
        new_mesh.polygons.foreach_set("loop_start", loop_start)
    new_mesh.polygons.foreach_set("vertices", faces.reshape(-1))
    new_mesh.update(calc_edges=True)
    return new_mesh

def object_from_VF(name, vertices, faces):
    new_mesh = objectdata_from_VF(vertices, faces)
    new_object = bpy.data.objects.new(name, new_mesh)
    new_object.rotation_euler = (0, 0, 0)
    return new_object

def convert_face_array(face_array):
    l = face_array.shape[0]
    min_indices = np.argmin(face_array, axis=1)
    u = face_array[list(np.arange(l)), min_indices]
    v = face_array[list(np.arange(l)), (min_indices + 1) % 3]
    w = face_array[list(np.arange(l)), (min_indices + 2) % 3]
    return np.stack([u, v, w], -1)

class Mesh:
    def __init__(self, normal_mode=NormalMode.Mean,
        path=None,
        heightmap=None, L=None, downsample=1,
        vertices=None, faces=None, vertex_attributes=None,
        obj=None, mesh_only=False, **kwargs
    ):
        self.normal_mode = normal_mode
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
                verts[:, j, 1] =  (-1 + 2 * j / (N - 1)) * L / 2
            verts[:, :, 2] = heightmap
            verts = verts.reshape((-1, 3))
            faces = np.zeros((2, N - 1, N - 1, 3), np.int32)
            for i in range(N - 1):
                faces[0, i, :, :] += [i * N, (i+1) * N, i * N]
                faces[1, i, :, :] += [i * N, (i+1) * N, (i+1) * N]
            for j in range(N - 1): 
                faces[0, :, j, :] += [j, j, j+1]
                faces[1, :, j, :] += [j+1, j, j+1]
            faces = faces.reshape((-1, 3))
            _trimesh = trimesh.Trimesh(verts, faces)
        elif vertices is not None:
            _trimesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int32), vertex_attributes=vertex_attributes, process=False)
        elif obj is not None:
            verts_bpy = obj.data.vertices
            faces_bpy = obj.data.polygons
            verts = np.zeros((len(verts_bpy)*3), dtype=float)
            verts_bpy.foreach_get("co", verts)
            verts = verts.reshape((-1, 3))
            faces = np.zeros((len(faces_bpy)*3), dtype=np.int32)
            faces_bpy.foreach_get("vertices", faces)
            faces = faces.reshape((-1, 3))
            _trimesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            if not mesh_only:
                vertex_attributes = object_to_vertex_attributes(obj)
                _trimesh.vertex_attributes.update(vertex_attributes)
            for key in kwargs:
                setattr(self, key, kwargs[key])
        else:
            _trimesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int32))
        
        self._trimesh = _trimesh


    def to_trimesh(self):
        return self._trimesh
    
    @property
    def vertex_attributes(self):
        return self._trimesh.vertex_attributes
    
    @vertex_attributes.setter
    def vertex_attributes(self, value):
        self._trimesh.vertex_attributes = value
    
    @property
    def vertices(self):
        return self._trimesh.vertices
    
    @vertices.setter
    def vertices(self, value):
        self._trimesh.vertices = value
    
    @property
    def faces(self):
        return self._trimesh.faces

    @faces.setter
    def faces(self, value):
        self._trimesh.faces = value

    def save(self, path):
        for attr in self._trimesh.vertex_attributes:
            self._trimesh.vertex_attributes[attr] = self._trimesh.vertex_attributes[attr].astype(np.float32)
        self._trimesh.export(path)

    def make_unique(self):
        transposed_array = self.vertices.T
        sorted_indices = np.lexsort(transposed_array)
        self.vertices = self.vertices[sorted_indices]
        self.faces = np.argsort(sorted_indices)[self.faces]
        for attr_name in self.vertex_attributes:
            self.vertex_attributes[attr_name] = self.vertex_attributes[attr_name][sorted_indices]
        self.faces = convert_face_array(self.faces)
        transposed_array = self.faces.T
        sorted_indices = np.lexsort(transposed_array)
        self.faces = self.faces[sorted_indices]

    def export_blender(self, name, collection="Collection", material=None):
        self.make_unique()
        new_object = object_from_VF(name, self.vertices, self.faces)
        for attr_name in self.vertex_attributes:
            attr_name_ls = attr_name.lstrip("_") # this is because of trimesh bug
            dim = self.vertex_attributes[attr_name].shape[1] if self.vertex_attributes[attr_name].ndim != 1 else 1
            type_key = NPTYPEDIM_ATTR[(str(self.vertex_attributes[attr_name].dtype), dim)]
            new_object.data.attributes.new(name=attr_name_ls, type=type_key, domain='POINT')
            new_object.data.attributes[attr_name_ls].data.foreach_set(ATTRTYPE_FIELDS[type_key], AC(self.vertex_attributes[attr_name].reshape(-1)))
        if material is not None:
            new_object.data.materials.append(material)
        butil.put_in_collection(bpy.data.objects[name], butil.get_collection('terrain'))
        return new_object
    
    @property
    def vertex_normals(self):
        if self.normal_mode == NormalMode.Mean:
            mean_normals = trimesh.geometry.weighted_vertex_normals(len(self.vertices), self.faces, self.face_normals, np.ones((len(self.faces), 3)), use_loop=False)
            return mean_normals
        elif self.normal_mode == NormalMode.AngleWeighted:
            w_normals = trimesh.geometry.weighted_vertex_normals(len(self.vertices), self.faces, self.face_normals, self._trimesh.face_angles, use_loop=False)
            return w_normals
    
    def facewise_mean(self, attr):
        dll = load_cdll("terrain/lib/cpu/meshing/utils.so")
        facewise_mean = dll.facewise_mean
        facewise_mean.argtypes = [POINTER(c_double), POINTER(c_int32), c_int32, POINTER(c_double)]
        facewise_mean.restype = None
        result = AC(np.zeros(len(self.faces), dtype=np.float64))
        facewise_mean(ASDOUBLE(AC(attr.astype(np.float64))), ASINT(AC(self.faces.astype(np.int32))), len(self.faces), ASDOUBLE(result))
        return result
    
    def facewise_intmax(self, attr):
        dll = load_cdll("terrain/lib/cpu/meshing/utils.so")
        facewise_intmax = dll.facewise_intmax
        facewise_intmax.argtypes = [POINTER(c_int32), POINTER(c_int32), c_int32, POINTER(c_int32)]
        facewise_intmax.restype = None
        result = AC(np.zeros(len(self.faces), dtype=np.int32))
        facewise_intmax(ASINT(AC(attr.astype(np.int32))), ASINT(AC(self.faces.astype(np.int32))), len(self.faces), ASINT(result))
        return result

    def get_adjacency(self):
        dll = load_cdll("terrain/lib/cpu/meshing/utils.so")
        get_adjacency = dll.get_adjacency
        get_adjacency.argtypes = [c_int32, c_int32, POINTER(c_int32), POINTER(c_int32)]
        get_adjacency.restype = None
        result = AC(np.zeros((len(self.faces), 3), dtype=np.int32))
        pairs = self._trimesh.face_adjacency.astype(np.int32)
        get_adjacency(len(self.faces), len(pairs), ASINT(AC(pairs)), ASINT(result))
        return result

    @property
    def face_normals(self):
        dll = load_cdll(f"terrain/lib/cpu/meshing/utils.so")
        compute_face_normals = dll.compute_face_normals
        compute_face_normals.argtypes = [POINTER(c_double), POINTER(c_int32), c_int32, POINTER(c_double)]
        compute_face_normals.restype = None
        normals = AC(np.zeros((len(self.faces), 3), dtype=np.float64))
        compute_face_normals(ASDOUBLE(AC(self.vertices)), ASINT(AC(self.faces.astype(np.int32))), len(self.faces), ASDOUBLE(normals))
        return normals

    def cat(meshes):
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int)
        lenv = 0
        vertex_attributes = {}
        for mesh in meshes:
            verts = np.concatenate((verts, mesh.vertices), 0)
            faces = np.concatenate((faces, mesh.faces + lenv), 0)

            for attr in mesh.vertex_attributes:
                if mesh.vertex_attributes[attr].ndim == 1:
                    mesh.vertex_attributes[attr] = mesh.vertex_attributes[attr].reshape((-1, 1))
                mesh_va = mesh.vertex_attributes[attr]
                if attr not in vertex_attributes:
                    va = np.zeros((lenv, mesh.vertex_attributes[attr].shape[1]), dtype=mesh.vertex_attributes[attr].dtype)
                else:
                    va = vertex_attributes[attr]
                vertex_attributes[attr] = np.concatenate((va, mesh_va))
            lenv += len(mesh.vertices)
            
            for attr in vertex_attributes:
                if len(vertex_attributes[attr]) != lenv:
                    fillup = np.zeros((lenv - len(vertex_attributes[attr]), vertex_attributes[attr].shape[1]), dtype=vertex_attributes[attr].dtype)
                    vertex_attributes[attr] = np.concatenate((vertex_attributes[attr], fillup))
        return Mesh(vertices=verts, faces=faces, vertex_attributes=vertex_attributes)

    def camera_annotation(self, cameras, fs, fe, relax=0.01):
        cam_poses = []
        coords_trans_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        fc = bpy.context.scene.frame_current
        for f in range(fs, fe + 1):
            bpy.context.scene.frame_set(f)
            for cam in cameras:
                cam_pose = np.array(cam.matrix_world)
                cam_pose = np.dot(np.array(cam_pose), coords_trans_matrix)
                cam_poses.append(cam_pose)
                fov_rad  = cam.data.angle
        bpy.context.scene.frame_set(fc)
        
        H, W = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x
        fov0 = np.arctan(H / 2 / (W / 2 / np.tan(fov_rad / 2))) * 2
        fov = (fov0, fov_rad)
        K = getK(fov, H, W)
        
        self.vertex_attributes["invisible"] = np.zeros(len(self.vertices), bool)
        
        for cam_pose in cam_poses:
            coords = np.matmul(K, np.matmul(np.linalg.inv(cam_pose), np.concatenate((self.vertices.transpose(), np.ones((1, len(self.vertices)))), 0))[:3, :])
            coords[:2, :] /= coords[2]
            self.vertex_attributes["invisible"] |= ((coords[2] > 0) & (coords[0] > -relax * W) & (coords[0] < (1 + relax) * W) & (coords[1] > -relax * H) & (coords[1] < (1 + relax) * H))
        
        self.vertex_attributes["invisible"] = (~self.vertex_attributes["invisible"]).astype(np.float32)
        

def move_modifier(target_obj, m):
    with Timer(f"copying {m.name}"):
        modifier = target_obj.modifiers.new(m.name, "NODES")
        modifier.node_group = m.node_group
        for i, inp in enumerate(modifier.node_group.inputs):
            if i > 0:
                id = inp.identifier
                modifier[f'{id}_attribute_name'] = inp.name
                modifier[f'{id}_use_attribute'] = True
        for i, outp in enumerate(modifier.node_group.outputs):
            if i > 0:
                id = outp.identifier
                modifier[f'{id}_attribute_name'] = m[f'{id}_attribute_name']

def write_attributes(elements, mesh=None, meshes=[]):
    n_elements = len(elements)
    if mesh is not None:
        returns = []
        N = len(mesh.vertices)
        for element in elements:
            ret = element(mesh.vertices)
            returns.append(ret)
        surface_element = np.stack([ret[Vars.SDF] for ret in returns], -1).argmin(axis=-1)

        attributes = {}
        for i in range(n_elements):
            if hasattr(elements[i], "tag"):
                returns[i][Attributes.ElementTag] = np.zeros(N, dtype=np.int32) + elements[i].tag
            for output in returns[i]:
                if output == Vars.SDF or output == Vars.Offset: continue
                if returns[i][output].ndim == 1:
                    returns[i][output] *= (surface_element == i)
                else:
                    returns[i][output] *= (surface_element == i).reshape((-1, 1))

                if output not in attributes:
                    attributes[output] = returns[i][output]
                else:
                    attributes[output] += returns[i][output]
        mesh.vertex_attributes = attributes
    if meshes != []:
        assert(len(meshes) == n_elements)
        for i in range(n_elements):
            N = len(meshes[i].vertices)
            if N == 0: continue
            returns = elements[i](meshes[i].vertices)
            attributes = {}
            for output in returns:
                if output == Vars.SDF or output == Vars.Offset: continue
                attributes[output] = returns[output]
            if hasattr(elements[i], "tag"):
                attributes[Attributes.ElementTag] = np.zeros(N, dtype=np.int32) + elements[i].tag
            meshes[i].vertex_attributes = attributes

