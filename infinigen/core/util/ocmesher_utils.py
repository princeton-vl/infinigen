# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from ctypes import POINTER, c_float, c_int32

import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.core.nodes.nodegroups import transfer_attributes
from infinigen.core.util import blender as butil
from infinigen.OcMesher.ocmesher import OcMesher
from infinigen.terrain.utils import ASFLOAT, ASINT, Mesh, get_caminfo, load_cdll


def create_sdf_from_mesh(mesh):
    dll = load_cdll("terrain/lib/cpu/sdf_from_mesh/sdf_from_mesh.so")
    sdf_from_mesh = dll.call
    sdf_from_mesh.argtypes = [
        c_int32,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_int32),
        c_int32,
    ]
    sdf_from_mesh.restype = None
    vertices = AC(mesh.vertices.astype(np.float32))
    faces = AC(mesh.faces.astype(np.int32))

    def func(XYZ):
        n = len(XYZ)
        sdf = np.zeros(n, dtype=np.float32)
        sdf_from_mesh(
            n,
            ASFLOAT(AC(XYZ.astype(np.float32))),
            ASFLOAT(sdf),
            ASFLOAT(vertices),
            ASINT(faces),
            len(faces),
        )
        return sdf

    return func


def run_ocmesher(obj, cameras):
    mesh = Mesh(obj=obj)
    room_sdf_func = create_sdf_from_mesh(mesh)
    xyz_min, xyz_max = butil.bounds(obj)
    xyz_size = xyz_max - xyz_min
    xyz_min -= xyz_size * 0.01
    xyz_max += xyz_size * 0.01
    bounds = (xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2])
    mesher = OcMesher(get_caminfo(cameras)[0], bounds, pixels_per_cube=3)
    mesh = Mesh(mesh=mesher([room_sdf_func], structure_mesh=mesh)[0][0])
    ocmesh = mesh.export_blender(obj.name + ".ocmesher")
    transfer_attributes.transfer_all(source=obj, target=ocmesh, uvs=True)
    return ocmesh
