# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_float, c_int32, c_size_t

import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.terrain.utils import KernelDataType, Vars, load_cdll, register_func, ASFLOAT, ASINT, KERNELDATATYPE_DIMS, KERNELDATATYPE_NPTYPE, Mesh

from .kernelizer import Kernelizer


class SurfaceKernel:
    def __init__(self, name, attribute, modifier, device):
        self.name = name
        self.attribute = attribute
        self.device = device
        _, inputs, outputs = Kernelizer()(modifier)
        dll = load_cdll(f"terrain/lib/{self.device}/surfaces/{name}.so")
        call_param_type = [c_size_t]
        self.use_position = Vars.Position in inputs
        if self.use_position:
            call_param_type.append(POINTER(c_float))
        self.use_normal = Vars.Normal in inputs
        if self.use_normal:
            call_param_type.append(POINTER(c_float))
        imp_values_of_type = {}
        for var in sorted(inputs.keys()):
            if var in [Vars.Position, Vars.Normal]: continue
            dtype, value = inputs[var]
            if dtype in imp_values_of_type:
                imp_values_of_type[dtype].append(value)
            else:
                imp_values_of_type[dtype] = [value]
        for dtype in sorted(imp_values_of_type.keys()):
            call_param_type.append(c_size_t)
            if dtype != KernelDataType.int:
                imp_values_of_type[dtype] = np.concatenate(imp_values_of_type[dtype])
                call_param_type.append(POINTER(c_float))
            else:
                imp_values_of_type[dtype] = np.concatenate(imp_values_of_type[dtype])
                call_param_type.append(POINTER(c_int32))
        for var in outputs:
            dtype = outputs[var]
            if dtype != KernelDataType.int:
                call_param_type.append(POINTER(c_float))
            else:
                call_param_type.append(POINTER(c_int32))
        self.outputs = outputs
        register_func(self, dll, "call", call_param_type)
        self.imp_values_of_type = imp_values_of_type

    def __call__(self, params):
        ret = {}
        values = []
        for dtype in sorted(self.imp_values_of_type.keys()):
            M = len(self.imp_values_of_type[dtype]) // int(np.product(KERNELDATATYPE_DIMS[dtype]))
            values.append(M)
            if dtype != KernelDataType.int:
                values.append(ASFLOAT(self.imp_values_of_type[dtype]))
            else:
                values.append(ASINT(self.imp_values_of_type[dtype]))
        if isinstance(params, dict):
            positions = AC(params[Vars.Position].astype(np.float32))
            N = len(positions)
        elif isinstance(params, Mesh):
            positions = AC(params.vertices.astype(np.float32))
            N = len(positions)
        for var in self.outputs:
            dtype = self.outputs[var]
            ret[var] = AC(np.zeros((N, *KERNELDATATYPE_DIMS[dtype]), dtype=KERNELDATATYPE_NPTYPE[dtype]))
            if dtype != KernelDataType.int:
                values.append(ASFLOAT(ret[var]))
            else:
                values.append(ASINT(ret[var]))
        if isinstance(params, dict):
            normals = AC(np.concatenate((np.zeros((N, 2), dtype=np.float32), np.ones((N, 1), dtype=np.float32)), -1))
            pvalues = [N]
            if self.use_position: pvalues.append(ASFLOAT(positions))
            if self.use_normal: pvalues.append(ASFLOAT(normals))
            values = pvalues + values
            self.call(*values)

            for var in self.outputs:
                dtype = self.outputs[var]
                shape = [1] * len(KERNELDATATYPE_DIMS[dtype])
                ret[var] *= params[self.attribute].reshape((N, *shape))
                if var == Vars.Offset:
                    ret[var] = ret[var][:, 2]
            return ret
        elif isinstance(params, Mesh):
            normals = AC(params.vertex_normals.astype(np.float32))
            pvalues = [N]
            if self.use_position: pvalues.append(ASFLOAT(positions))
            if self.use_normal: pvalues.append(ASFLOAT(normals))
            values = pvalues + values
            self.call(*values)
            for var in self.outputs:
                dtype = self.outputs[var]
                shape = [1] * len(KERNELDATATYPE_DIMS[dtype])
                ret[var] *= params.vertex_attributes[self.attribute].reshape((N, *shape))
                if var == Vars.Offset:
                    params.vertices += ret[var]
                else:
                    params.vertex_attributes[var] = ret[var]
                