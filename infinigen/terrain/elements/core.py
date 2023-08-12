# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_float, c_int32, c_size_t

import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.utils import ASFLOAT, ASINT, Vars, load_cdll, register_func
from infinigen.core.util.organization import Materials


class Element:
    called_time = {}
    def __init__(self, lib_name, material, transparency):
        if lib_name in Element.called_time:
            lib_name_X = f"{lib_name}_{Element.called_time[lib_name]}"
            print(f"{lib_name} already loaded, loading {lib_name_X} instead")
            Element.called_time[lib_name] += 1
        else:
            lib_name_X = lib_name
            Element.called_time[lib_name] = 1
        dll = load_cdll(f"terrain/lib/{self.device}/elements/{lib_name_X}.so")
        call_param_type = [c_size_t, POINTER(c_float), POINTER(c_float)]
        self.attributes = [material]

        if hasattr(self, "aux_names") and self.aux_names != []:
            for aux_name in self.aux_names:
                if aux_name in Materials.all:
                    self.attributes.append(aux_name)
            call_param_type.append(POINTER(c_float))
        register_func(self, dll, "call", call_param_type)
        register_func(self, dll, "init", [
            c_int32, c_int32,
            c_size_t, POINTER(c_int32), c_size_t, POINTER(c_float),
            c_size_t, POINTER(c_int32), c_size_t, POINTER(c_float),
            c_size_t, POINTER(c_int32), c_size_t, POINTER(c_float),
        ])
        register_func(self, dll, "cleanup")

        self.material = material
        self.transparency = transparency

        if hasattr(self, "meta_params"):
            meta_param = self.meta_params[0]
            if len(self.meta_params) > 1:
                meta_param2 = self.meta_params[1]
            else:
                meta_param2 = 0
        else:
            meta_param = meta_param2 = 0
        if not hasattr(self, "int_params2"): self.int_params2 = np.zeros(0, dtype=np.int32)
        if not hasattr(self, "float_params2"): self.float_params2 = np.zeros(0, dtype=np.float32)
        if not hasattr(self, "int_params3"): self.int_params3 = np.zeros(0, dtype=np.int32)
        if not hasattr(self, "float_params3"): self.float_params3 = np.zeros(0, dtype=np.float32)
        self.init(
            meta_param, meta_param2,
            len(self.int_params), ASINT(self.int_params), len(self.float_params), ASFLOAT(self.float_params),
            len(self.int_params2), ASINT(self.int_params2), len(self.float_params2), ASFLOAT(self.float_params2),
            len(self.int_params3), ASINT(self.int_params3), len(self.float_params3), ASFLOAT(self.float_params3),
        )
        self.displacement = []


    def __call__(self, positions, sdf_only=False):
        N = len(positions)
        sdf = AC(np.zeros(N, dtype=np.float32))
        auxs = []
        flag = False
        if hasattr(self, "aux_names"):
            if not sdf_only or self.displacement != []:
                flag = True
                auxs.append(AC(np.zeros(N * len(self.aux_names), dtype=np.float32)))
            else:
                auxs.append(None)
        self.call(N, ASFLOAT(AC(positions.astype(np.float32))), ASFLOAT(sdf), *[POINTER(c_float)() if x is None else ASFLOAT(x) for x in auxs])
        ret = {}
        ret[Vars.SDF] = sdf

        if flag:
            aux = auxs[0].reshape((N, len(self.aux_names)))
            for i, aux_name in enumerate(self.aux_names):
                if aux_name is not None:
                    ret[aux_name] = aux[:, i]
        if not sdf_only or self.displacement != []:
            ret[self.material] = np.ones(N, dtype=np.float32)
            for aux_name in ret:
                if aux_name != self.material and aux_name in Materials.all:
                    ret[self.material] -= ret[aux_name]
        for surface in self.displacement:
            ret.update(surface({Vars.Position: positions, **ret}))
            ret[Vars.SDF] -= ret.pop(Vars.Offset)
        return ret

    def get_heightmap(self, X, Y):
        N = X.shape[0]
        positions = np.stack((X.reshape(-1), Y.reshape(-1), np.zeros(N * N)), -1)
        return -self.__call__(positions)[Vars.SDF].reshape((N, N))