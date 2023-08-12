# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import sys
from ctypes import CDLL, POINTER, c_double, c_float, c_int32, RTLD_LOCAL
from pathlib import Path


# note: size of x should not exceed maximum
def ASINT(x):
    return x.ctypes.data_as(POINTER(c_int32))
def ASDOUBLE(x):
    return x.ctypes.data_as(POINTER(c_double))
def ASFLOAT(x):
    return x.ctypes.data_as(POINTER(c_float))

def register_func(me, dll, name, argtypes=[], restype=None, caller_name=None):
    if caller_name is None: caller_name = name
    setattr(me, caller_name, getattr(dll, name))
    func = getattr(me, caller_name)
    func.argtypes = argtypes
    func.restype = restype

def load_cdll(path):
    root = Path(__file__).parent.parent.parent
    return CDLL(root/path, mode=RTLD_LOCAL)
