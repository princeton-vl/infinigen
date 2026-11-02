# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t
cnp.import_array()
cdef extern from "_adaptive_core.hpp":
 void* av_create_mesh(const float*,const int32_t*,int,const double*,double,double) nogil
 void av_free(void*) nogil
 int av_clear(void*,void*,const double*,double) nogil
 void av_stats(void*,long*) nogil
cdef class SurfaceTree:
 cdef void* pointer
 def __cinit__(self,cnp.ndarray[cnp.float32_t,ndim=2,mode='c'] co,cnp.ndarray[cnp.int32_t,ndim=2,mode='c'] triangles,cnp.ndarray[cnp.float64_t,ndim=2,mode='c'] matrix,double height,double cell=.004):
  self.pointer=NULL
  if co.shape[1]!=3 or triangles.shape[1]!=3 or matrix.shape[0]!=3 or matrix.shape[1]!=3:raise ValueError('Invalid mesh shapes')
  if not np.isfinite(co).all() or not np.isfinite(matrix).all() or not np.isfinite(height) or not np.isfinite(cell) or cell<=0:raise ValueError('Invalid geometry')
  if triangles.size and (triangles.min()<0 or triangles.max()>=co.shape[0]):raise ValueError('Invalid indices')
  with nogil:self.pointer=av_create_mesh(<float*>co.data,<int32_t*>triangles.data,triangles.shape[0],<double*>matrix.data,height,cell)
  if self.pointer==NULL:raise MemoryError('Unable to construct surface tree')
 def close(self):
  if self.pointer!=NULL:av_free(self.pointer);self.pointer=NULL
 def __dealloc__(self):
  if self.pointer!=NULL:av_free(self.pointer)
 def clear(self,SurfaceTree other,offset,double threshold=.008):
  cdef cnp.ndarray[cnp.float64_t,ndim=1,mode='c'] delta=np.ascontiguousarray(offset,dtype=np.float64)
  if self.pointer==NULL or other.pointer==NULL:raise RuntimeError('Tree is closed')
  if delta.shape[0]!=3 or not np.isfinite(delta).all() or not np.isfinite(threshold) or threshold<0:raise ValueError('Invalid clearance query')
  cdef int result
  with nogil:result=av_clear(self.pointer,other.pointer,<double*>delta.data,threshold)
  if result<0:raise RuntimeError('Clearance query failed')
  return bool(result)
 def stats(self):
  if self.pointer==NULL:raise RuntimeError('Tree is closed')
  cdef cnp.ndarray[cnp.int64_t,ndim=1,mode='c'] result=np.zeros(5,dtype=np.int64)
  av_stats(self.pointer,<long*>result.data)
  return dict(zip(['nodes','splits','triangle_tests','node_pairs','triangles'],map(int,result)))
