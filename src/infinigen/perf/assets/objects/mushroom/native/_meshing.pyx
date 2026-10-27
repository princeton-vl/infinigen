# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy
from libcpp.vector cimport vector
cnp.import_array()
cdef extern from "_meshing_core.hpp" namespace "mushroom_meshing":
 void mesh_edges(const int32_t*,const int32_t*,int,vector[int32_t]&,vector[int32_t]&) except + nogil
 int64_t triangle_count(const float*,int,double) except + nogil
 void revolve_edges(const float*,int,double,vector[float]&,vector[int32_t]&,vector[int32_t]&,vector[int32_t]&,vector[int32_t]&) except + nogil
 void revolve(const float*,int,double,vector[float]&,vector[int32_t]&,vector[int32_t]&) except + nogil

def profile_mesh(cnp.ndarray[cnp.float32_t,ndim=2,mode='c'] profile,double step):
 if profile.shape[1]!=3 or profile.shape[0]<2 or not np.isfinite(profile).all() or not np.isfinite(step) or step<=0:raise ValueError('Invalid profile')
 if np.abs(profile[:,1]).max()>1e-7:raise ValueError('Profile must lie in XZ plane')
 cdef vector[float] co
 cdef vector[int32_t] loops,counts
 with nogil:revolve(<float*>profile.data,profile.shape[0],step,co,loops,counts)
 cdef cnp.ndarray coords=np.empty((co.size()//3,3),dtype=np.float32)
 cdef cnp.ndarray indices=np.empty(loops.size(),dtype=np.int32)
 cdef cnp.ndarray sizes=np.empty(counts.size(),dtype=np.int32)
 if co.size():memcpy(coords.data,co.data(),co.size()*sizeof(float))
 if loops.size():memcpy(indices.data,loops.data(),loops.size()*sizeof(int32_t))
 if counts.size():memcpy(sizes.data,counts.data(),counts.size()*sizeof(int32_t))
 return coords,indices,sizes

def profile_triangle_count(cnp.ndarray[cnp.float32_t,ndim=2,mode='c'] profile,double step):
 if profile.shape[1]!=3 or profile.shape[0]<2 or not np.isfinite(profile).all() or not np.isfinite(step) or step<=0:raise ValueError('Invalid profile')
 if np.abs(profile[:,1]).max()>1e-7:raise ValueError('Profile must lie in XZ plane')
 cdef int64_t count
 with nogil:count=triangle_count(<float*>profile.data,profile.shape[0],step)
 return count


def edge_buffers(cnp.ndarray[cnp.int32_t,ndim=1,mode='c'] loops,cnp.ndarray[cnp.int32_t,ndim=1,mode='c'] counts):
 if (counts<3).any() or counts.sum(dtype=np.int64)!=loops.size or (loops<0).any():raise ValueError('Invalid face buffers')
 cdef vector[int32_t] edges,corners
 with nogil:mesh_edges(<int32_t*>loops.data,<int32_t*>counts.data,counts.size,edges,corners)
 cdef cnp.ndarray e=np.empty((edges.size()//2,2),np.int32)
 cdef cnp.ndarray c=np.empty(corners.size(),np.int32)
 if edges.size():memcpy(e.data,edges.data(),edges.size()*sizeof(int32_t))
 if corners.size():memcpy(c.data,corners.data(),corners.size()*sizeof(int32_t))
 return e,c

def profile_mesh_with_edges(cnp.ndarray[cnp.float32_t,ndim=2,mode='c'] profile,double step):
 if profile.shape[1]!=3 or profile.shape[0]<2 or not np.isfinite(profile).all() or not np.isfinite(step) or step<=0:raise ValueError('Invalid profile')
 if np.abs(profile[:,1]).max()>1e-7:raise ValueError('Profile must lie in XZ plane')
 cdef vector[float] co
 cdef vector[int32_t] loops,counts,edges,corners
 with nogil:revolve_edges(<float*>profile.data,profile.shape[0],step,co,loops,counts,edges,corners)
 cdef cnp.ndarray coords=np.empty((co.size()//3,3),dtype=np.float32)
 cdef cnp.ndarray indices=np.empty(loops.size(),dtype=np.int32)
 cdef cnp.ndarray sizes=np.empty(counts.size(),dtype=np.int32)
 if co.size():memcpy(coords.data,co.data(),co.size()*sizeof(float))
 if loops.size():memcpy(indices.data,loops.data(),loops.size()*sizeof(int32_t))
 if counts.size():memcpy(sizes.data,counts.data(),counts.size()*sizeof(int32_t))
 cdef cnp.ndarray e=np.empty((edges.size()//2,2),np.int32)
 cdef cnp.ndarray c=np.empty(corners.size(),np.int32)
 if edges.size():memcpy(e.data,edges.data(),edges.size()*sizeof(int32_t))
 if corners.size():memcpy(c.data,corners.data(),corners.size()*sizeof(int32_t))
 return coords,indices,sizes,e,c
