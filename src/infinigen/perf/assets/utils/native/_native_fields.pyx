"""Compiled field evaluation for the optimized leaves and tree bark."""
import numpy as np
cimport numpy as cnp
from libc.stddef cimport size_t

cdef extern from "native_fields.h" nogil:
    void broadleaf_fields(const float*, const unsigned char*, size_t, const float*, const float*, int, const float*, float*)
    void native_ops(const float*, size_t, int, float*)
    void planar_leaf_fields(const float*, size_t, int, const float*, const float*, int, const float*, float*, int, float*, const unsigned char*)
    void planar_leaf_coverage(const float*, size_t, const float*, size_t, int, const float*, const float*, int, const float*, float, unsigned char*)
    void planar_leaf_bend(const float*, size_t, int, const float*, const float*, int, const float*, const float*, float*)
    void bark_displacement(const float*, const float*, const float*, size_t, const float*, float*)

def evaluate_planar_leaf(xy, int kind, parameters, tables, ranges, bint defer_bend=False, coverage_mask=None):
    """Compiled V2/Ginkgo/Maple planar leaf formulas (`kind` 0/1/2) at `xy`."""
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] coords=np.ascontiguousarray(xy,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] params=np.ascontiguousarray(parameters,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] lut=np.ascontiguousarray(tables,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] domains=np.ascontiguousarray(ranges,dtype=np.float32)
    if kind<0 or kind>2:raise ValueError("unknown planar formula")
    nc=(5,4,7)[kind]
    if coords.shape[1]!=2 or params.shape[0]!=(8,6,8)[kind] or lut.shape[0]!=nc or lut.shape[1]<2 or domains.shape[0]!=nc or domains.shape[1]!=2:
        raise ValueError("invalid coordinate, parameter or curve table shape")
    if not all(np.isfinite(a).all() for a in (coords,params,lut,domains)) or np.any(domains[:,1]<=domains[:,0]):
        raise ValueError("expected finite inputs and increasing curve domains")
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result=np.empty((coords.shape[0],6),dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] bounds=np.empty(5,dtype=np.float32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] coverage
    cdef const unsigned char* maskptr=NULL
    if coverage_mask is not None:
        coverage=np.ascontiguousarray(coverage_mask,dtype=np.uint8).reshape(-1)
        if coverage.shape[0]!=coords.shape[0]:raise ValueError("coverage must match coordinates")
        maskptr=<const unsigned char*>coverage.data
    with nogil:
        planar_leaf_fields(<const float*>coords.data,coords.shape[0],kind,<const float*>params.data,<const float*>lut.data,lut.shape[1],<const float*>domains.data,<float*>result.data,defer_bend,<float*>bounds.data,maskptr)
    if defer_bend:return result,bounds
    return result

def planar_coverage(x, y, int kind, parameters, tables, ranges, float reach=0):
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] xs=np.ascontiguousarray(x,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] ys=np.ascontiguousarray(y,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] params=np.ascontiguousarray(parameters,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] lut=np.ascontiguousarray(tables,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] domains=np.ascontiguousarray(ranges,dtype=np.float32)
    if kind<0 or kind>2:raise ValueError("unknown planar formula")
    nc=(5,4,7)[kind]
    if params.shape[0]!=(8,6,8)[kind] or lut.shape[0]!=nc or lut.shape[1]<2 or domains.shape[0]!=nc or domains.shape[1]!=2 or not np.isfinite(reach) or reach<0:
        raise ValueError("invalid coverage input shape")
    if not all(np.isfinite(a).all() for a in (xs,ys,params,lut,domains)) or np.any(domains[:,1]<=domains[:,0]):
        raise ValueError("invalid coverage input values")
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] result=np.empty((ys.shape[0],xs.shape[0]),dtype=np.uint8)
    with nogil:
        planar_leaf_coverage(<const float*>xs.data,xs.shape[0],<const float*>ys.data,ys.shape[0],kind,<const float*>params.data,<const float*>lut.data,lut.shape[1],<const float*>domains.data,reach,<unsigned char*>result.data)
    return result

def bend_planar_leaf(xyz, int kind, parameters, tables, ranges, bounds):
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] coords=np.ascontiguousarray(xyz,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] params=np.ascontiguousarray(parameters,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] lut=np.ascontiguousarray(tables,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] domains=np.ascontiguousarray(ranges,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] box=np.ascontiguousarray(bounds,dtype=np.float32)
    if kind<0 or kind>2:raise ValueError("unknown planar formula")
    nc=(5,4,7)[kind]
    if coords.shape[1]!=3 or params.shape[0]!=(8,6,8)[kind] or lut.shape[0]!=nc or lut.shape[1]<2 or domains.shape[0]!=nc or domains.shape[1]!=2 or box.shape[0]!=5:
        raise ValueError("invalid bend input shape")
    if not all(np.isfinite(a).all() for a in (coords,params,lut,domains,box)) or np.any(domains[:,1]<=domains[:,0]) or box[1]<box[0] or box[3]<box[2]:
        raise ValueError("invalid bend input values")
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result=np.empty((coords.shape[0],3),dtype=np.float32)
    with nogil:
        planar_leaf_bend(<const float*>coords.data,coords.shape[0],kind,<const float*>params.data,<const float*>lut.data,lut.shape[1],<const float*>domains.data,<const float*>box.data,<float*>result.data)
    return result

def evaluate_broadleaf(xy, parameters, tables, ranges, mask=None):
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] coords=np.ascontiguousarray(xy,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] params=np.ascontiguousarray(parameters,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] lut=np.ascontiguousarray(tables,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] domains=np.ascontiguousarray(ranges,dtype=np.float32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] coverage
    cdef const unsigned char* maskptr=NULL
    if coords.shape[1]!=2 or params.shape[0]!=11 or lut.shape[0]!=8 or lut.shape[1]<2 or domains.shape[0]!=8 or domains.shape[1]!=2:
        raise ValueError("invalid coordinate, parameter or curve table shape")
    if not all(np.isfinite(a).all() for a in (coords,params,lut,domains)):
        raise ValueError("field inputs must be finite")
    if np.any(domains[:,1]<=domains[:,0]):
        raise ValueError("curve domains must be increasing")
    if mask is not None:
        coverage=np.ascontiguousarray(mask,dtype=np.uint8).reshape(-1)
        if coverage.shape[0]!=coords.shape[0]:raise ValueError("mask must match coordinates")
        maskptr=<const unsigned char*>coverage.data
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result=np.empty((coords.shape[0],8),dtype=np.float32)
    with nogil:
        broadleaf_fields(<const float*>coords.data,maskptr,coords.shape[0],<const float*>params.data,<const float*>lut.data,lut.shape[1],<const float*>domains.data,<float*>result.data)
    return result

def evaluate_operation(xyz, int operation):
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] coords=np.ascontiguousarray(xyz,dtype=np.float32)
    if coords.shape[1]!=3 or not np.isfinite(coords).all() or operation<0 or operation>3:
        raise ValueError("expected finite Nx3 coordinates and operation 0..3")
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] result=np.empty(coords.shape[0],dtype=np.float32)
    with nogil:
        native_ops(<const float*>coords.data,coords.shape[0],operation,<float*>result.data)
    return result

def evaluate_bark_displacement(positions, parent_locations, skeleton_locations, parameters):
    """Evaluate the BarkRandom scalar normal displacement at arbitrary points."""
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] coords=np.ascontiguousarray(positions,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] parents=np.ascontiguousarray(parent_locations,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] skeletons=np.ascontiguousarray(skeleton_locations,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] params=np.ascontiguousarray(parameters,dtype=np.float32)
    if coords.ndim!=2 or coords.shape[1]!=3 or parents.ndim!=2 or parents.shape[1]!=3 or skeletons.ndim!=2 or skeletons.shape[1]!=3:
        raise ValueError("positions and bark attributes must be Nx3")
    if parents.shape[0]!=coords.shape[0] or skeletons.shape[0]!=coords.shape[0]:
        raise ValueError("bark attributes must match positions")
    if params.shape[0]!=11:
        raise ValueError("expected 11 BarkRandom parameters")
    if not all(np.isfinite(a).all() for a in (coords,parents,skeletons,params)):
        raise ValueError("bark field inputs must be finite")
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] result=np.empty(coords.shape[0],dtype=np.float32)
    with nogil:
        bark_displacement(<const float*>coords.data,<const float*>parents.data,<const float*>skeletons.data,coords.shape[0],<const float*>params.data,<float*>result.data)
    return result
