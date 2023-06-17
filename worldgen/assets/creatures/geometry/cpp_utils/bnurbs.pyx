import numpy as np
import cython
cimport numpy as np

import bpy 

# IMPORTANT: The structs below are copied from DNA_curve_types.h of Blender 3.1.2 source  
# May not work for versions of Blender

ctypedef unsigned char uint8_t

#/**
# * Keyframes on F-Curves (allows code reuse of Bezier eval code) and
# * Points on Bezier Curves/Paths are generally BezTriples.
# *
# * \note #BezTriple.tilt location in struct is abused by Key system.
# *
# * \note vec in BezTriple looks like this:
# * - vec[0][0] = x location of handle 1
# * - vec[0][1] = y location of handle 1
# * - vec[0][2] = z location of handle 1 (not used for FCurve Points(2d))
# * - vec[1][0] = x location of control point
# * - vec[1][1] = y location of control point
# * - vec[1][2] = z location of control point
# * - vec[2][0] = x location of handle 2
# * - vec[2][1] = y location of handle 2
# * - vec[2][2] = z location of handle 2 (not used for FCurve Points(2d))
# */
ctypedef struct BezTriple: 
  float vec[3][3]
  #/** Tilt in 3D View. */
  float tilt
  #/** Used for softbody goal weight. */
  float weight
  #/** For bevel tapering & modifiers. */
  float radius

  #/** Ipo: interpolation mode for segment from this BezTriple to the next. */
  char ipo

  #/** H1, h2: the handle type of the two handles. */
  uint8_t h1, h2
  #/** F1, f2, f3: used for selection status. */
  uint8_t f1, f2, f3

  #/** Hide: used to indicate whether BezTriple is hidden (3D),
  # * type of keyframe (eBezTriple_KeyframeType). */
  char hide

  #/** Easing: easing type for interpolation mode (eBezTriple_Easing). */
  char easing
  #/** BEZT_IPO_BACK. */
  float back
  #/** BEZT_IPO_ELASTIC. */
  float amplitude, period

  #/** Used during auto handle calculation to mark special cases (local extremes). */
  char auto_handle_type
  char _pad[3]
 

#/**
# * \note #BPoint.tilt location in struct is abused by Key system.
# */
ctypedef struct BPoint: 
  float vec[4]
  #/** Tilt in 3D View. */
  float tilt
  #/** Used for softbody goal weight. */
  float weight
  #/** F1: selection status,  hide: is point hidden or not. */
  uint8_t f1
  char _pad1[1]
  short hide
  #/** User-set radius per point for beveling etc. */
  float radius
  char _pad[4]

ctypedef struct Nurb
#/**
# * \note Nurb name is misleading, since it can be used for polygons too,
# * also, it should be NURBS (Nurb isn't the singular of Nurbs).
# */
ctypedef struct Nurb: 
  #/** Multiple nurbs per curve object are allowed. */
  Nurb *next, *prev
  short type
  #/** Index into material list. */
  short mat_nr
  short hide, flag
  #/** Number of points in the U or V directions. */
  int pntsu, pntsv
  char _pad[4]
  #/** Tessellation resolution in the U or V directions. */
  short resolu, resolv
  short orderu, orderv
  short flagu, flagv

  float *knotsu, *knotsv
  BPoint *bp
  BezTriple *bezt

  #/** KEY_LINEAR, KEY_CARDINAL, KEY_BSPLINE. */
  short tilt_interp
  short radius_interp

  #/* only used for dynamically generated Nurbs created from OB_FONT's */
  int charidx

cdef int cal_knot_count(int pnt_count, int order, bint is_cyclic):
    #  from blender source: const int knot_count = pnts + order + (is_cyclic ? order - 1 : 0);
    return pnt_count + order + (order - 1 if is_cyclic else 0)

cdef Nurb* get_nurb(spline: bpy.types.Spline):
    s = spline
    cdef unsigned long long ptr = s.as_pointer()
    cdef Nurb* nu = <Nurb*>ptr 
    return nu

def get_knotsu(spline: bpy.types.Spline):
    s = spline
    cdef Nurb* nu = get_nurb(spline)
    cdef int knot_cnt = cal_knot_count(s.point_count_u, s.order_u, s.use_cyclic_u) 
    res = np.zeros(knot_cnt,dtype=np.double) 
    cdef double[:] res_view = res
    for i in range(knot_cnt):
        res_view[i] = nu.knotsu[i]
    return res

def get_knotsv(spline: bpy.types.Spline):
    s = spline
    cdef Nurb* nu = get_nurb(spline)
    cdef int knot_cnt = cal_knot_count(s.point_count_v, s.order_v, s.use_cyclic_v) 
    res = np.zeros(knot_cnt,dtype=np.double) 
    cdef double[:] res_view = res
    for i in range(knot_cnt):
        res_view[i] = nu.knotsv[i]
    return res

def set_knotsu(spline: bpy.types.Spline, knots, redraw_in_UI=True):
    s = spline
    cdef Nurb* nu = get_nurb(spline)
    cdef int knot_cnt = cal_knot_count(s.point_count_u, s.order_u, s.use_cyclic_u) 
    for i in range(knot_cnt):
        nu.knotsu[i] = knots[i]

    # To trigger redrawing the shape in blender UI
    if redraw_in_UI:
        spline.resolution_u = spline.resolution_u

def set_knotsv(spline: bpy.types.Spline, knots, redraw_in_UI=True):
    s = spline
    cdef Nurb* nu = get_nurb(spline)
    cdef int knot_cnt = cal_knot_count(s.point_count_v, s.order_v, s.use_cyclic_v) 
    for i in range(knot_cnt):
        nu.knotsv[i] = knots[i]

    # To trigger redrawing the shape in blender UI
    if redraw_in_UI:
        spline.resolution_v = spline.resolution_v
