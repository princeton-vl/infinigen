import pdb
import warnings

import bpy
import mathutils

import numpy as np
import gin

from util import blender as butil
from nodes.nodegroups import transfer_attributes

IS_COARSE = False # Global VARIABLE, set by generate.py and used only for whether to emit warnings

@gin.configurable
    if isinstance(obj, bpy.types.Object):
        if IS_COARSE:
            logger.warn(f'target_face_size({obj.name=}) is using the cameras location which is unsafe for {IS_COARSE=}')
        bbox = np.array([obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box])
        dists = np.linalg.norm(bbox - np.array(camera.location), axis=-1)
        eval_point = bbox[dists.argmin()]
    elif hasattr(obj, '__len__') and len(obj) == 3:
        if IS_COARSE:
            logger.warn(f'target_face_size({obj.name=}) is using the cameras location which is unsafe for {IS_COARSE=}')
        eval_point = mathutils.Vector(obj)
    elif isinstance(obj, (float, int)):
        dist = obj
    else:
        raise ValueError(f'target_face_size() could not handle {obj=}, {type(obj)=}')

    if camera is None:
        camera = bpy.context.scene.camera
    if camera is None:
    camd = camera.data

    scene = bpy.context.scene
    mm_to_meter = 0.001
    f_m = mm_to_meter * camd.lens
    sensor_dims = mm_to_meter * np.array([camd.sensor_width, camd.sensor_height])

    pixel_dims = (sensor_dims / pixel_shape) * (dist / f_m)

    res = min(pixel_dims)

    return np.clip(global_multiplier * res, global_clip_min, global_clip_max)

def remesh_with_attrs(obj, face_size, apply=True, min_remesh_size=None, attributes=None):

    remesh_size = face_size if min_remesh_size is None else max(face_size, min_remesh_size)

    bpy.data.objects.remove(temp_copy, do_unlink=True)

    if remesh_size > face_size:
        subdivide_to_face_size(obj, remesh_size, face_size, apply=True)

    return obj

    if to_facesize > from_facesize:
    logger.debug(f'subdivide_to_face_size applying {levels=} of subsurf to {obj.name=}')
    return mod # None if apply=True

def merged_by_distance_col(col, face_size, inplace=False):
    if not inplace:
        with butil.SelectObjects(list(col.objects)):
            bpy.ops.object.duplicate()

    for obj in col.objects:
        butil.merge_by_distance(obj, face_size)

    return col

def min_max_edgelen(mesh):
    verts = np.array([v.co for v in mesh.vertices])
    edges = np.array([e.vertices for e in mesh.edges])
    lens = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=-1)
    return lens.min(), lens.max()


def adapt_mesh_resolution(obj, face_size, method, approx=0.2, **kwargs):
    
    assert obj.type == 'MESH'
    assert 0 <= approx and approx <= 0.5

    logger.debug(f'adapt_mesh_resolution on {obj.name} with {method=} to {face_size=:.6f}')
        logger.debug(f'Ignoring adapt_mesh_resolution on {obj.name=} due to no polygons')
    lmin, lmax = min_max_edgelen(obj.data)

    if method == 'subdivide':
        if lmax > face_size:
            subdivide_to_face_size(obj, from_facesize=lmax, to_facesize=face_size, **kwargs)
        approx_facesize = np.sqrt(np.percentile(areas, q=1-approx))
        else:
            logger.debug(f'No subdivision necessary on {obj.name=} {approx_facesize} < {face_size}')
    elif method == 'merge_down':
        if lmin < face_size:
            butil.merge_by_distance(obj, face_size)
    elif method == 'remesh':
        remesh_with_attrs(obj, face_size, **kwargs)
    else:
