# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell


import bpy
import numpy as np

from . import helper

from mathutils import Vector, Matrix

C = bpy.context
D = bpy.data


def init_mesh(name, verts=[], edges=[], faces=[], coll=None):  
  mesh = D.meshes.new(name)
  obj = D.objects.new(mesh.name, mesh)

  if coll is None:
    coll = bpy.context.scene.collection
  else:
    coll = D.collections[coll]
  
  coll.objects.link(obj)
  helper.set_active_obj(obj)

  mesh.from_pydata(verts, edges, faces)

  return obj


def duplicate_obj(obj, name):
  new_obj = obj.copy()
  new_obj.name = name
  new_obj.data = new_obj.data.copy()

  col = obj.users_collection[0]
  col.objects.link(new_obj)
  
  helper.set_active_obj(new_obj)
  return new_obj


def finalize_obj(obj):
  helper.set_active_obj(obj)
  bpy.ops.object.convert(target='MESH')


def init_vertex(pos):
  bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=True, align='WORLD',
                                  location=pos, scale=(1, 1, 1))
  bpy.ops.mesh.merge(type='COLLAPSE')
  bpy.ops.object.editmode_toggle()

  return C.active_object


def get_all_vtx_pos(obj):
  n_cds = len(obj.data.vertices)
  all_cds = np.zeros(n_cds * 3)
  obj.data.vertices.foreach_get('co', all_cds)
  return all_cds.reshape(-1, 3)


def vtx2cds(vtxs, world_mat):
  n_cds = len(vtxs)
  all_cds = np.zeros(n_cds * 3)
  vtxs.foreach_get('co', all_cds)
  all_cds = all_cds.reshape(-1,3)
  all_cds = add_ones(all_cds.reshape(-1, 3))
  m_world = np.array(world_mat)
  all_cds = np.matmul(m_world, all_cds.T).T[:,:3]

  return all_cds


def sample_vtxs(obj, emit_from='VOLUME', n=1000, seed=1):
  # Make object current active object
  bpy.ops.object.mode_set(mode='OBJECT')
  C.view_layer.objects.active = obj

  # Add particle system modifier
  bpy.ops.object.modifier_add(type='PARTICLE_SYSTEM')
  p = D.particles[-1]

  # Adjust modifier settings
  p.count = n
  p.frame_end = 1
  p.emit_from = emit_from
  p.distribution = 'RAND'
  p.use_modifier_stack = True
  p.physics_type = 'NO'
  obj.particle_systems[-1].seed = seed

  # Get particle locations (relative to object)
  obj_eval = obj.evaluated_get(C.evaluated_depsgraph_get())
  all_cds = np.zeros(n * 3)
  obj_eval.particle_systems[-1].particles.foreach_get('location', all_cds)

  obj.modifiers.remove(obj.modifiers[-1])
  D.particles.remove(D.particles[-1])

  return all_cds.reshape(-1,3)


def get_pts_from_shape(shape_fn, n=10, emit_from="VOLUME", loc=(0,0,0),
                       scaling=1, pt_offset=0):
    if isinstance(pt_offset, list):
        pt_offset = np.array([pt_offset])
    if isinstance(scaling, list):
        scaling = Vector(scaling)
    shape_fn(location=loc)
    obj = C.active_object
    obj.scale *= scaling
    pts = sample_vtxs(obj, n=n, emit_from=emit_from, seed=np.random.randint(100))
    pts += pt_offset
    D.objects.remove(obj)
    return pts


def select_vtx_by_pos(obj, pos):
  bpy.ops.object.mode_set(mode = 'EDIT')
  bpy.ops.mesh.select_mode(type="VERT")
  bpy.ops.mesh.select_all(action = 'DESELECT')
  bpy.ops.object.mode_set(mode = 'OBJECT')
  n_cds = len(obj.data.vertices)
  all_cds = np.zeros(n_cds * 3)
  obj.data.vertices.foreach_get('co', all_cds)
  idx = np.abs(all_cds.reshape(n_cds, 3) - pos).sum(1).argmin()
  obj.data.vertices[idx].select = True
  bpy.ops.object.mode_set(mode = 'EDIT')

  return idx


def select_vtx_by_idx(obj, idx, deselect=False):
  if not isinstance(idx, list):
    idx = [idx]
  bpy.ops.object.mode_set(mode = 'EDIT')
  bpy.ops.mesh.select_mode(type="VERT")
  if deselect:
    bpy.ops.mesh.select_all(action = 'DESELECT')
  bpy.ops.object.mode_set(mode = 'OBJECT')
  for i in idx:
    obj.data.vertices[i].select = True
  bpy.ops.object.mode_set(mode = 'EDIT')

  return idx


def extrude_path(obj, path):
  helper.set_active_obj(obj)
  bpy.ops.object.mode_set(mode='EDIT')
  src_idx = select_vtx_by_pos(obj, path[0])
  deltas = path[1:] - path[:-1]
  start_idx = len(obj.data.vertices)
  for i in range(len(deltas)):
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":deltas[i]})

  return src_idx, start_idx


def get_vtx_obj():
  if not 'vtx' in D.objects:
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD',
                                    location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.merge(type='COLLAPSE')
    bpy.ops.object.editmode_toggle()
    obj = C.active_object
    obj.name = 'vtx'

  return D.objects['vtx']


def subsample_vertices(v, max_num=500):
  if len(v) > max_num:
    rand_order = np.random.permutation(len(v))
    return np.sort(rand_order[:max_num])
  else:
    return np.arange(len(v))


def add_ones(x):
  return np.concatenate([x, np.ones_like(x[:,:1])], 1)


def get_world_coords(obj, subset=None):
  dgraph = C.evaluated_depsgraph_get()
  obj_eval = obj.evaluated_get(dgraph)

  vts = obj_eval.data.vertices
  all_cds = np.zeros(len(vts) * 3)
  vts.foreach_get('co', all_cds)
  all_cds = add_ones(all_cds.reshape(-1, 3))
  if subset is not None:
    all_cds = all_cds[subset]

  m_world = np.array(obj_eval.matrix_world)
  all_cds = np.matmul(m_world, all_cds.T).T[:,:3]

  return all_cds


def arr_world_to_camera_view(scene, obj, coord):
  # Modified to support array operations from bpy_extras.object_utils.world_to_camera_view
  cam_matrix = np.array(obj.matrix_world.normalized().inverted())
  co_local = np.matmul(cam_matrix, add_ones(coord).T).T[:,:3]
  z = -co_local[:,2]

  camera = obj.data
  frame = [np.array(v) for v in camera.view_frame(scene=scene)[:3]]
  if camera.type != 'ORTHO':
    frame = [(-v / v[2])[None,:] * z[:,None] for v in frame]
    for i in range(len(frame)):
      frame[i][z == 0][:,:2] = .5

  min_x, max_x = frame[2][:,0], frame[1][:,0]
  min_y, max_y = frame[1][:,1], frame[0][:,1]

  x = (co_local[:,0] - min_x) / (max_x - min_x)
  y = (co_local[:,1] - min_y) / (max_y - min_y)

  return np.stack([x, y, z], 1)


def get_coords_clip(obj, f0, f1, subset=None):
  all_cds = []
  for i in range(f0,f1):
    C.scene.frame_set(i)
    cds = get_world_coords(obj, subset)
    all_cds += [cds]

  return np.stack(all_cds, 0)


def get_visible_vertices(cam, vertices, co2D=None, limit=0.02):
  if co2D is None:
    co2D = arr_world_to_camera_view(C.scene, cam, vertices)

  bpy.ops.mesh.primitive_cube_add()
  bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))
  cube = C.active_object

  in_frame = (co2D[:,0] >= 0) & (co2D[:,0] <= 1)
  in_frame &= (co2D[:,1] >= 0) & (co2D[:,1] <= 1)
  in_frame &= (co2D[:,2] > 0)

  is_visible = in_frame.copy()

  valid_idxs = np.arange(len(in_frame))[in_frame]

  for i in valid_idxs:
    v = Vector(vertices[i])
    cube.location = v
    depsgraph = C.evaluated_depsgraph_get()

    # Try a ray cast, in order to test the vertex visibility from the camera
    location= C.scene.ray_cast(depsgraph, cam.location, (v - cam.location).normalized() )
    # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
    if not (location[0] and (v - location[1]).length < limit):
      is_visible[i] = False

  bpy.ops.object.select_all(action='DESELECT')
  cube.select_set(True)
  bpy.ops.object.delete(confirm=False)

  return co2D, is_visible, in_frame

def sanity_check_viz(all_pts, is_visible, in_frame, frame_idx=0):
  C.scene.frame_set(frame_idx)
  for i in range(all_pts.shape[1]):
      pt = all_pts[frame_idx,i]
      vis = is_visible[frame_idx,i]

      bpy.ops.mesh.primitive_cube_add()
      bpy.ops.transform.resize(value=(0.02, 0.02, 0.02))
      cube = C.active_object
      cube.location = pt
      bpy.ops.object.material_slot_add()
      cube.material_slots[0].material = D.materials[2] if vis else D.materials[1]
      if not in_frame[frame_idx,i]:
          cube.material_slots[0].material = D.materials[0]
