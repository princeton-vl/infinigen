# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell


import bpy
import numpy as np

from infinigen.core.util.logging import Suppress

C = bpy.context
D = bpy.data


def set_active_obj(obj):
  if not C.active_object == obj:
    try:
      bpy.ops.object.mode_set(mode='OBJECT')
    except:
      pass
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    C.view_layer.objects.active = obj
  bpy.ops.object.mode_set(mode='OBJECT')


def config_rendering(resolution=(480, 640), renderer='cycles', render_samples=64,
                     render_exr=False, thread_limit=8):
  """Adjust rendering settings.

  Args:
    resolution: Integer tuple for image resolution
    renderer: Either 'cycles' or 'eevee'
    render_samples: Integer that determines sample quality, rendering time
    use_gpu: Whether to use the GPU for rendering
    render_exr: Set true to output segmentation and depth ground truth
  """

  if renderer == 'eevee':
    C.scene.render.engine = 'BLENDER_EEVEE'
    C.scene.eevee.taa_render_samples = render_samples

  elif renderer == 'cycles':
    C.scene.render.engine = 'CYCLES'
    # C.scene.cycles.device = 'GPU'
    C.scene.cycles.samples = render_samples
    C.scene.cycles.use_denoising = True
    # C.scene.cycles.denoiser = 'OPTIX'

  C.scene.render.resolution_x = resolution[1]
  C.scene.render.resolution_y = resolution[0]
  # C.scene.render.threads_mode = 'FIXED'
  # C.scene.render.threads = thread_limit

  if render_exr:
    C.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    C.scene.render.image_settings.color_mode = 'RGBA'
    C.scene.render.image_settings.color_depth = '32'
    C.window.view_layer.use_pass_object_index = True
    C.window.view_layer.use_pass_material_index = True
    C.window.view_layer.use_pass_z = True

  else:
    C.scene.render.image_settings.color_mode = 'RGB'


def create_collection(name, objs):
  c_names = []
  for c_idx, c in enumerate(D.collections):
    if c_idx > 0:
      c_names += [c.name]

  name_ = name
  count = 1
  while name_ in c_names:
    name_ = f'{name}_{count}'
    count += 1

  bpy.ops.object.select_all(action='DESELECT')
  for o in objs:
    o.select_set(True)

  with Suppress():
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name=name_)

  return name_


def traverse_tree(t):
  # https://blender.stackexchange.com/questions/172559/python-how-to-move-collection-into-another-collection
  yield t
  for child in t.children:
    yield from traverse_tree(child)


def parent_lookup(coll):
  parent_lookup = {}
  for coll in traverse_tree(coll):
    for c in coll.children.keys():
      parent_lookup.setdefault(c, coll)
  return parent_lookup


def collect_collections(name, colls):
  # Get all collections of the scene and their parents in a dict
  coll_scene = C.scene.collection
  coll_parents = parent_lookup(coll_scene)

  # Create target collection
  D.collections.new(name)
  coll_target = D.collections[name]
  coll_scene.children.link(coll_target)

  for coll in colls:
    coll_parent = coll_parents.get(coll.name)
    coll_parent.children.unlink(coll)
    coll_target.children.link(coll)


def remove_collection(name):
  collection = D.collections.get(name)
  for obj in collection.objects:
      D.objects.remove(obj, do_unlink=True)
  D.collections.remove(collection)


def hide_collection(collection):
  if isinstance(collection, str):
    name = collection
    collection = D.collections[name]
  else:
    name = collection.name

  vlayer = C.scene.view_layers[0]
  vlayer.layer_collection.children[name].hide_viewport = True
  collection.hide_render = True


def clear_collections():
  c_names = []
  for c_idx, c in enumerate(D.collections):
    if c_idx > 0:
      c_names += [c.name]

  for c_name in c_names:
    remove_collection(c_name)


def run_cleanup():
  for d in [D.meshes, D.materials, D.images, D.particles]:
    for d_ in d:
      if d_.users == 0:
        d.remove(d_)
  for d in [D.textures, D.node_groups]:
    for d_ in d:
      d.remove(d_)


def reset_scene(add_camera=False, clear_materials=False, obj_to_keep_list=[]):
  """Clear and reset scene."""
  set_active_obj(D.objects[0])

  for obj in D.objects:
      obj.hide_viewport = False

  # Delete everything
  clear_collections()
  # bpy.ops.object.select_all(action='SELECT')
  for obj in bpy.context.scene.objects:
    if obj.name not in obj_to_keep_list:
      obj.select_set(True)
  bpy.ops.object.delete(confirm=False)
  run_cleanup()

  if add_camera:
    # Initialize camera
    v = min(1,max(0,(np.random.randn() * .3 + .5)))
    v = 0
    camera_height = .5 + 3 * v # np.random.uniform(1,5) # + np.random.randn() * .2
    camera_pitch = np.pi * .45 # + np.random.randn() * np.pi * .1
    camera_pitch = min(max(camera_pitch, np.pi * .4), np.pi * .5)
    camera_pitch = np.pi * .65  # (1-v) * np.pi * .6 + np.pi * .2

    camera_pitch = np.pi * 0.5
    camera_height = 3
    
    bpy.ops.object.camera_add(location=(0, -6, camera_height), rotation=(camera_pitch, 0, 0))
    cam = D.objects[0]
    C.scene.camera = cam
    cam.data.lens = 20

  if clear_materials: # Regardless of number of users
    for m_idx in range(len(D.materials)):
      D.materials.remove(D.materials[-1])


# ==============================================================================
# Transformation utils
# ==============================================================================


def compute_dists(a, b):
  deltas = a[:,None] - b[None]
  d = np.linalg.norm(deltas, axis=-1)
  return d, deltas


def get_cos_sin(angle, convert_to_rad=False):
  if convert_to_rad:
    angle = angle * np.pi / 180
  return np.cos(angle), np.sin(angle)


def rodrigues_rot(vec, axis, angle, convert_to_rad=False):
  axis = axis / np.linalg.norm(axis)
  cs, sn = get_cos_sin(angle, convert_to_rad)
  return vec * cs + sn * np.cross(axis, vec) + axis * np.dot(axis, vec) * (1 - cs)


def get_T_mat(distance, angle, convert_to_rad=True):
  T = np.identity(3)
  T[0,2] = distance
  rot = np.identity(3)
  cs, sn = get_cos_sin(angle, convert_to_rad)
  rot[0,:2] = cs, -sn
  rot[1,:2] = sn, cs

  return np.matmul(rot, T)


def valid_pos(d0=2, d1=10):
  camera_pos = C.scene.camera.location
  view_angle = C.scene.camera.rotation_euler[2]
  tmp_ang = (C.scene.camera.data.angle / 2) * .9
  tmp_ang = np.random.rand() * 2 * tmp_ang - tmp_ang
  tmp_ang += view_angle
  tmp_dist = np.random.rand() * (d1 - d0) + d0
  root_pos = np.array([camera_pos[0], camera_pos[1]])
  v_dir = np.array([-np.sin(tmp_ang), np.cos(tmp_ang)])

  return root_pos + tmp_dist * v_dir
