# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell


import numpy as np
import os
import sys
import colorsys

import bpy

from . import helper

C = bpy.context
D = bpy.data


def get_materials(prefix=''):
  return [m for m in D.materials if f'{prefix}Material' in m.name]


def new_material(prefix=''):
  n_idx = len(get_materials(prefix))
  m = D.materials.new(f'{prefix}Material{n_idx:04d}')
  m.use_nodes = True

  return m


def init_color_material(color, prefix='', hsv_variance=[0,0,0],
                        roughness=.8, specular=.05, is_hsv=True,
                        is_emission=False, emit_strength=1):
  m = new_material(prefix)
  nt = m.node_tree
  color = np.array(color) + np.random.randn(3) * np.array(hsv_variance)
  color = list(color.clip(0,1))
  color = (*colorsys.hsv_to_rgb(*color), 1)

  if is_emission:
    out_node = nt.nodes.get('Material Output')
    nt.nodes.new('ShaderNodeEmission')
    em = nt.nodes.get('Emission')
    em.inputs.get('Strength').default_value = emit_strength
    em.inputs.get('Color').default_value = color
    new_link(nt, em, 'Emission', out_node, 'Surface')

  else:
    bsdf_node = nt.nodes.get('Principled BSDF')
    bsdf_node.inputs.get('Base Color').default_value = color
    bsdf_node.inputs.get('Roughness').default_value = roughness
    bsdf_node.inputs.get('Specular').default_value = specular

  return m


def assign_material(obj, m=None, prefix='', m_idx=0, slot_idx=0):
  helper.set_active_obj(obj)
  while len(obj.material_slots) < (slot_idx+1):
    bpy.ops.object.material_slot_add()
  obj.active_material_index = slot_idx

  if m is not None:
    obj.active_material = m
  else:
    obj.active_material = get_materials(prefix)[m_idx]


def uv_smart_project(obj):
  helper.set_active_obj(obj)
  bpy.ops.object.mode_set(mode='EDIT')
  bpy.ops.mesh.select_all(action='SELECT')
  bpy.ops.uv.smart_project()
  bpy.ops.object.mode_set(mode='OBJECT')


def new_link(nt, node1, field1, node2, field2):
  node_out = node1.outputs[field1] if isinstance(field1, int) else node1.outputs.get(field1)
  node_inp = node2.inputs[field2] if isinstance(field2, int) else node2.inputs.get(field2)
  nt.links.new(node_out, node_inp)


def create_leaf_material(src_hue, glow=False):
  m = new_material('Leaf')
  nt = m.node_tree

  if glow:
    out_node = nt.nodes.get('Material Output')
    nt.nodes.new('ShaderNodeEmission')
    em = nt.nodes.get('Emission')
    em.inputs.get('Strength').default_value = 1
    em.inputs.get('Color').default_value = (*colorsys.hsv_to_rgb(src_hue + np.random.randn() * .1, 1, 1), 1)
    new_link(nt, em, 'Emission', out_node, 'Surface')

  else:
    info_node = nt.nodes.new('ShaderNodeObjectInfo')
    add_node = nt.nodes.new('ShaderNodeVectorMath')
    mult_node = nt.nodes.new('ShaderNodeVectorMath')
    add2_node = nt.nodes.new('ShaderNodeVectorMath')
    noise_node = nt.nodes.new('ShaderNodeTexWhiteNoise')
    sep_node = nt.nodes.new('ShaderNodeSeparateXYZ')
    hsv_node = nt.nodes.new('ShaderNodeCombineHSV')

    sep_loc_node = nt.nodes.new('ShaderNodeSeparateXYZ')
    loc_mult_node = nt.nodes.new('ShaderNodeMath')
    loc_add_node = nt.nodes.new('ShaderNodeMath')

    bsdf_node = nt.nodes.get('Principled BSDF')
    mult_node.operation = 'MULTIPLY'
    loc_mult_node.operation = 'MULTIPLY'

    add_node.inputs[1].default_value += np.random.randn(3)
    # mult_node.inputs[1].default_value = [.07,.2,.2]
    # add2_node.inputs[1].default_value = [.22,.9,.1]
    # loc_mult_node.inputs[1].default_value = 0
    mult_node.inputs[1].default_value = [.05,.4,.4]
    add2_node.inputs[1].default_value = [src_hue + np.random.randn() * .05,.6,.1]
    loc_mult_node.inputs[1].default_value = 0 #-.01
    # add2_node.inputs[1].default_value += np.random.randn(3) * .1

    # Get HSV color (output of sep_node)
    new_link(nt, info_node, 'Random', add_node, 0)
    new_link(nt, add_node, 0, noise_node, 'Vector')
    new_link(nt, noise_node, 'Color', mult_node, 0)
    new_link(nt, mult_node, 0, add2_node, 0)
    new_link(nt, add2_node, 0, sep_node, 0)

    # Modify H based on Z
    nt.links.new(info_node.outputs.get('Location'), sep_loc_node.inputs[0])
    nt.links.new(sep_loc_node.outputs.get('Z'), loc_mult_node.inputs[0])
    nt.links.new(loc_mult_node.outputs[0], loc_add_node.inputs[0])
    nt.links.new(sep_node.outputs[0], loc_add_node.inputs[1])

    # Combine and assign color
    nt.links.new(loc_add_node.outputs[0], hsv_node.inputs.get('H'))
    nt.links.new(sep_node.outputs[1], hsv_node.inputs.get('S'))
    nt.links.new(sep_node.outputs[2], hsv_node.inputs.get('V'))
    nt.links.new(hsv_node.outputs[0], bsdf_node.inputs.get('Base Color'))


def get_tex_nodes(m):
  """Returns Image Texture node, creates one if it doesn't exist."""
  nt = m.node_tree
  m.cycles.displacement_method = 'DISPLACEMENT'

  # Check whether the Image Texture node has been added
  diff_img_node = nt.nodes.get('Image Texture')
  rough_img_node = nt.nodes.get('Image Texture.001')
  disp_img_node = nt.nodes.get('Image Texture.002')

  if diff_img_node is None:
    # Create new node for linking images
    nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.new('ShaderNodeMapRange')
    diff_img_node = nt.nodes.get('Image Texture')
    rough_img_node = nt.nodes.get('Image Texture.001')
    rough_scaling_node = nt.nodes.get('Map Range')
    disp_img_node = nt.nodes.get('Image Texture.002')

    # Link to main node
    bsdf_node = nt.nodes.get('Principled BSDF')
    nt.links.new(diff_img_node.outputs.get('Color'),
           bsdf_node.inputs.get('Base Color'))
    nt.links.new(rough_img_node.outputs.get('Color'),
           rough_scaling_node.inputs.get('Value'))
    nt.links.new(rough_scaling_node.outputs.get('Result'),
           bsdf_node.inputs.get('Roughness'))

    # Set up nodes for mixing in color
    disp_node = nt.nodes.new('ShaderNodeDisplacement')
    disp_node.space = 'WORLD'
    disp_node.inputs.get('Scale').default_value = 0.05
    out_node = nt.nodes.get('Material Output')
    nt.links.new(disp_img_node.outputs.get('Color'),
           disp_node.inputs.get('Height'))
    nt.links.new(disp_node.outputs.get('Displacement'),
           out_node.inputs.get('Displacement'))

  return diff_img_node, rough_img_node, disp_img_node


def setup_material(m, txt_paths, metal_prob=.2, transm_prob=.2, emit_prob=0):
  """Initialize material given list of paths to diff, rough, disp images."""

  # Load any images that haven't been loaded already
  img_ref = [tpath.split('/')[-1] for tpath in txt_paths]
  for img_idx, img in enumerate(img_ref):
    if not img in D.images:
      try:
        D.images.load(txt_paths[img_idx])
      except:
        pass

  # Initialize and update diff, rough, and disp shader nodes
  txt_nodes = get_tex_nodes(m)
  for n_idx, n in enumerate(txt_nodes):
    try:
      im = D.images.get(img_ref[n_idx])
      if n_idx > 0:
        im.colorspace_settings.name = 'Non-Color'
      n.image = im
    except:
      pass

  nt = m.node_tree
  bsdf = nt.nodes.get('Principled BSDF')
  rough_scale = nt.nodes.get('Map Range')

  bsdf.inputs.get('Metallic').default_value = 0
  bsdf.inputs.get('Transmission').default_value = 0
  bsdf.inputs.get('IOR').default_value = 1.45
  rough_scale.inputs.get('To Max').default_value = 1

  if np.random.rand() < metal_prob:
    bsdf.inputs.get('Metallic').default_value = 1
    rough_scale.inputs.get('To Max').default_value = .5

  elif np.random.rand() < transm_prob:
    bsdf.inputs.get('Transmission').default_value = 1
    bsdf.inputs.get('IOR').default_value = 1.05 + np.random.rand() * .3
    rough_scale.inputs.get('To Max').default_value = .2

  if np.random.rand() < emit_prob:
    out_node = nt.nodes.get('Material Output')

    nt.nodes.new('ShaderNodeEmission')
    nt.nodes.new('ShaderNodeTexNoise')
    nt.nodes.new('ShaderNodeValToRGB') # ColorRamp
    nt.nodes.new('ShaderNodeMixShader')

    em = nt.nodes.get('Emission')
    em.inputs.get('Strength').default_value = 5
    em.inputs.get('Color').default_value = (*colorsys.hsv_to_rgb(np.random.rand(), 1, 1), 1)

    noise = nt.nodes.get('Noise Texture')
    noise.inputs.get('Scale').default_value = np.random.uniform(1,10)
    noise.inputs.get('Distortion').default_value = np.random.uniform(3,10)

    ramp = nt.nodes.get('ColorRamp')
    ramp.color_ramp.elements[0].position = .4
    ramp.color_ramp.elements[1].position = .45
    new_link(nt, noise, 'Color', ramp, 'Fac')

    mix = nt.nodes.get('Mix Shader')
    new_link(nt, ramp, 'Color', mix, 'Fac')
    new_link(nt, bsdf, 'BSDF', mix, 'Shader')
    new_link(nt, em, 'Emission', mix, 'Shader')
    new_link(nt, mix, 'Shader', out_node, 'Surface')
