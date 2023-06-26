# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Lahav Lipson - Render, flat shading, etc
# - Alex Raistrick - Compositing
# - Hei Law - Initial version
# Date Signed: May 2 2023

import os
import time
import warnings
import bpy
import gin
import json
import os
import cv2
import numpy as np
import logging

from imageio import imwrite, imread
from pathlib import Path
from placement import camera as cam_util
from nodes.node_wrangler import NodeWrangler, Nodes
from rendering.post_render import exr_depth_to_jet, flow_to_colorwheel, mask_to_color

from .auto_exposure import nodegroup_auto_exposure

from util.camera import get_calibration_matrix_K_from_blender
from util.logging import Timer
from surfaces import surface
from util import blender as butil, exporting as exputil

TRANSPARENT_SHADERS = {Nodes.TranslucentBSDF, Nodes.TransparentBSDF}

logger = logging.getLogger(__name__)

def remove_translucency():
    # The asserts were added since these edge cases haven't appeared yet -Lahav
    for material in bpy.data.materials:
        nw = NodeWrangler(material.node_tree)
        for node in nw.nodes:
            if node.bl_idname == Nodes.MixShader:
                fac_soc, shader_1_soc, shader_2_soc = node.inputs
                assert shader_1_soc.is_linked and len(shader_1_soc.links) == 1
                assert shader_2_soc.is_linked and len(shader_2_soc.links) == 1
                shader_1_type = shader_1_soc.links[0].from_node.bl_idname
                shader_2_type = shader_2_soc.links[0].from_node.bl_idname
                assert not (shader_1_type in TRANSPARENT_SHADERS and shader_2_type in TRANSPARENT_SHADERS)
                if shader_1_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 1.0
                elif shader_2_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 0.0

def save_and_set_pass_indices(output_folder):
    file_tree = {}
    set_pass_indices(bpy.context.scene.collection, 1, file_tree)
    json_object = json.dumps(file_tree)
    (output_folder / "object_tree.json").write_text(json_object)

def set_pass_indices(parent_collection, index, tree_output):
    for child_obj in parent_collection.objects:
        child_obj.pass_index = index
        object_dict = {
            "type": child_obj.type, "pass_index": index,
            "bbox": np.asarray(child_obj.bound_box[:]).tolist(),
            "matrix_world": np.asarray(child_obj.matrix_world[:]).tolist(),
        }
        if child_obj.type == "MESH":
            object_dict['polycount'] = len(child_obj.data.polygons)
            object_dict['materials'] = child_obj.material_slots.keys()
            object_dict['unapplied_modifiers'] = child_obj.modifiers.keys()
        tree_output[child_obj.name] = object_dict
        index += 1
    for col in parent_collection.children:
        tree_output[col.name] = {"type": "Collection", "hide_viewport": col.hide_viewport, "children": {}}
        index = set_pass_indices(col, index, tree_output=tree_output[col.name]["children"])
    return index

# Can be pasted directly into the blender console
def make_clay():
	clay_material = bpy.data.materials.new(name="clay")
	clay_material.diffuse_color = (0.2, 0.05, 0.01, 1)
	for obj in bpy.data.objects:
		if "atmosphere" not in obj.name.lower() and not obj.hide_render:
			if len(obj.material_slots) == 0:
				obj.active_material = clay_material
			else:
				for mat_slot in obj.material_slots:
					mat_slot.material = clay_material

def enable_gpu(engine_name = 'CYCLES'):
    # from: https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/utility/Initializer.py
    compute_device_type = None
    prefs = bpy.context.preferences.addons['cycles'].preferences
    # Use cycles
    bpy.context.scene.render.engine = engine_name
    bpy.context.scene.cycles.device = 'GPU'

    preferences = bpy.context.preferences.addons['cycles'].preferences
    for device_type in preferences.get_device_types(bpy.context):
        preferences.get_devices_for_type(device_type[0])

    for gpu_type in ['OPTIX', 'CUDA']:#, 'METAL']:
        found = False
        for device in preferences.devices:
            if device.type == gpu_type and (compute_device_type is None or compute_device_type == gpu_type):
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = gpu_type
                logger.info('Device {} of type {} found and used.'.format(device.name, device.type))
                found = True
                break
        if found:
            break

    # make sure that all visible GPUs are used
    for device in prefs.devices:
        device.use = True
    return prefs.devices

@gin.configurable
def compositor_postprocessing(nw, source, show=True, autoexpose=False, autoexpose_level=-2, color_correct=True, distort=0, glare=False):

    if autoexpose:
        source = nw.new_node(nodegroup_auto_exposure().name, input_kwargs={'Image': source, 'EV Comp': autoexpose_level})

    if distort > 0:
        source = nw.new_node(Nodes.LensDistortion,
            input_kwargs={'Image': source, 'Dispersion': distort})
    
    if color_correct:
        source = nw.new_node(Nodes.BrightContrast,
            input_kwargs={'Image': source, 'Bright': 1.0, 'Contrast': 4.0})
    
    if glare:
        source = nw.new_node(
            Nodes.Glare,
            input_kwargs={'Image': source},
            attrs={"glare_type": "GHOSTS", "threshold": 0.5, "mix": -0.99},
        )

    if show:
        nw.new_node(Nodes.Composite, input_kwargs={'Image': source})

    return source.outputs[0]

@gin.configurable
def configure_compositor_output(nw, frames_folder, image_denoised, image_noisy, passes_to_save, saving_ground_truth, use_denoised=False):
    file_output_node = nw.new_node(Nodes.OutputFile, attrs={
        "base_path": str(frames_folder),
        "format.file_format": 'OPEN_EXR' if saving_ground_truth else 'PNG',
        "format.color_mode": 'RGB'
    })
    file_slot_list = []
    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    render_layers = nw.new_node(Nodes.RenderLayers)
    for viewlayer_pass, socket_name in passes_to_save:
        if hasattr(viewlayer, f"use_pass_{viewlayer_pass}"):
            setattr(viewlayer, f"use_pass_{viewlayer_pass}", True)
        else:
            setattr(viewlayer.cycles, f"use_pass_{viewlayer_pass}", True)
        slot_input = file_output_node.file_slots.new(socket_name)
        render_socket = render_layers.outputs[socket_name]
        nw.links.new(render_socket, slot_input)
        file_slot_list.append(file_output_node.file_slots[slot_input.name])

    slot_input = file_output_node.file_slots['Image']
    image = image_denoised if use_denoised else image_noisy
    nw.links.new(image, file_output_node.inputs['Image'])
    if saving_ground_truth:
        slot_input.path = 'Unique_Instances'
    else:
        image_exr_output_node = nw.new_node(Nodes.OutputFile, attrs={
            "base_path": str(frames_folder),
            "format.file_format": 'OPEN_EXR',
            "format.color_mode": 'RGB'
        })
        rgb_exr_slot_input = file_output_node.file_slots['Image']
        nw.links.new(image, image_exr_output_node.inputs['Image'])
        file_slot_list.append(image_exr_output_node.file_slots[rgb_exr_slot_input.path])
    file_slot_list.append(file_output_node.file_slots[slot_input.path])

    return file_slot_list

def shader_random(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo_Shader)

    white_noise_texture = nw.new_node(Nodes.WhiteNoiseTexture,
        input_kwargs={'Vector': object_info.outputs["Random"]})

    nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': white_noise_texture.outputs["Color"]})

def apply_random(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_random, selection=selection)


def global_flat_shading():

    for obj in bpy.context.scene.view_layers['ViewLayer'].objects:
        if obj.name.lower() in {"atmosphere", "atmosphere_fine"}:
            bpy.data.objects.remove(obj)
        elif obj.active_material is not None:
            nw = obj.active_material.node_tree
            for node in nw.nodes:
                if node.bl_idname == Nodes.MaterialOutput:
                    vol_socket = node.inputs['Volume']
                    if len(vol_socket.links) > 0:
                        nw.links.remove(vol_socket.links[0])

    for obj in bpy.context.scene.view_layers['ViewLayer'].objects:
        if obj.type != 'MESH':
            continue
        obj.hide_viewport = False
        if not hasattr(obj, 'material_slots'):
            print(obj.name, 'NONE')
            continue
        with butil.SelectObjects(obj):
            while len(obj.material_slots):
                bpy.ops.object.material_slot_remove()

    for obj in bpy.context.scene.view_layers['ViewLayer'].objects:
        apply_random(obj)

@gin.configurable
def render_image(
    camera_id,
    min_samples,
    num_samples,
    time_limit,
    frames_folder,
    adaptive_threshold,
    exposure,
    passes_to_save,
    flat_shading,
    use_dof=False,
    dof_aperture_fstop=2.8,
    motion_blur=False,
    motion_blur_shutter=0.5,
    render_resolution_override=None,
    excludes=[],
):
    tic = time.time()

    camera_rig_id, subcam_id = camera_id

    for exclude in excludes:
        bpy.data.objects[exclude].hide_render = True

    with Timer(f"Enable GPU"):
        devices = enable_gpu()

    with Timer(f"Render/Cycles settings"):
        if motion_blur: bpy.context.scene.cycles.motion_blur_position = 'START'

        bpy.context.scene.cycles.samples = num_samples # i.e. infinity
        bpy.context.scene.cycles.adaptive_min_samples = min_samples
        bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold # i.e. noise threshold
        bpy.context.scene.cycles.time_limit = time_limit
    
        bpy.context.scene.cycles.film_exposure = exposure
        bpy.context.scene.render.use_motion_blur = motion_blur
        bpy.context.scene.render.motion_blur_shutter = motion_blur_shutter

        bpy.context.scene.cycles.use_denoising = True
        try:
            bpy.context.scene.cycles.denoiser = 'OPTIX'
        except:
            warnings.warn("Cannot use OPTIX denoiser")
        tmp_dir = frames_folder.parent.resolve() / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        bpy.context.scene.render.filepath = f"{tmp_dir}{os.sep}"


    if flat_shading:
        with Timer("Set object indices"):
            save_and_set_pass_indices(frames_folder)

        with Timer("Flat Shading"):
            global_flat_shading()


    with Timer(f"Compositing Setup"):
        if not bpy.context.scene.use_nodes:
            bpy.context.scene.use_nodes = True
            compositor_node_tree = bpy.context.scene.node_tree
            nw = NodeWrangler(compositor_node_tree)

            render_layers        = nw.new_node(Nodes.RenderLayers)
            final_image_denoised = compositor_postprocessing(nw, source=render_layers.outputs["Image"])
            final_image_noisy    = compositor_postprocessing(nw, source=render_layers.outputs["Noisy Image"], show=False)

            compositor_nodes = configure_compositor_output(
                nw,
                frames_folder,
                image_denoised=final_image_denoised,
                image_noisy=final_image_noisy,
                passes_to_save=passes_to_save,
                saving_ground_truth=flat_shading
            )

    ## Update output names
    for file_slot in compositor_nodes:
        file_slot.path = f"{file_slot.path}_####_{camera_rig_id:02d}_{subcam_id:02d}"

    with Timer(f"get_camera"):
        camera = cam_util.get_camera(camera_rig_id, subcam_id)
        if use_dof == 'IF_TARGET_SET':
            use_dof = camera.data.dof.focus_object is not None
        if use_dof is not None:
            camera.data.dof.use_dof = use_dof
            camera.data.dof.aperture_fstop = dof_aperture_fstop

    if render_resolution_override is not None:
        bpy.context.scene.render.resolution_x = render_resolution_override[0]
        bpy.context.scene.render.resolution_y = render_resolution_override[1]

    # Render the scene
    bpy.context.scene.camera = camera
    with Timer(f"Actual rendering"):
        bpy.ops.render.render(animation=True)

    if flat_shading:
        with Timer(f"Post Processing"):
            for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
                bpy.context.scene.frame_set(frame)

                K = get_calibration_matrix_K_from_blender(camera.data)
                cameras_folder = frames_folder / "cameras"
                cameras_folder.mkdir(exist_ok=True, parents=True)
                np.save(
                    frames_folder / f"K{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.npy",
                    np.asarray(K, dtype=np.float64),
                )
                np.save(
                    frames_folder / f"T{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.npy",
                    np.asarray(camera.matrix_world, dtype=np.float64),
                )

                # Save flow visualization. Takes about 3 seconds
                flow_dst_path = frames_folder / f"Vector_{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.exr"
                if flow_dst_path.exists():
                    flow_color = flow_to_colorwheel(flow_dst_path)
                    imwrite(flow_dst_path.with_name(f"Flow_{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.png"), flow_color)

                # Save depth visualization. Also takes about 3 seconds
                depth_dst_path = frames_folder / f"Depth_{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.exr"
                if depth_dst_path.exists():
                    depth_color = exr_depth_to_jet(depth_dst_path)
                    imwrite(depth_dst_path.with_name(f"Depth_{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.png"), depth_color)

                # Save Segmentation visualization. Also takes about 3 seconds
                seg_dst_path = frames_folder / f"IndexOB_{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.exr"
                if seg_dst_path.exists():
                    seg_color = mask_to_color(seg_dst_path)
                    imwrite(seg_dst_path.with_name(f"Segmentation_{frame:04d}_{camera_rig_id:02d}_{subcam_id:02d}.png"), seg_color)

    for file in tmp_dir.glob('*.png'):
        file.unlink()

    logger.info(f"rendering time: {time.time() - tic}")
