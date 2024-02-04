# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Authors: David Yan


import bpy
import os
import sys
import argparse
import shutil
import subprocess
import logging

from pathlib import Path

FORMAT_CHOICES = ["fbx", "obj", "usdc", "usda" "stl", "ply"]
BAKE_TYPES = {'DIFFUSE': 'Base Color', 'ROUGHNESS': 'Roughness'} #  'EMIT':'Emission' #  "GLOSSY": 'Specular', 'TRANSMISSION':'Transmission' don't export
SPECIAL_BAKE = {'METAL': 'Metallic'}

def apply_all_modifiers(obj):
    for mod in obj.modifiers:
        if (mod is None): continue
        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj 
            bpy.ops.object.modifier_apply(modifier=mod.name)
            logging.info(f"Applied modifier {mod.name} on {obj}")
            obj.select_set(False)
        except RuntimeError:
            logging.info(f"Can't apply {mod.name} on {obj}")
            obj.select_set(False)
            return 

def realizeInstances(obj):
    for mod in obj.modifiers:
        if (mod is None or mod.type != 'NODES'): continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes['Group Output']

        logging.info(f"Realizing instances on {mod.name}")
        link = outputNode.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        realizeNode = geo_group.nodes.new(type = 'GeometryNodeRealizeInstances')
        geo_group.links.new(realizeNode.inputs[0], from_socket)
        geo_group.links.new(outputNode.inputs[0], realizeNode.outputs[0])

def remove_shade_smooth(obj):
    for mod in obj.modifiers:
        if (mod is None or mod.type != 'NODES'): continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes['Group Output']
        if  geo_group.nodes.get('Set Shade Smooth'):
            logging.info("Removing shade smooth on " + obj.name)
            smooth_node = geo_group.nodes['Set Shade Smooth']
        else:
            continue

        link = smooth_node.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        geo_group.links.new(outputNode.inputs[0], from_socket)

def check_material_geonode(node_tree):
    if node_tree.nodes.get("Set Material"):
        logging.info("Found set material!")
        return True

    for node in node_tree.nodes:
        if node.type == 'GROUP' and check_material_geonode(node.node_tree):
            return True
    
    return False

def handle_geo_modifiers(obj, export_usd):
    has_geo_nodes = False
    for mod in obj.modifiers:
        if (mod is None or mod.type != 'NODES'): continue
        has_geo_nodes = True
    
    if has_geo_nodes and not obj.data.materials:
        mat = bpy.data.materials.new(name=f"{mod.name} shader")
        obj.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])

    if not export_usd:
        realizeInstances(obj)
    
def clean_names():
    for obj in bpy.data.objects:
        obj.name = (obj.name).replace(' ','_')
        obj.name = (obj.name).replace('.','_')

        if obj.type == 'MESH':
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace('.', '_') # if uv has '.' in name the node will export wrong in USD

    for mat in bpy.data.materials:
        if (mat is None): continue
        mat.name = (mat.name).replace(' ','_')
        mat.name = (mat.name).replace('.','_')

def remove_obj_parents():
    for obj in bpy.data.objects:
        world_loc = obj.matrix_world.to_translation()
        obj.parent = None
        obj.matrix_world.translation = world_loc

def update_visibility(export_usd):
    outliner_area = next(a for a in bpy.context.screen.areas if a.type == 'OUTLINER')
    space = outliner_area.spaces[0]
    space.show_restrict_column_viewport = True  # Global visibility (Monitor icon)
    revealed_collections = []
    hidden_objs = []
    for collection in bpy.data.collections:
        if export_usd:
            collection.hide_viewport = False #reenables viewports for all
            # enables renders for all collections
            if collection.hide_render:
                collection.hide_render = False
                revealed_collections.append(collection)

        elif collection.hide_render: # hides assets if we are realizing instances
            for obj in collection.objects:
                obj.hide_render = True
    
    # disables viewports and renders for all objs
    if export_usd:
        for obj in bpy.data.objects:
            obj.hide_viewport = True
            if not obj.hide_render:
                hidden_objs.append(obj)
                obj.hide_render = True
    
    return revealed_collections, hidden_objs
    
def uv_unwrap(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj 

    obj.data.uv_layers.new(name='ExportUV')
    bpy.context.object.data.uv_layers['ExportUV'].active = True 

    logging.info("UV Unwrapping")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        bpy.ops.uv.smart_project()
    except RuntimeError: 
        logging.info("UV Unwrap failed, skipping mesh")
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
        return False
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)
    return True

def bakeVertexColors(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj 
    vertColor = bpy.context.object.data.color_attributes.new(name='VertColor',domain='CORNER',type='BYTE_COLOR')
    bpy.context.object.data.attributes.active_color = vertColor
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, target ='VERTEX_COLORS')
    obj.select_set(False)

def apply_baked_tex(obj, paramDict={}):
    bpy.context.view_layer.objects.active = obj 
    bpy.context.object.data.uv_layers['ExportUV'].active_render = True
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None): 
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        logging.info("Reapplying baked texs on " + mat.name)

        # delete all nodes except baked nodes and bsdf
        excludedNodes = [type + '_node' for type in BAKE_TYPES]
        excludedNodes.extend([type + '_node' for type in SPECIAL_BAKE])
        excludedNodes.extend(['Material Output','Principled BSDF'])
        for n in nodes: 
            if n.name not in excludedNodes:
                nodes.remove(n) # deletes an arbitrary principled BSDF in the case of a mix, which is handled below

        output = nodes['Material Output']
        
        # stick baked texture in material
        if nodes.get('Principled BSDF') is None: # no bsdf
            logging.info("No BSDF, creating new one")
            principled_bsdf_node = nodes.new('ShaderNodeBsdfPrincipled') 
        elif len(output.inputs[0].links) != 0 and output.inputs[0].links[0].from_node.bl_idname == 'ShaderNodeBsdfPrincipled': # trivial bsdf graph
            logging.info("Trivial shader graph, using old BSDF")
            principled_bsdf_node = nodes['Principled BSDF'] 
        else:
            logging.info("Non-trivial shader graph, creating new BSDF")
            nodes.remove(nodes['Principled BSDF'])  # shader graph was a mix of bsdfs
            principled_bsdf_node = nodes.new('ShaderNodeBsdfPrincipled') 

        links = mat.node_tree.links
        
        # create the new shader node links
        links.new(output.inputs[0], principled_bsdf_node.outputs[0])       
        for type in BAKE_TYPES:
            if not nodes.get(type + '_node'): continue
            tex_node = nodes[type + '_node']
            links.new(principled_bsdf_node.inputs[BAKE_TYPES[type]], tex_node.outputs[0])
        for type in SPECIAL_BAKE:
            if not nodes.get(type + '_node'): continue
            tex_node = nodes[type + '_node']
            links.new(principled_bsdf_node.inputs[BAKE_TYPES[type]], tex_node.outputs[0])

        # bring back cleared param values
        if mat.name in paramDict:
            principled_bsdf_node.inputs['Metallic'].default_value = paramDict[mat.name]['Metallic']
            principled_bsdf_node.inputs['Sheen'].default_value = paramDict[mat.name]['Sheen']
            principled_bsdf_node.inputs['Clearcoat'].default_value = paramDict[mat.name]['Clearcoat']

def create_glass_shader(node_tree):
    nodes = node_tree.nodes
    color = nodes['Glass BSDF'].inputs[0].default_value
    roughness = nodes['Glass BSDF'].inputs[1].default_value
    ior = nodes['Glass BSDF'].inputs[2].default_value
    if nodes.get('Principled BSDF'): 
        nodes.remove(nodes['Principled BSDF'])
        
    principled_bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    principled_bsdf_node.inputs['Base Color'].default_value = color
    principled_bsdf_node.inputs['Roughness'].default_value = roughness
    principled_bsdf_node.inputs['IOR'].default_value = ior
    principled_bsdf_node.inputs['Transmission'].default_value = 1
    node_tree.links.new(principled_bsdf_node.outputs[0], nodes['Material Output'].inputs[0])

def process_glass_materials(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None or not mat.use_nodes): continue
        nodes = mat.node_tree.nodes
        outputNode = nodes['Material Output']
        if nodes.get('Glass BSDF'):
            if outputNode.inputs[0].links[0].from_node.bl_idname == 'ShaderNodeBsdfGlass':
                create_glass_shader(mat.node_tree)
            else:
                logging.info(f"Non-trivial glass material on {obj.name}, material export will be inaccurate")
    
def bake_pass(
    obj, 
    dest: Path, 
    img_size, 
    bake_type, 
):
    
    img = bpy.data.images.new(f'{obj.name}_{bake_type}',img_size,img_size) 
    clean_name = (obj.name).replace(' ','_').replace('.','_')
    file_path = dest/f'{clean_name}_{bake_type}.png'
    dest = dest/'textures'

    bake_obj = False
    bake_exclude_mats = {}

     # materials are stored as stack so when removing traverse the reversed list
    for index, slot in reversed(list(enumerate(obj.material_slots))):
        mat = slot.material
        if mat is None: 
            bpy.context.object.active_material_index = index
            bpy.ops.object.material_slot_remove()
            continue
        
        logging.info(mat.name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        output = nodes["Material Output"]

        img_node = nodes.new('ShaderNodeTexImage')
        img_node.name = f'{bake_type}_node'
        img_node.image = img
        nodes.active = img_node

        if len(output.inputs[0].links) != 0:
            surface_node = output.inputs[0].links[0].from_node
            if surface_node.bl_idname == 'ShaderNodeBsdfPrincipled' and len(surface_node.inputs[BAKE_TYPES[bake_type]].links) == 0: # trivial bsdf graph
                logging.info(f"{mat.name} has no procedural input for {bake_type}, not using baked textures")
                bake_exclude_mats[mat] = img_node
                continue
    
        bake_obj = True 

    if (bake_type == 'METAL'):
        internal_bake_type = 'EMIT'
    else:
        internal_bake_type = bake_type

    if bake_obj:  
        logging.info(f'Baking {bake_type} pass') 
        bpy.ops.object.bake(type=internal_bake_type, pass_filter={'COLOR'}, save_mode='EXTERNAL')
        img.filepath_raw = str(file_path)
        img.save()
        logging.info(f"Saving to {file_path}")
    else:
        logging.info(f"No necessary materials to bake on {obj.name}, skipping bake")

    for mat, img_node in bake_exclude_mats.items():
        mat.node_tree.nodes.remove(img_node)

def bake_metal(obj, dest, img_size): # metal baking is not really set up for node graphs w/ 2 mixed BSDFs. 
    metal_map_mats = []
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None or not mat.use_nodes): continue
        nodes = mat.node_tree.nodes
        if nodes.get('Principled BSDF') and nodes.get('Material Output'):
            principled_bsdf_node = nodes['Principled BSDF']
            outputNode = nodes['Material Output']
        else: continue
        
        links = mat.node_tree.links

        if len(principled_bsdf_node.inputs['Metallic'].links) != 0:
            link = principled_bsdf_node.inputs['Metallic'].links[0]
            from_socket = link.from_socket
            links.remove(link)
            links.new(outputNode.inputs[0], from_socket)
            metal_map_mats.append(mat)

    if len(metal_map_mats) != 0:
        bake_pass(obj, dest, img_size, 'METAL')
        
    for mat in metal_map_mats:
        links.remove(outputNode.inputs[0].links[0])
        links.new(outputNode.inputs[0], principled_bsdf_node.outputs[0])

def remove_params(mat, node_tree):
    paramDict = {}
    nodes = node_tree.nodes
    if nodes.get('Material Output'):
        output = nodes['Material Output']
    elif nodes.get('Group Output'):
        output = nodes['Group Output']
    else:
        raise ValueError("Could not find material output node")
    if nodes.get('Principled BSDF') and output.inputs[0].links[0].from_node.bl_idname == 'ShaderNodeBsdfPrincipled':
        principled_bsdf_node = nodes['Principled BSDF']
        metal = principled_bsdf_node.inputs['Metallic'].default_value # store metallic value and set to 0
        sheen = principled_bsdf_node.inputs['Sheen'].default_value 
        clearcoat = principled_bsdf_node.inputs['Clearcoat'].default_value
        paramDict[mat.name] = {'Metallic': metal, 'Sheen': sheen, 'Clearcoat': clearcoat}
        principled_bsdf_node.inputs['Metallic'].default_value = 0
        principled_bsdf_node.inputs['Sheen'].default_value = 0
        principled_bsdf_node.inputs['Clearcoat'].default_value = 0
    return paramDict
        
def process_interfering_params(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None or not mat.use_nodes): continue
        paramDict = remove_params(mat, mat.node_tree)
        if len(paramDict) == 0:
            for node in mat.node_tree.nodes:  # only handles one level of sub-groups
                if node.type == 'GROUP':
                    paramDict = remove_params(mat, node.node_tree)
                    
    return paramDict

def bake_object(obj, dest, img_size):
    if not uv_unwrap(obj):
        return

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True) 

    for slot in obj.material_slots:
        mat = slot.material
        if mat is not None:
            slot.material = mat.copy() # we duplicate in the case of distinct meshes sharing materials

    process_glass_materials(obj)     
  
    bake_metal(obj, dest, img_size)

    paramDict = process_interfering_params(obj)

    for bake_type in BAKE_TYPES:
        bake_pass(obj, dest, img_size, bake_type)

    apply_baked_tex(obj, paramDict)
    
    obj.select_set(False)

def skipBake(obj, export_usd):
    if not obj.data.materials:
        logging.info("No material on mesh, skipping...")
        return True 

    if obj.hide_render and not export_usd:
        logging.info("Mesh hidden from render, skipping ...")
        return True
    
    if len(obj.data.vertices) == 0:
        logging.info("Mesh has no vertices, skipping ...")
        return True

    return False

def bake_scene(folderPath: Path, image_res, vertex_colors, export_usd):

    for obj in bpy.data.objects:
        logging.info("---------------------------")
        logging.info(obj.name)
        
        if obj.type != 'MESH' or obj not in list(bpy.context.view_layer.objects):
            logging.info("Not mesh, skipping ...")
            continue
     
        handle_geo_modifiers(obj, export_usd)

        if skipBake(obj, export_usd): continue
        
        if format == "stl": 
            continue

        if vertex_colors: 
            bakeVertexColors(obj)
            continue
        
        if export_usd: 
            obj.hide_render = False 
            obj.hide_viewport = False
        
        bake_object(obj, folderPath, image_res)

        if export_usd: 
            obj.hide_render = True
            obj.hide_viewport = True

def run_export(exportPath: Path, format: str, vertex_colors: bool, individual_export: bool):

    assert exportPath.parent.exists()
    exportPath = str(exportPath)    
    
    if format == "obj":
        if vertex_colors:
            bpy.ops.wm.obj_export(filepath = exportPath, export_colors=True, export_selected_objects=individual_export)  
        else:         
            bpy.ops.wm.obj_export(filepath = exportPath, path_mode='COPY', export_materials=True, export_pbr_extensions=True, export_selected_objects=individual_export)

    if format == "fbx":
        if vertex_colors:
            bpy.ops.export_scene.fbx(filepath = exportPath, colors_type='SRGB', use_selection = individual_export)
        else:
            bpy.ops.export_scene.fbx(filepath = exportPath, path_mode='COPY', embed_textures = True, use_selection=individual_export)
    
    if format == "stl": bpy.ops.export_mesh.stl(filepath = exportPath, use_selection = individual_export)

    if format == "ply": bpy.ops.export_mesh.ply(filepath = exportPath, export_selected_objects = individual_export)
    
    if format in ["usda", "usdc"]: bpy.ops.wm.usd_export(filepath = exportPath, export_textures=True, use_instancing=True, selected_objects_only=individual_export)

def export_scene(
    input_blend: Path, 
    output_folder: Path, 
    pipeline_folder=None, 
    task_uniqname=None,
    **kwargs,
):

    bpy.ops.wm.open_mainfile(filepath=str(input_blend))

    folder = output_folder/input_blend.name
    folder.mkdir(exist_ok=True, parents=True)
    result = export_curr_scene(folder, **kwargs)
    
    if pipeline_folder is not None and task_uniqname is not None :
        (pipeline_folder / "logs" / f"FINISH_{task_uniqname}").touch()

    return result

def export_curr_scene(
    output_folder: Path,
    format: str, 
    image_res: int, 
    vertex_colors=False, 
    individual_export=False,
    pipeline_folder=None, 
    task_uniqname=None
) -> Path:
    
    export_usd = format in ["usda", "usdc"]
    
    export_folder = output_folder
    export_folder.mkdir(exist_ok=True)
    export_file = export_folder/output_folder.with_suffix(f'.{format}').name

    logging.info(f"Exporting to directory {export_folder=}")

    # remove grid
    if bpy.data.objects.get("Grid"):
        bpy.data.objects.remove(bpy.data.objects["Grid"], do_unlink=True)
    
    remove_obj_parents()

    scatter_cols = []
    if export_usd:
        if bpy.data.collections.get("scatter"): 
            scatter_cols.append(bpy.data.collections["scatter"])
        if bpy.data.collections.get("scatters"):
            scatter_cols.append(bpy.data.collections["scatters"])
        for col in scatter_cols:
            for obj in col.all_objects:
                remove_shade_smooth(obj) 

    # remove 0 polygon meshes except for scatters
    # if export_usd:
    #     for obj in bpy.data.objects:
    #         if obj.type == 'MESH' and len(obj.data.polygons) == 0:
    #             if scatter_cols is not None:
    #                 if any(x in scatter_cols for x in obj.users_collection): 
    #                     continue
    #             logging.info(f"{obj.name} has no faces, removing...")  
    #             bpy.data.objects.remove(obj, do_unlink=True)   
    
    revealed_collections, hidden_objs = update_visibility(export_usd)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 1 # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    # iterate through all objects and bake them
    bake_scene(
        folderPath=export_folder/'textures',
        image_res=image_res, 
        vertex_colors=vertex_colors, 
        export_usd=export_usd
    )

    for collection in revealed_collections:
        logging.info(f"Hiding collection {collection.name} from render")
        collection.hide_render = True

    for obj in hidden_objs:
        logging.info(f"Unhiding object {obj.name} from render")
        obj.hide_render = False

    # remove all hidden assets if we realized
    if not export_usd:
        for obj in bpy.data.objects:
            if obj.hide_render:
                bpy.data.objects.remove(obj, do_unlink=True)    

    clean_names()

    if individual_export:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.location_clear() # send all objects to (0,0,0)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type != 'MESH' or obj.hide_render or len(obj.data.vertices) == 0 or obj not in list(bpy.context.view_layer.objects):
                continue

            export_subfolder = export_folder/obj.name
            export_subfolder.mkdir(exist_ok=True)
            export_file = export_subfolder/f'{obj.name}.{format}'

            logging.info(f"Exporting file to {export_file=}")
            obj.hide_viewport = False
            obj.select_set(True)
            run_export(export_file, format, vertex_colors, individual_export)
            obj.select_set(False)     
    else:
        logging.info(f"Exporting file to {export_file=}")
        run_export(export_file, format, vertex_colors, individual_export)

    return export_folder 

def main(args):

    args.output_folder.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.DEBUG)
    
    targets = sorted(list(args.input_folder.iterdir()))
    for blendfile in targets:

        if not blendfile.suffix == '.blend':
            print(f'Skipping non-blend file {blendfile}')
            continue
        
        folder = export_scene(
            blendfile, 
            args.output_folder, 
            format=args.format, 
            image_res=args.resolution, 
            vertex_colors=args.vertex_colors,
            individual_export=args.individual,
        )

        # wanted to use shutil here but kept making corrupted files
        subprocess.call(['zip', '-r', str(folder.absolute().with_suffix('.zip')), str(folder.absolute())]) 
    
    bpy.ops.wm.quit_blender()

def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=Path)
    parser.add_argument('--output_folder', type=Path)

    parser.add_argument('-f', '--format', type=str, choices=FORMAT_CHOICES)

    parser.add_argument('-v', '--vertex_colors', action = 'store_true')
    parser.add_argument('-r', '--resolution', default= 1024, type=int)
    parser.add_argument('-i', '--individual', action = 'store_true')
    
    args = parser.parse_args()

    if args.format not in FORMAT_CHOICES:
        raise ValueError("Unsupported or invalid file format.")

    if args.vertex_colors and args.format not in ["ply", "fbx", "obj"]:
        raise ValueError("File format does not support vertex colors.")

    if (args.format == "ply" and not args.vertex_colors):
        raise ValueError(".ply export must use vertex colors.")

    return args

if __name__ == '__main__':
    args = make_args()
    main(args)
