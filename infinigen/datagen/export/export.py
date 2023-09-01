# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Portions of bakeTexture heavily modified from https://blender.stackexchange.com/a/191841

# Authors: David Yan


import bpy
import os
import sys
import argparse
import shutil

from infinigen.core.init import parse_args_blender

def realizeInstances(obj):
    for mod in obj.modifiers:
        if (mod is None or mod.type != 'NODES'): continue
        print(mod)
        print(mod.node_group)
        print("Realizing instances on " + obj.name)
        geo_group = mod.node_group
        outputNode = geo_group.nodes['Group Output']
        for link in geo_group.links: #search for link to the output node
            if (link.to_node == outputNode):
                print("Found Link!")
                from_socket = link.from_socket
                geo_group.links.remove(link)
                realizeNode = geo_group.nodes.new(type = 'GeometryNodeRealizeInstances')
                geo_group.links.new(realizeNode.inputs[0], from_socket)
                geo_group.links.new(outputNode.inputs[0], realizeNode.outputs[0])
                print("Applying modifier")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj 
                bpy.ops.object.modifier_apply(modifier= mod.name)
                obj.select_set(True)
                return

def bakeVertexColors(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj 
    vertColor = bpy.context.object.data.color_attributes.new(name="VertColor",domain='CORNER',type='BYTE_COLOR')
    bpy.context.object.data.attributes.active_color = vertColor
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, target ='VERTEX_COLORS')
    obj.select_set(False)

def bakeTexture(obj, dest, img_size): # modified from https://blender.stackexchange.com/a/191841
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj 

    imgDiffuse = bpy.data.images.new(obj.name + '_Diffuse',img_size,img_size) 
    imgRough = bpy.data.images.new(obj.name + '_Rough',img_size,img_size) 

    #UV Unwrap
    print("UV Unwrapping")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin= 0.001)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    diffuse_file_name = obj.name + '_Diffuse.png'
    diffuse_file_path = os.path.join(dest, diffuse_file_name)
    
    metalDict = {}
    noBSDF = False

    # Iterate on all objects and their materials and bake to an image texture

    # Diffuse pass
    noMaterials = True
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None): continue
        noMaterials = False
        print(mat.name)
        slot.material = mat.copy() # we duplicate in the case of distinct meshes sharing materials
        mat = slot.material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        diffuse_node = nodes.new('ShaderNodeTexImage')
        diffuse_node.name = 'Diffuse_node'
        diffuse_node.image = imgDiffuse
        nodes.active = diffuse_node

        if (nodes.get("Principled BSDF") is None):
            noBSDF = True
        else:
            principled_bsdf_node = nodes["Principled BSDF"]
            metalDict[mat.name] = principled_bsdf_node.inputs["Metallic"].default_value # store metallic value and set to 0
            principled_bsdf_node.inputs["Metallic"].default_value = 0

    print(metalDict)

    if (noMaterials):
        return

    print("Baking Diffuse...")
    bpy.ops.object.bake(type='DIFFUSE',pass_filter={'COLOR'}, save_mode='EXTERNAL')
    
    # Roughness pass
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None): continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        rough_node = nodes.new('ShaderNodeTexImage')
        rough_node.name = 'Rough_node'
        rough_node.image = imgRough
        nodes.active = rough_node
  
    rough_file_name = obj.name + '_Rough.png'
    rough_file_path = os.path.join(dest, rough_file_name)
    
    print("Baking Roughness...")
    bpy.ops.object.bake(type='ROUGHNESS', save_mode='EXTERNAL')
    
    print("Saving to " + diffuse_file_path)
    print("Saving to " + rough_file_path)

    imgDiffuse.filepath_raw = diffuse_file_path
    imgRough.filepath_raw = rough_file_path
    imgDiffuse.save()
    imgRough.save()

    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None): continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        print("Reapplying baked texs on " + mat.name)
   
        # delete all nodes except baked nodes and bsdf
        for n in nodes: 
            excludedNodes = {'Principled BSDF','Material Output', "Rough_node", "Diffuse_node"}
            if n.name not in excludedNodes:
                nodes.remove(n)

        diffuse_node = nodes["Diffuse_node"]
        rough_node = nodes["Rough_node"]
        output = nodes["Material Output"]
        
        # stick baked texture in material
        if (noBSDF):
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        else:
            principled_bsdf_node = nodes["Principled BSDF"]

        links = mat.node_tree.links

        # create the new shader node links
        links.new(output.inputs[0], principled_bsdf_node.outputs[0])       
        links.new(principled_bsdf_node.inputs["Base Color"], diffuse_node.outputs[0])
        links.new(principled_bsdf_node.inputs["Roughness"], rough_node.outputs[0])

        # bring back metallic values
        if not noBSDF:
            principled_bsdf_node.inputs["Metallic"].default_value = metalDict[mat.name]
    
    # strip spaces and dots from names
    for slot in obj.material_slots:
        mat = slot.material
        if (mat is None): continue
        mat.name = (mat.name).replace(' ','_')
        mat.name = (mat.name).replace('.','_')

    obj.select_set(False)

              

def main(args, source, dest):
    for filename in os.listdir(source):
        if not filename.endswith('.blend'):
            continue
        
        # setting up directory and files
        filePath = os.path.join(source, filename)
        
        bpy.ops.wm.open_mainfile(filepath = filePath) 
                                 
        projName = bpy.path.basename(bpy.context.blend_data.filepath) #gets basename e.g. thisfile.blend
        
        baseName = os.path.splitext(projName)[0] #gets the filename without .blend extension e.g. thisfile
    
        folderPath = os.path.join(dest, baseName) # folder path with name of blend file
        
        if not os.path.exists(folderPath):
           os.mkdir(folderPath)

        if args.obj:
            exportName = baseName + ".obj" #changes extension
        if args.fbx:
            exportName = baseName + ".fbx"
        if args.stl:
            exportName = baseName + ".stl"
        if args.ply:
            exportName = baseName + ".ply"
        
        exportPath = os.path.join(folderPath, exportName) # path

        print("Exporting to " + exportPath)
        
        # some objects may be in a collection hidden from render 
        # but not actually hidden themselves. this hides those objects
        for collection in bpy.data.collections:     
            if (collection.hide_render):
                for obj in collection.objects:
                    obj.hide_render = True

        # remove grid
        if (bpy.data.objects.get("Grid") is not None):
            bpy.data.objects.remove(bpy.data.objects["Grid"], do_unlink=True)

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 1 # choose render sample
        
        # iterate through all objects and bake them
        for obj in bpy.data.objects:
            print("---------------------------")
            print(obj.name)

            obj.name = (obj.name).replace(' ','_')
            obj.name = (obj.name).replace('.','_')
            
            if obj.type != 'MESH':
                print("Not mesh, skipping ...")
                continue
            
            if obj.hide_render:
                print("Mesh hidden from render, skipping ...")
                continue
            
            if (len(obj.data.vertices) == 0):
                print("Mesh has no vertices, skipping ...")
                continue

            realizeInstances(obj)
            if args.stl: 
                continue
            if args.vertex_colors: 
                bakeVertexColors(obj)
                continue
            bpy.ops.object.select_all(action='DESELECT')
            bakeTexture(obj,folderPath, args.resolution) 

        # remove all the hidden objects
        for obj in bpy.data.objects:
            if obj.hide_render:
                bpy.data.objects.remove(obj, do_unlink=True)        
        
        if args.obj:
            bpy.ops.export_scene.obj(filepath = exportPath, path_mode='COPY', use_materials =True)

        if args.fbx:
            if args.vertex_colors:
                bpy.ops.export_scene.fbx(filepath = exportPath, colors_type='SRGB')
            else:
                bpy.ops.export_scene.fbx(filepath = exportPath, path_mode='COPY', embed_textures = True)
        
        if args.stl:
            bpy.ops.export_mesh.stl(filepath = exportPath)

        if args.ply:
            bpy.ops.export_mesh.ply(filepath = exportPath)

        shutil.make_archive(folderPath, 'zip', folderPath)
        shutil.rmtree(folderPath)

    bpy.ops.wm.quit_blender()

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def make_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('-b', '--blend_folder', type=dir_path)
    parser.add_argument('-e', '--export_folder', type=dir_path)

    group.add_argument('-f', '--fbx', action = 'store_true') # fbx export has some minor issues with roughness map accuracy
    group.add_argument('-o', '--obj', action = 'store_true')
    group.add_argument('-s', '--stl', action = 'store_true')
    group.add_argument('-p', '--ply', action = 'store_true')

    parser.add_argument('-v', '--vertex_colors', action = 'store_true')
    parser.add_argument('-r', '--resolution', default= 1024, type=int)
    
    args = parse_args_blender(parser)

    if (args.vertex_colors and (args.obj or args.stl)):
        raise ValueError("File format does not support vertex colors.")

    if (args.ply and not args.vertex_colors):
        raise ValueError(".ply export must use vertex colors.")

    return args

if __name__ == '__main__':
    args = make_args()
    main(args, args.blend_folder, args.export_folder)
