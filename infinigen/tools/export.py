# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Authors: David Yan


import argparse
import logging
import math
import shutil
import subprocess
from pathlib import Path

import bpy
import gin

from infinigen.core.util import blender as butil

FORMAT_CHOICES = ["fbx", "obj", "usdc", "usda", "stl", "ply"]
BAKE_TYPES = {
    "DIFFUSE": "Base Color",
    "ROUGHNESS": "Roughness",
}  # 'EMIT':'Emission Color' #  "GLOSSY": 'Specular IOR Level', 'TRANSMISSION':'Transmission Weight' don't export
SPECIAL_BAKE = {"METAL": "Metallic", "NORMAL": "Normal"}
ALL_BAKE = BAKE_TYPES | SPECIAL_BAKE


def apply_all_modifiers(obj):
    for mod in obj.modifiers:
        if mod is None:
            continue
        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
            logging.info(f"Applied modifier {mod} on {obj}")
            obj.select_set(False)
        except RuntimeError:
            logging.info(f"Can't apply {mod} on {obj}")
            obj.select_set(False)
            return


def realizeInstances(obj):
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes["Group Output"]

        logging.info(f"Realizing instances on {mod}")
        link = outputNode.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        realizeNode = geo_group.nodes.new(type="GeometryNodeRealizeInstances")
        geo_group.links.new(realizeNode.inputs[0], from_socket)
        geo_group.links.new(outputNode.inputs[0], realizeNode.outputs[0])


def remove_shade_smooth(obj):
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes["Group Output"]
        if geo_group.nodes.get("Set Shade Smooth"):
            logging.info("Removing shade smooth on " + obj.name)
            smooth_node = geo_group.nodes["Set Shade Smooth"]
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
        if node.type == "GROUP" and check_material_geonode(node.node_tree):
            return True

    return False


def handle_geo_modifiers(obj, export_usd):
    has_geo_nodes = False
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        has_geo_nodes = True

    if has_geo_nodes and not obj.data.materials:
        mat = bpy.data.materials.new(name=f"{mod.name} shader")
        obj.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])

    if not export_usd:
        realizeInstances(obj)


def split_glass_mats():
    split_objs = []
    for obj in bpy.data.objects:
        if obj.hide_render or obj.hide_viewport:
            continue
        if any(
            exclude in obj.name
            for exclude in ["BowlFactory", "CupFactory", "OvenFactory", "BottleFactory"]
        ):
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            if ("shader_glass" in mat.name or "shader_lamp_bulb" in mat.name) and len(
                obj.material_slots
            ) >= 2:
                logging.info(f"Splitting {obj}")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.separate(type="MATERIAL")
                bpy.ops.object.mode_set(mode="OBJECT")
                obj.select_set(False)
                split_objs.append(obj.name)
                break

    matches = [
        obj
        for split_obj in split_objs
        for obj in bpy.data.objects
        if split_obj in obj.name
    ]
    for match in matches:
        mat = match.material_slots[0].material
        if mat is None:
            continue
        if "shader_glass" in mat.name or "shader_lamp_bulb" in mat.name:
            match.name = f"{match.name}_SPLIT_GLASS"


def clean_names(obj=None):
    if obj is not None:
        obj.name = (obj.name).replace(" ", "_")
        obj.name = (obj.name).replace(".", "_")

        if obj.type == "MESH":
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace(".", "_")

        for mat in bpy.data.materials:
            if mat is None:
                continue
            mat.name = (mat.name).replace(" ", "_")
            mat.name = (mat.name).replace(".", "_")

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.name = (mat.name).replace(" ", "_")
            mat.name = (mat.name).replace(".", "_")
        return

    for obj in bpy.data.objects:
        obj.name = (obj.name).replace(" ", "_")
        obj.name = (obj.name).replace(".", "_")

        if obj.type == "MESH":
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace(
                    ".", "_"
                )  # if uv has '.' in name the node will export wrong in USD

    for mat in bpy.data.materials:
        if mat is None:
            continue
        mat.name = (mat.name).replace(" ", "_")
        mat.name = (mat.name).replace(".", "_")


def remove_obj_parents(obj=None):
    if obj is not None:
        old_location = obj.matrix_world.to_translation()
        obj.parent = None
        obj.matrix_world.translation = old_location
        return

    for obj in bpy.data.objects:
        old_location = obj.matrix_world.to_translation()
        obj.parent = None
        obj.matrix_world.translation = old_location


def delete_objects():
    logging.info("Deleting placeholders collection")
    collection_name = "placeholders"
    collection = bpy.data.collections.get(collection_name)

    if collection:
        for scene in bpy.data.scenes:
            if collection.name in scene.collection.children:
                scene.collection.children.unlink(collection)

        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        def delete_child_collections(parent_collection):
            for child_collection in parent_collection.children:
                delete_child_collections(child_collection)
                bpy.data.collections.remove(child_collection)

        delete_child_collections(collection)
        bpy.data.collections.remove(collection)

    if bpy.data.objects.get("Grid"):
        bpy.data.objects.remove(bpy.data.objects["Grid"], do_unlink=True)

    if bpy.data.objects.get("atmosphere"):
        bpy.data.objects.remove(bpy.data.objects["atmosphere"], do_unlink=True)

    if bpy.data.objects.get("KoleClouds"):
        bpy.data.objects.remove(bpy.data.objects["KoleClouds"], do_unlink=True)


def rename_all_meshes(obj=None):
    if obj is not None:
        if obj.data and obj.data.users == 1:
            obj.data.name = obj.name
        return

    for obj in bpy.data.objects:
        if obj.data and obj.data.users == 1:
            obj.data.name = obj.name


def update_visibility():
    outliner_area = next(a for a in bpy.context.screen.areas if a.type == "OUTLINER")
    space = outliner_area.spaces[0]
    space.show_restrict_column_viewport = True  # Global visibility (Monitor icon)
    collection_view = {}
    obj_view = {}
    for collection in bpy.data.collections:
        collection_view[collection] = collection.hide_render
        collection.hide_viewport = False  # reenables viewports for all
        collection.hide_render = False  # enables renders for all collections

    # disables viewports and renders for all objs
    for obj in bpy.data.objects:
        obj_view[obj] = obj.hide_render
        obj.hide_viewport = True
        obj.hide_render = True
        obj.hide_set(0)

    return collection_view, obj_view


def uv_unwrap(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.uv_layers.new(name="ExportUV")
    bpy.context.object.data.uv_layers["ExportUV"].active = True

    logging.info("UV Unwrapping")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=0.7)
    except RuntimeError:
        logging.info("UV Unwrap failed, skipping mesh")
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        return False
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return True


def bakeVertexColors(obj):
    logging.info(f"Baking vertex color on {obj}")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    vertColor = bpy.context.object.data.color_attributes.new(
        name="VertColor", domain="CORNER", type="BYTE_COLOR"
    )
    bpy.context.object.data.attributes.active_color = vertColor
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, target="VERTEX_COLORS")
    obj.select_set(False)


def apply_baked_tex(obj, paramDict={}):
    bpy.context.view_layer.objects.active = obj
    bpy.context.object.data.uv_layers["ExportUV"].active_render = True
    for uv_layer in reversed(obj.data.uv_layers):
        if "ExportUV" not in uv_layer.name:
            logging.info(f"Removed extraneous UV Layer {uv_layer}")
            obj.data.uv_layers.remove(uv_layer)

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        logging.info("Reapplying baked texs on " + mat.name)

        # delete all nodes except baked nodes and bsdf
        excludedNodes = [type + "_node" for type in ALL_BAKE]
        excludedNodes.extend(["Material Output", "Principled BSDF"])
        for n in nodes:
            if n.name not in excludedNodes:
                nodes.remove(
                    n
                )  # deletes an arbitrary principled BSDF in the case of a mix, which is handled below

        output = nodes["Material Output"]

        # stick baked texture in material
        if nodes.get("Principled BSDF") is None:  # no bsdf
            logging.info("No BSDF, creating new one")
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        elif (
            len(output.inputs[0].links) != 0
            and output.inputs[0].links[0].from_node.bl_idname
            == "ShaderNodeBsdfPrincipled"
        ):  # trivial bsdf graph
            logging.info("Trivial shader graph, using old BSDF")
            principled_bsdf_node = nodes["Principled BSDF"]
        else:
            logging.info("Non-trivial shader graph, creating new BSDF")
            nodes.remove(nodes["Principled BSDF"])  # shader graph was a mix of bsdfs
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

        links = mat.node_tree.links

        # create the new shader node links
        links.new(output.inputs[0], principled_bsdf_node.outputs[0])
        for type in ALL_BAKE:
            if not nodes.get(type + "_node"):
                continue
            tex_node = nodes[type + "_node"]
            if type == "NORMAL":
                normal_node = nodes.new("ShaderNodeNormalMap")
                links.new(normal_node.inputs["Color"], tex_node.outputs[0])
                links.new(
                    principled_bsdf_node.inputs[ALL_BAKE[type]], normal_node.outputs[0]
                )
                continue
            links.new(principled_bsdf_node.inputs[ALL_BAKE[type]], tex_node.outputs[0])

        # bring back cleared param values
        if mat.name in paramDict:
            principled_bsdf_node.inputs["Metallic"].default_value = paramDict[mat.name][
                "Metallic"
            ]
            principled_bsdf_node.inputs["Sheen Weight"].default_value = paramDict[
                mat.name
            ]["Sheen Weight"]
            principled_bsdf_node.inputs["Coat Weight"].default_value = paramDict[
                mat.name
            ]["Coat Weight"]


def create_glass_shader(node_tree, export_usd):
    nodes = node_tree.nodes
    if nodes.get("Glass BSDF"):
        color = nodes["Glass BSDF"].inputs[0].default_value
        roughness = nodes["Glass BSDF"].inputs[1].default_value
        ior = nodes["Glass BSDF"].inputs[2].default_value

    if nodes.get("Principled BSDF"):
        nodes.remove(nodes["Principled BSDF"])

    principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

    if nodes.get("Glass BSDF"):
        principled_bsdf_node.inputs["Base Color"].default_value = color
        principled_bsdf_node.inputs["Roughness"].default_value = roughness
        principled_bsdf_node.inputs["IOR"].default_value = ior
    else:
        principled_bsdf_node.inputs["Roughness"].default_value = 0

    principled_bsdf_node.inputs["Transmission Weight"].default_value = 1
    if export_usd:
        principled_bsdf_node.inputs["Alpha"].default_value = 0
    node_tree.links.new(
        principled_bsdf_node.outputs[0], nodes["Material Output"].inputs[0]
    )


def process_glass_materials(obj, export_usd):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        outputNode = nodes["Material Output"]
        if nodes.get("Glass BSDF"):
            if (
                outputNode.inputs[0].links[0].from_node.bl_idname
                == "ShaderNodeBsdfGlass"
            ):
                logging.info(f"Creating glass material on {obj.name}")
            else:
                logging.info(
                    f"Non-trivial glass material on {obj.name}, material export will be inaccurate"
                )
            create_glass_shader(mat.node_tree, export_usd)
        elif "glass" in mat.name or "shader_lamp_bulb" in mat.name:
            logging.info(f"Creating glass material on {obj.name}")
            create_glass_shader(mat.node_tree, export_usd)


def bake_pass(obj, dest: Path, img_size, bake_type, export_usd):
    img = bpy.data.images.new(f"{obj.name}_{bake_type}", img_size, img_size)
    clean_name = (obj.name).replace(" ", "_").replace(".", "_")
    file_path = dest / f"{clean_name}_{bake_type}.png"
    dest = dest / "textures"

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

        img_node = nodes.new("ShaderNodeTexImage")
        img_node.name = f"{bake_type}_node"
        img_node.image = img
        img_node.select = True
        nodes.active = img_node
        img_node.select = True

        if len(output.inputs[0].links) == 0:
            logging.info(f"{mat.name} has no surface output, not using baked textures")
            bake_exclude_mats[mat] = img_node
            continue

        surface_node = output.inputs[0].links[0].from_node
        if (
            bake_type in ALL_BAKE
            and surface_node.bl_idname == "ShaderNodeBsdfPrincipled"
            and len(surface_node.inputs[ALL_BAKE[bake_type]].links) == 0
        ):  # trivial bsdf graph
            logging.info(
                f"{mat.name} has no procedural input for {bake_type}, not using baked textures"
            )
            bake_exclude_mats[mat] = img_node
            continue

        bake_obj = True

    if bake_type == "METAL":
        internal_bake_type = "EMIT"
    else:
        internal_bake_type = bake_type

    if bake_obj:
        logging.info(f"Baking {bake_type} pass")
        bpy.ops.object.bake(
            type=internal_bake_type, pass_filter={"COLOR"}, save_mode="EXTERNAL"
        )
        img.filepath_raw = str(file_path)
        if not export_usd:
            img.save()
        logging.info(f"Saving to {file_path}")
    else:
        logging.info(f"No necessary materials to bake on {obj.name}, skipping bake")

    for mat, img_node in bake_exclude_mats.items():
        mat.node_tree.nodes.remove(img_node)


def bake_metal(
    obj, dest, img_size, export_usd
):  # metal baking is not really set up for node graphs w/ 2 mixed BSDFs.
    metal_map_mats = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        if nodes.get("Principled BSDF") and nodes.get("Material Output"):
            principled_bsdf_node = nodes["Principled BSDF"]
            outputNode = nodes["Material Output"]
        else:
            continue

        links = mat.node_tree.links

        if len(principled_bsdf_node.inputs["Metallic"].links) != 0:
            link = principled_bsdf_node.inputs["Metallic"].links[0]
            from_socket = link.from_socket
            links.remove(link)
            links.new(outputNode.inputs[0], from_socket)
            metal_map_mats.append(mat)

    if len(metal_map_mats) != 0:
        bake_pass(obj, dest, img_size, "METAL", export_usd)

    for mat in metal_map_mats:
        nodes = mat.node_tree.nodes
        outputNode = nodes["Material Output"]
        principled_bsdf_node = nodes["Principled BSDF"]
        links.remove(outputNode.inputs[0].links[0])
        links.new(outputNode.inputs[0], principled_bsdf_node.outputs[0])


def bake_normals(obj, dest, img_size, export_usd):
    bake_obj = False
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        if nodes.get("Material Output"):
            outputNode = nodes["Material Output"]
        else:
            continue

        if len(outputNode.inputs["Displacement"].links) != 0:
            bake_obj = True

    if bake_obj:
        bake_pass(obj, dest, img_size, "NORMAL", export_usd)


def remove_params(mat, node_tree):
    nodes = node_tree.nodes
    paramDict = {}
    if nodes.get("Material Output"):
        output = nodes["Material Output"]
    elif nodes.get("Group Output"):
        output = nodes["Group Output"]
    else:
        raise ValueError("Could not find material output node")

    if (
        nodes.get("Principled BSDF")
        and output.inputs[0].links[0].from_node.bl_idname == "ShaderNodeBsdfPrincipled"
    ):
        principled_bsdf_node = nodes["Principled BSDF"]
        metal = principled_bsdf_node.inputs[
            "Metallic"
        ].default_value  # store metallic value and set to 0
        sheen = principled_bsdf_node.inputs["Sheen Weight"].default_value
        clearcoat = principled_bsdf_node.inputs["Coat Weight"].default_value
        paramDict[mat.name] = {
            "Metallic": metal,
            "Sheen Weight": sheen,
            "Coat Weight": clearcoat,
        }
        principled_bsdf_node.inputs["Metallic"].default_value = 0
        principled_bsdf_node.inputs["Sheen Weight"].default_value = 0
        principled_bsdf_node.inputs["Coat Weight"].default_value = 0
        return paramDict

    for node in nodes:
        if node.type == "GROUP":
            paramDict = remove_params(mat, node.node_tree)
            if len(paramDict) != 0:
                return paramDict

    return paramDict


def process_interfering_params(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        paramDict = remove_params(mat, mat.node_tree)
    return paramDict


def skipBake(obj):
    if not obj.data.materials:
        logging.info("No material on mesh, skipping...")
        return True

    if len(obj.data.vertices) == 0:
        logging.info("Mesh has no vertices, skipping ...")
        return True

    return False


def triangulate_meshes():
    logging.debug("Triangulating Meshes")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            view_state = obj.hide_viewport
            obj.hide_viewport = False
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            logging.debug(f"Triangulating {obj}")
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.select_set(False)
            obj.hide_viewport = view_state


def adjust_wattages():
    logging.info("Adjusting light wattage")
    for obj in bpy.context.scene.objects:
        if obj.type == "LIGHT" and obj.data.type == "POINT":
            light = obj.data
            if hasattr(light, "energy") and hasattr(light, "shadow_soft_size"):
                X = light.energy
                r = light.shadow_soft_size
                # candelas * 1000 / (4 * math.pi * r**2). additionally units come out of blender at 1/100 scale
                new_wattage = (
                    (X * 20 / (4 * math.pi)) * 1000 / (4 * math.pi * r**2) * 100
                )
                light.energy = new_wattage


def set_center_of_mass():
    logging.info("Resetting center of mass of objects")
    for obj in bpy.context.scene.objects:
        if not obj.hide_render:
            view_state = obj.hide_viewport
            obj.hide_viewport = False
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
            obj.select_set(False)
            obj.hide_viewport = view_state


def bake_object(obj, dest, img_size, export_usd):
    if not uv_unwrap(obj):
        return

    bpy.ops.object.select_all(action="DESELECT")

    with butil.SelectObjects(obj):
        for slot in obj.material_slots:
            mat = slot.material
            if mat is not None:
                slot.material = (
                    mat.copy()
                )  # we duplicate in the case of distinct meshes sharing materials

        process_glass_materials(obj, export_usd)
        bake_metal(obj, dest, img_size, export_usd)
        bake_normals(obj, dest, img_size, export_usd)
        paramDict = process_interfering_params(obj)
        for bake_type in BAKE_TYPES:
            bake_pass(obj, dest, img_size, bake_type, export_usd)

        apply_baked_tex(obj, paramDict)


def bake_scene(folderPath: Path, image_res, vertex_colors, export_usd):
    for obj in bpy.data.objects:
        logging.info("---------------------------")
        logging.info(obj.name)

        if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
            logging.info("Not mesh, skipping ...")
            continue

        if skipBake(obj):
            continue

        if format == "stl":
            continue

        obj.hide_render = False
        obj.hide_viewport = False

        if vertex_colors:
            bakeVertexColors(obj)
        else:
            bake_object(obj, folderPath, image_res, export_usd)

        obj.hide_render = True
        obj.hide_viewport = True


def run_blender_export(
    exportPath: Path, format: str, vertex_colors: bool, individual_export: bool
):
    assert exportPath.parent.exists()
    exportPath = str(exportPath)

    if format == "obj":
        if vertex_colors:
            bpy.ops.wm.obj_export(
                filepath=exportPath,
                export_colors=True,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=individual_export,
            )
        else:
            bpy.ops.wm.obj_export(
                filepath=exportPath,
                path_mode="COPY",
                export_materials=True,
                export_pbr_extensions=True,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=individual_export,
            )

    if format == "fbx":
        if vertex_colors:
            bpy.ops.export_scene.fbx(
                filepath=exportPath, colors_type="SRGB", use_selection=individual_export
            )
        else:
            bpy.ops.export_scene.fbx(
                filepath=exportPath,
                path_mode="COPY",
                embed_textures=True,
                use_selection=individual_export,
            )

    if format == "stl":
        bpy.ops.export_mesh.stl(filepath=exportPath, use_selection=individual_export)

    if format == "ply":
        bpy.ops.wm.ply_export(
            filepath=exportPath, export_selected_objects=individual_export
        )

    if format in ["usda", "usdc"]:
        bpy.ops.wm.usd_export(
            filepath=exportPath,
            export_textures=True,
            # use_instancing=True,
            overwrite_textures=True,
            selected_objects_only=individual_export,
            root_prim_path="/World",
        )


def export_scene(
    input_blend: Path,
    output_folder: Path,
    pipeline_folder=None,
    task_uniqname=None,
    **kwargs,
):
    folder = output_folder / f"export_{input_blend.name}"
    folder.mkdir(exist_ok=True, parents=True)
    export_curr_scene(folder, **kwargs)

    if pipeline_folder is not None and task_uniqname is not None:
        (pipeline_folder / "logs" / f"FINISH_{task_uniqname}").touch()

    return folder


# side effects: will remove parents of inputted obj and clean its name, hides viewport of all objects
def export_single_obj(
    obj: bpy.types.Object,
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
):
    export_usd = format in ["usda", "usdc"]

    export_folder = output_folder
    export_folder.mkdir(exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name

    logging.info(f"Exporting to directory {export_folder=}")

    remove_obj_parents(obj)
    rename_all_meshes(obj)

    collection_views, obj_views = update_visibility()

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
        raise ValueError("Object not mesh")

    if export_usd:
        apply_all_modifiers(obj)
    else:
        realizeInstances(obj)
        apply_all_modifiers(obj)

    if not skipBake(obj) and format != "stl":
        if vertex_colors:
            bakeVertexColors(obj)
        else:
            obj.hide_render = False
            obj.hide_viewport = False
            bake_object(obj, export_folder / "textures", image_res, export_usd)
            obj.hide_render = True
            obj.hide_viewport = True

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    clean_names(obj)

    old_loc = obj.location.copy()
    obj.location = (0, 0, 0)

    if (
        obj.type != "MESH"
        or obj.hide_render
        or len(obj.data.vertices) == 0
        or obj not in list(bpy.context.view_layer.objects)
    ):
        raise ValueError("Object is not mesh or hidden from render")

    export_subfolder = export_folder / obj.name
    export_subfolder.mkdir(exist_ok=True)
    export_file = export_subfolder / f"{obj.name}.{format}"

    logging.info(f"Exporting file to {export_file=}")
    obj.hide_viewport = False
    obj.select_set(True)
    run_blender_export(export_file, format, vertex_colors, individual_export=True)
    obj.select_set(False)
    obj.location = old_loc

    return export_file


@gin.configurable
def export_curr_scene(
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
    individual_export=False,
    omniverse_export=False,
    pipeline_folder=None,
    task_uniqname=None,
) -> Path:
    export_usd = format in ["usda", "usdc"]

    export_folder = output_folder
    export_folder.mkdir(exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name

    logging.info(f"Exporting to directory {export_folder=}")

    remove_obj_parents()
    delete_objects()
    triangulate_meshes()
    if omniverse_export:
        split_glass_mats()
    rename_all_meshes()

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
    #                      continue
    #             logging.info(f"{obj.name} has no faces, removing...")
    #             bpy.data.objects.remove(obj, do_unlink=True)

    collection_views, obj_views = update_visibility()

    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
            continue
        if export_usd:
            apply_all_modifiers(obj)
        else:
            realizeInstances(obj)
            apply_all_modifiers(obj)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    # iterate through all objects and bake them
    bake_scene(
        folderPath=export_folder / "textures",
        image_res=image_res,
        vertex_colors=vertex_colors,
        export_usd=export_usd,
    )

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    clean_names()

    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    if omniverse_export:
        adjust_wattages()
        set_center_of_mass()
        # remove 0 polygon meshes
        for obj in bpy.data.objects:
            if obj.type == "MESH" and len(obj.data.polygons) == 0:
                logging.info(f"{obj.name} has no faces, removing...")
                bpy.data.objects.remove(obj, do_unlink=True)

    if individual_export:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.location_clear()  # send all objects to (0,0,0)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if (
                obj.type != "MESH"
                or obj.hide_render
                or len(obj.data.vertices) == 0
                or obj not in list(bpy.context.view_layer.objects)
            ):
                continue

            obj_name = obj.name.replace('/', '_')
            export_subfolder = export_folder / obj_name
            export_subfolder.mkdir(exist_ok=True, parents=True)
            export_file = export_subfolder / f"{obj_name}.{format}"

            logging.info(f"Exporting file to {export_file=}")
            obj.hide_viewport = False
            obj.select_set(True)
            run_blender_export(export_file, format, vertex_colors, individual_export)
            obj.select_set(False)
    else:
        logging.info(f"Exporting file to {export_file=}")
        run_blender_export(export_file, format, vertex_colors, individual_export)

        return export_file


def main(args):
    args.output_folder.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=args.output_folder / "export_logs.log",
        level=logging.DEBUG,
        filemode="w+",
    )

    targets = sorted(list(args.input_folder.iterdir()))
    for blendfile in targets:
        if blendfile.stem == "solve_state":
            shutil.copy(blendfile, args.output_folder / "solve_state.json")

        if not blendfile.suffix == ".blend":
            print(f"Skipping non-blend file {blendfile}")
            continue

        bpy.ops.wm.open_mainfile(filepath=str(blendfile))

        folder = export_scene(
            blendfile,
            args.output_folder,
            format=args.format,
            image_res=args.resolution,
            vertex_colors=args.vertex_colors,
            individual_export=args.individual,
            omniverse_export=args.omniverse,
        )
        # wanted to use shutil here but kept making corrupted files
        subprocess.call(["zip", "-r", str(folder.with_suffix(".zip")), str(folder)])

    bpy.ops.wm.quit_blender()


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", type=Path)
    parser.add_argument("--output_folder", type=Path)

    parser.add_argument("-f", "--format", type=str, choices=FORMAT_CHOICES)

    parser.add_argument("-v", "--vertex_colors", action="store_true")
    parser.add_argument("-r", "--resolution", default=1024, type=int)
    parser.add_argument("-i", "--individual", action="store_true")
    parser.add_argument("-o", "--omniverse", action="store_true")

    args = parser.parse_args()

    if args.format not in FORMAT_CHOICES:
        raise ValueError("Unsupported or invalid file format.")

    if args.vertex_colors and args.format not in ["ply", "fbx", "obj"]:
        raise ValueError("File format does not support vertex colors.")

    if args.format == "ply" and not args.vertex_colors:
        raise ValueError(".ply export must use vertex colors.")

    return args


if __name__ == "__main__":
    args = make_args()
    main(args)
