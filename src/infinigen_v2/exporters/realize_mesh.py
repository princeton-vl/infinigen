import bpy

# TODO: remove infinigen dependency
# ruff: noqa: TID251
from infinigen.core.nodes.node_wrangler import (
    geometry_node_group_empty_new,
)


def bake_displacement_to_vcols(obj, vcol_name="Displacement"):
    # Cycles fast bake
    scn = bpy.context.scene
    old_settings = {
        "engine": scn.render.engine,
        "device": scn.cycles.device,
        "samples": scn.cycles.samples,
        "view": scn.view_settings.view_transform,
    }
    scn.render.engine = "CYCLES"
    scn.cycles.device = "GPU"
    scn.cycles.samples = 1
    scn.view_settings.view_transform = "Standard"

    # Ensure vertex color attribute
    if vcol_name not in obj.data.color_attributes:
        obj.data.color_attributes.new(
            name=vcol_name, domain="POINT", type="FLOAT_COLOR"
        )
    obj.data.attributes.active_color = obj.data.color_attributes[vcol_name]

    # Patch materials: Surface <- Emission(displacement)
    patched = []
    for slot in obj.material_slots:
        mat = slot.material
        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links

        out = nodes["Material Output"]

        disp_input = out.inputs["Displacement"]
        if not disp_input.is_linked:
            continue

        disp_link = disp_input.links[0]
        disp_source_socket = disp_link.from_socket

        # Save original Surface link
        surf_link = out.inputs["Surface"].links[0]
        orig_from = surf_link.from_socket
        links.remove(surf_link)

        # Temp emission: displacement -> add(0.5) -> emission -> surface
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        add.name = "__TMP_ADD__"

        emit = nodes.new("ShaderNodeEmission")
        emit.name = "__TMP_EMIT__"

        links.new(disp_source_socket, add.inputs[0])
        links.new(add.outputs["Vector"], emit.inputs["Color"])
        links.new(emit.outputs["Emission"], out.inputs["Surface"])

        patched.append((mat, orig_from, emit, add))

    # Bake
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", target="VERTEX_COLORS")
    obj.select_set(False)

    # Restore materials
    for mat, orig_from, emit, add in patched:
        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links
        out = nodes["Material Output"]

        links.remove(out.inputs["Surface"].links[0])  # tmp emit link
        links.remove(out.inputs["Displacement"].links[0])
        nodes.remove(emit)
        nodes.remove(add)
        links.new(orig_from, out.inputs["Surface"])

    scn.render.engine = old_settings["engine"]
    scn.cycles.device = old_settings["device"]
    scn.cycles.samples = old_settings["samples"]
    scn.view_settings.view_transform = old_settings["view"]

    return


def add_geo_displacement(obj, scale_val=1.0, vcol_name="Displacement", apply=True):
    mod = obj.modifiers.new("Displacement", "NODES")
    mod.node_group = geometry_node_group_empty_new()
    ng = mod.node_group
    nodes, links = ng.nodes, ng.links

    group_in = nodes["Group Input"]
    group_out = nodes["Group Output"]

    nodes.new("GeometryNodeInputNormal")

    attr = nodes.new("GeometryNodeInputNamedAttribute")
    attr.data_type = "FLOAT_COLOR"
    attr.inputs[0].default_value = vcol_name

    sub = nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.5)  # SUBTRACT OFFSET

    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = scale_val

    set_pos = nodes.new("GeometryNodeSetPosition")

    links.new(group_in.outputs["Geometry"], set_pos.inputs["Geometry"])

    # Logic: Attribute -> Subtract(0.5) -> Scale -> Offset
    links.new(attr.outputs[0], sub.inputs[0])
    links.new(sub.outputs[0], scale.inputs["Vector"])
    links.new(scale.outputs["Vector"], set_pos.inputs["Offset"])

    links.new(set_pos.outputs["Geometry"], group_out.inputs["Geometry"])

    if apply:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)


def convert_shader_displacement(obj, scale_val=1.0, apply_geo_modifier=False):
    bake_displacement_to_vcols(obj, vcol_name="Displacement")
    add_geo_displacement(
        obj, scale_val=scale_val, vcol_name="Displacement", apply=apply_geo_modifier
    )


def realize_scene():
    """
    Realizes entire scene, potentially expensive ; could be modified to on realize only in view.
    """
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data:
            continue

        has_displacement = False
        for slot in obj.material_slots:
            if not slot.material or not slot.material.node_tree:
                continue
            nodes = slot.material.node_tree.nodes
            if "Material Output" not in nodes:
                continue
            out = nodes["Material Output"]
            if out.inputs["Displacement"].is_linked:
                has_displacement = True
                break

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for mod in list(obj.modifiers):
            if mod.type == "SUBSURF":
                bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)

        if has_displacement:
            convert_shader_displacement(obj)
