# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import logging
from dataclasses import dataclass
from typing import Dict, List

import bpy

from infinigen2.exporters.util.format import ExportType, RenderPass

__all__ = [
    "configure_materials_for_aovs",
    "setup_eevee_aovs",
]

logger = logging.getLogger(__name__)


@dataclass
class _AOVConfig:
    name: str
    type: str  # 'COLOR' or 'VALUE'
    export_type: ExportType


AOV_CONFIGS = {
    ExportType.OBJECT_INDEX: _AOVConfig(
        name="object_index", type="VALUE", export_type=ExportType.OBJECT_INDEX
    ),
    ExportType.MATERIAL_INDEX: _AOVConfig(
        name="material_index", type="VALUE", export_type=ExportType.MATERIAL_INDEX
    ),
}


def setup_eevee_aovs(
    view_layer: bpy.types.ViewLayer,
    render_passes: List[RenderPass],
) -> Dict[ExportType, str]:
    """Setup AOVs in the view layer for the requested render passes.

    Returns mapping from ExportType to AOV name that was created.
    """
    aov_mapping = {}

    for render_pass in render_passes:
        if render_pass.type in AOV_CONFIGS:
            aov_config = AOV_CONFIGS[render_pass.type]

            # Check if AOV already exists
            existing_aov = None
            for aov in view_layer.aovs:
                if aov.name == aov_config.name:
                    existing_aov = aov
                    break

            if not existing_aov:
                aov = view_layer.aovs.add()
                aov.name = aov_config.name
                aov.type = aov_config.type
                logger.info(f"Created AOV {aov_config.name} of type {aov_config.type}")
            else:
                logger.info(f"Using existing AOV {aov_config.name}")

            aov_mapping[render_pass.type] = aov_config.name

    return aov_mapping


def _setup_object_attributes_for_aovs(
    aov_mapping: Dict[ExportType, str],
):
    """Add custom attributes to objects for their pass indices."""

    if ExportType.OBJECT_INDEX in aov_mapping:
        # Add pass_index as a custom attribute to all mesh objects
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                # Ensure the mesh has the attribute
                if "pass_index" not in obj.data.attributes:
                    attr = obj.data.attributes.new(
                        name="pass_index",
                        type="INT",
                        domain="POINT",  # Use vertex domain instead of face
                    )
                else:
                    attr = obj.data.attributes["pass_index"]

                # Set all vertices to have the object's pass_index
                attr.data.foreach_set(
                    "value", [obj.pass_index] * len(obj.data.vertices)
                )

                # Force update
                obj.data.update()

                logger.debug(
                    f"Set pass_index attribute for {obj.name} to {obj.pass_index}"
                )


def configure_materials_for_aovs(
    aov_mapping: Dict[ExportType, str],
):
    """Add AOV output nodes to all materials for object and material indices."""

    object_index_aov = aov_mapping.get(ExportType.OBJECT_INDEX)
    material_index_aov = aov_mapping.get(ExportType.MATERIAL_INDEX)

    if not object_index_aov and not material_index_aov:
        return

    # First setup object attributes if needed
    if object_index_aov:
        _setup_object_attributes_for_aovs(aov_mapping)

    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            _configure_material_aov_outputs(
                mat,
                object_index_aov,
                material_index_aov,
            )


def _configure_material_aov_outputs(
    material: bpy.types.Material,
    object_index_aov: str = None,
    material_index_aov: str = None,
):
    """Add AOV output nodes to a single material."""

    node_tree = material.node_tree

    # Find material output node
    material_output = None
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            material_output = node
            break

    if not material_output:
        return

    # Position for new AOV nodes
    x_offset = -300
    y_offset = -200
    y_spacing = -150

    if object_index_aov:
        # Check if already exists
        existing = None
        for node in node_tree.nodes:
            if (
                node.type == "OUTPUT_AOV"
                and hasattr(node, "aov_name")
                and node.aov_name == object_index_aov
            ):
                existing = node
                break

        if not existing:
            # Create attribute node to read pass_index
            attr_node = node_tree.nodes.new("ShaderNodeAttribute")
            attr_node.attribute_name = "pass_index"
            attr_node.location = (
                material_output.location[0] + x_offset - 200,
                material_output.location[1] + y_offset,
            )

            # Create AOV output node directly
            aov_node = node_tree.nodes.new("ShaderNodeOutputAOV")
            aov_node.name = f"AOV_{object_index_aov}"
            aov_node.aov_name = object_index_aov
            aov_node.location = (
                material_output.location[0] + x_offset,
                material_output.location[1] + y_offset,
            )

            # Link attribute value to AOV value input
            node_tree.links.new(attr_node.outputs["Fac"], aov_node.inputs["Value"])

            logger.debug(f"Added object index AOV to material {material.name}")

        y_offset += y_spacing

    if material_index_aov:
        # Check if already exists
        existing = None
        for node in node_tree.nodes:
            if (
                node.type == "OUTPUT_AOV"
                and hasattr(node, "aov_name")
                and node.aov_name == material_index_aov
            ):
                existing = node
                break

        if not existing:
            # Create value node directly with material pass index
            value_node = node_tree.nodes.new("ShaderNodeValue")
            value_node.outputs[0].default_value = float(material.pass_index)
            value_node.location = (
                material_output.location[0] + x_offset - 200,
                material_output.location[1] + y_offset,
            )

            # Create AOV output node directly
            aov_node = node_tree.nodes.new("ShaderNodeOutputAOV")
            aov_node.name = f"AOV_{material_index_aov}"
            aov_node.aov_name = material_index_aov
            aov_node.location = (
                material_output.location[0] + x_offset,
                material_output.location[1] + y_offset,
            )

            # Link value to AOV value input
            node_tree.links.new(value_node.outputs["Value"], aov_node.inputs["Value"])

            logger.debug(f"Added material index AOV to material {material.name}")
