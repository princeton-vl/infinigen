# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import json
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List
from xml.dom.minidom import parseString

import bpy
import numpy as np

import infinigen.core.sim.exporters.utils as exputils
from infinigen.core import surface
from infinigen.core.sim.exporters.base import JointType, PathItem, RigidBody, SimBuilder
from infinigen.core.sim.kinematic_node import (
    KinematicNode,
)
from infinigen.core.sim.physics import joint_dynamics as jointdyna
from infinigen.core.sim.physics import material_physics as mtlphysics
from infinigen.tools.export import export_sim_ready, skipBake


def create_element(tag: str, **kwargs) -> ET.Element:
    return ET.Element(tag, attrib=kwargs)


class MJCFBuilder(SimBuilder):
    def __init__(self, assets_dir):
        super().__init__(assets_dir)

        self.mujoco = self._initialize_mjcf()

        self.asset_freq = defaultdict(int)
        self.joint_freq = defaultdict(int)
        self.joint_map = dict()

        self.link_count = 0

    @property
    def xml(self):
        """Returns the MJCF as a string."""
        rough_string = ET.tostring(self.mujoco, "utf-8")
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _initialize_mjcf(self) -> ET.Element:
        """
        Initializes an MJCF file required to construct an asset.
        """
        mujoco = create_element("mujoco")

        # adding compiler attributes
        compiler = create_element("compiler", angle="radian", meshdir="assets")
        mujoco.append(compiler)

        # creating general defaults
        default = create_element("default")
        geom_default = create_element("geom", rgba="1 1 1 1")
        default.append(geom_default)
        mujoco.append(default)

        # create root level for the worldbody and the asset
        self.asset = create_element("asset")
        self.worldbody = create_element("worldbody")
        self.main_body = create_element("body", name="object")
        self.worldbody.append(self.main_body)
        self.contact = create_element("contact")
        mujoco.append(self.asset)
        mujoco.append(self.worldbody)
        mujoco.append(self.contact)

        return mujoco

    def build(
        self,
        blend_obj: bpy.types.Object,
        kinematic_root: KinematicNode,
        sample_joint_params_fn: Callable,
        metadata: Dict,
        visual_only: bool = False,
        image_res: int = 512,
    ):
        super().build(blend_obj, metadata)

        # construct a skeleton for the rigid body
        root, _ = self._construct_rigid_body_skeleton(kinematic_root)
        self._simplify_skeleton(root)

        asset_body, _ = self._populate_mjcf(
            root, visual_only=visual_only, image_res=image_res
        )
        self._populate_joints(asset_body, sample_joint_params_fn)
        self.main_body.append(asset_body)

        self._sort_asset_elements(self.asset)

    def _populate_mjcf(
        self,
        root: RigidBody,
        joint_nodes: List[KinematicNode] = [],
        pos_offset: np.array = np.zeros(3),
        visual_only: bool = False,
        image_res: int = 512,
    ):
        """Populates the mjcf with assets and joints."""
        link_name = f"link_{self.link_count}"
        link = create_element("body", name=link_name)
        self.link_count += 1

        # add all the assets for the body
        visgeom_refs = []
        colgeom_refs = []
        assets = []
        for asset in root.assets:
            visgeom, colgeoms, asset = self._add_mesh(
                asset.attribs, link, visual_only, image_res
            )
            visgeom_refs.append(visgeom)
            colgeom_refs.append(colgeoms)
            assets.append(asset)

        aabb_center = exputils.get_aabb_center(assets)
        link.set("pos", exputils.array_to_string(aabb_center - pos_offset))

        for visgeom, colgeoms, asset in zip(visgeom_refs, colgeom_refs, assets):
            geom_center = exputils.get_aabb_center(asset)
            offset = geom_center - aabb_center
            visgeom.set("pos", exputils.array_to_string(offset))
            for colgeom in colgeoms:
                colgeom.set("pos", exputils.array_to_string(offset))

        # add joints to the body if they exist
        if joint_nodes:
            for node in joint_nodes:
                joint_name = self.metadata[node.idn]["joint label"]
                unique_joint_name = f"{joint_name}_{self.joint_freq[joint_name]}"
                self.joint_freq[joint_name] += 1
                joint_type = "hinge" if node.joint_type == JointType.HINGE else "slide"
                joint = create_element(
                    "joint",
                    name=unique_joint_name,
                    type=joint_type,
                    pos="0 0 0",
                    axis="0 0 1",
                )
                self.joint_map[unique_joint_name] = node.idn
                link.append(joint)

        # add all children bodies
        for child, joints in root.children.items():
            child_link, child_name = self._populate_mjcf(
                child,
                joint_nodes=joints,
                pos_offset=aabb_center,
                visual_only=visual_only,
            )
            link.append(child_link)

            # exclude contacts between parent and child bodies
            self.contact.append(
                create_element("exclude", body1=link_name, body2=child_name)
            )

        return link, link_name

    def _add_mesh(
        self,
        attribs: List[PathItem],
        body: ET.Element,
        visual_only: bool,
        image_res: int,
    ):
        asset = self._get_geometry(attribs)

        labels = self._get_labels(asset)
        if len(labels) == 0:
            asset_name = "geom"
        else:
            asset_name = "_".join(list(labels))
        unique_name = f"{asset_name}_{self.asset_freq[asset_name]}"
        self.asset_freq[asset_name] += 1

        # export the asset
        geometry_center = exputils.get_aabb_center(asset)
        export_paths = export_sim_ready(
            asset,
            output_folder=self.assets_dir,
            image_res=image_res,
            translation=-geometry_center,
            name=unique_name,
            visual_only=visual_only,
        )

        # add the visual asset to the list of assets in the scene
        visasset_path = export_paths["visual"][0]
        self._add_asset(
            asset_name=unique_name,
            asset_path=visasset_path,
            asset_type="visual",
            has_material=not skipBake(asset),
        )

        # getting material physical properties
        mat_physics = mtlphysics.get_material_properties(asset)

        # create and link a geom for the asset
        visgeom = create_element(
            "geom",
            name=unique_name,
            type="mesh",
            mesh=unique_name,
            group="1",
            contype="0",
            conaffinity="0",
            friction=f"{mat_physics['friction']} 0.005 0.0001",
            density=f"{mat_physics['density']}",
        )
        if not skipBake(asset):
            visgeom.set("material", f"{unique_name}_mat")
        body.append(visgeom)

        colgeoms = []
        if not visual_only:
            # add the collision asset to the list of assets in the scene
            for colasset_path in export_paths["collision"]:
                colasset_name = colasset_path.stem
                self._add_asset(
                    asset_name=colasset_name,
                    asset_path=colasset_path,
                    asset_type="collision",
                    has_material=False,
                )

                # create and link a geom for the asset
                colgeom = create_element(
                    "geom",
                    name=colasset_name,
                    type="mesh",
                    mesh=colasset_name,
                    group="0",
                    contype="1",
                    conaffinity="1",
                    friction=f"{mat_physics['friction']} 0.005 0.0001",
                    density=f"{mat_physics['density']}",
                )
                body.append(colgeom)
                colgeoms.append(colgeom)

        return visgeom, colgeoms, asset

    def _add_asset(
        self, asset_name: str, asset_path: Path, asset_type: str, has_material: bool
    ):
        """Adds a mesh along with its materials and texture to the mjcf."""
        mesh_element = create_element(
            "mesh", name=asset_name, file=str(f"{asset_type}/{asset_path.name}")
        )
        self.asset.append(mesh_element)

        # add a material if it exists for the part
        if has_material:
            texture_element = create_element(
                "texture",
                name=f"{asset_name}_tex",
                type="2d",
                file=f"assets/textures/{asset_name}_DIFFUSE.png",
            )
            material_element = create_element(
                "material", name=f"{asset_name}_mat", texture=f"{asset_name}_tex"
            )
            self.asset.append(texture_element)
            self.asset.append(material_element)

    def _populate_joints(self, body: ET.Element, sample_joint_params_fn: Callable):
        """
        Populates all the joints with the true value given the object.
        """
        # sample the physics distribution for the joints
        joint_params = sample_joint_params_fn()

        for joint in body.findall(".//joint"):
            # set the position and axis of the joints
            joint_name = joint.get("name")
            prefix = self.joint_map[joint_name]

            pos_vals = surface.read_attr_data(self.blend_obj, prefix + "_pos")
            pos_mask = np.any(pos_vals != 0.0, axis=1)
            if all(~pos_mask):
                position = np.zeros(3)
            else:
                position = pos_vals[pos_mask].mean(axis=0)
            joint.set("pos", exputils.array_to_string(position))
            axis_vals = surface.read_attr_data(self.blend_obj, prefix + "_axis")
            axis_mask = np.any(axis_vals != 0.0, axis=1)
            if all(~axis_mask):
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis_vals[axis_mask].mean(axis=0)
            joint.set("axis", exputils.array_to_string(axis))

            min_vals = surface.read_attr_data(self.blend_obj, prefix + "_min")
            min_mask = min_vals != 0.0
            if all(~min_mask):
                range_min = 0.0
            else:
                range_min = min_vals[min_mask].mean()

            max_vals = surface.read_attr_data(self.blend_obj, prefix + "_max")
            max_mask = max_vals != 0.0
            if all(~max_mask):
                range_max = 0.0
            else:
                range_max = max_vals[max_mask].mean()

            if not (np.isclose(range_max, 0.0) and np.isclose(range_min, 0.0)):
                joint.set("limited", "true")
                joint.set("range", f"{range_min} {range_max}")

            ref = 0.0
            if 0 < range_min:
                ref = range_min
            if range_max < 0:
                ref = range_max
            joint.set("ref", f"{ref}")

            # set joint physics parameters
            nonunique_joint_name = re.sub(r"_\d+$", "", joint_name)
            joint_properties = jointdyna.get_joint_properties(
                nonunique_joint_name, joint_params
            )

            if nonunique_joint_name in joint_params:
                joint.set("stiffness", str(joint_properties["stiffness"]))
                joint.set("damping", str(joint_properties["damping"]))
                joint.set("frictionloss", str(joint_properties["friction"]))

    def _sort_asset_elements(self, asset: ET.Element):
        mesh_elements = []
        material_elements = []
        texture_elements = []
        other_elements = []

        for child in list(asset):
            if child.tag == "mesh":
                mesh_elements.append(child)
            elif child.tag == "material":
                material_elements.append(child)
            elif child.tag == "texture":
                texture_elements.append(child)
            else:
                other_elements.append(child)

        # Clear the current children
        asset.clear()

        # Re-add in the desired order
        for elem in (
            mesh_elements + material_elements + texture_elements + other_elements
        ):
            asset.append(elem)


def export(
    blend_obj: bpy.types.Object,
    sim_blueprint: Dict,
    seed: int,
    sample_joint_params_fn: Callable,
    export_dir: Path = Path("./sim_exports/mjcf"),
    image_res: int = 512,
    visual_only: bool = True,
    **kwargs,
):
    """Export function for the MJCF file format."""
    # parse the provided blueprint and set the object export directory
    asset_name, kinematic_root, metadata = exputils.parse_sim_blueprint(sim_blueprint)

    # create export directories
    obj_export_dir = export_dir / asset_name / str(seed)
    obj_assets_dir = obj_export_dir / "assets"
    obj_export_dir.mkdir(parents=True, exist_ok=True)
    obj_assets_dir.mkdir(parents=True, exist_ok=True)

    # build asset
    builder = MJCFBuilder(obj_assets_dir)
    builder.build(
        blend_obj=blend_obj,
        kinematic_root=kinematic_root,
        sample_joint_params_fn=sample_joint_params_fn,
        metadata=metadata,
        visual_only=visual_only,
        image_res=image_res,
    )

    metadata.update(builder.get_bounding_box_info())

    # save the mjcf
    mjcf_path, metadata_path = save(
        fname=asset_name,
        export_dir=obj_export_dir,
        contents=builder.mujoco,
        metadata=metadata,
    )

    return mjcf_path, metadata_path


def save(fname: str, export_dir: Path, contents: ET.Element, metadata: Dict) -> None:
    """Save the MJCF contents."""
    mjcf_path = export_dir / f"{fname}.xml"
    with open(mjcf_path, "w") as f:
        raw_xml = ET.tostring(contents, encoding="unicode")
        formatted_xml = parseString(raw_xml).toprettyxml(indent="  ")
        lines = [line for line in formatted_xml.splitlines() if line.strip()]
        f.write("\n".join(lines))

    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return mjcf_path, metadata_path
