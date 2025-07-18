# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import json
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional
from xml.dom.minidom import parseString

import bmesh
import bpy
import numpy as np
import trimesh

import infinigen.core.sim.exporters.utils as exputils
from infinigen.core import surface
from infinigen.core.sim.exporters.base import JointType, PathItem, RigidBody, SimBuilder
from infinigen.core.sim.kinematic_node import (
    KinematicNode,
)
from infinigen.tools.export import export_sim_ready


def create_element(tag: str, **kwargs) -> ET.Element:
    return ET.Element(tag, attrib=kwargs)


class URDFBuilder(SimBuilder):
    def __init__(self, assets_dir):
        super().__init__(assets_dir)

        self.urdf = self._initialize_urdf()

        # create a joint that links the top most link to the world
        self._create_joint(
            name="world_joint",
            joint_type=JointType.WELD,
            origin=np.array([0.0, 0.0, 0.0]),
            parent_link="world",
            child_link="link_0",
        )

        self.asset_freq = defaultdict(int)
        self.joint_freq = defaultdict(int)
        self.joint_map = dict()

        self.link_count = 0

    @property
    def xml(self):
        """Returns the URDF as a string."""
        rough_string = ET.tostring(self.urdf, "utf-8")
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _initialize_urdf(self) -> ET.Element:
        """
        Initializes an URDF file required to construct an asset.
        """
        robot = create_element("robot", name="object")
        world = create_element("link", name="world")
        robot.append(world)

        return robot

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

        joint_params = sample_joint_params_fn()
        self._populate_links(
            root,
            visual_only=visual_only,
            image_res=image_res,
            joint_params=joint_params,
        )

    def _populate_links(
        self,
        root: RigidBody,
        joint_params: Dict,
        parent_link: str = "world",
        joint_nodes: List[KinematicNode] = [],
        pos_offset: np.array = np.zeros(3),
        visual_only: bool = False,
        image_res: int = 512,
    ):
        """Populates the urdf with links and joints."""
        # create a link for the body
        link_name = f"link_{self.link_count}"
        link = create_element("link", name=link_name)
        self.link_count += 1

        vis_origin_refs = []
        col_origin_refs = []
        assets = []
        for asset in root.assets:
            # export the mesh and set the filename
            visasset_path, colasset_paths, mesh = self._get_mesh(
                asset.attribs, visual_only=visual_only, image_res=image_res
            )

            # add all the assets for the given link
            visual = create_element("visual")
            visual_origin = create_element("origin", xyz="0.0 0.0 0.0")
            geometry = create_element("geometry")
            mesh_element = create_element("mesh")

            mesh_element.set("filename", f"assets/{visasset_path.name}")
            geometry.append(mesh_element)
            visual.append(geometry)
            visual.append(visual_origin)

            link.append(visual)

            mat_physics = exputils.get_material_properties(mesh)

            # Estimate the mass of the object given the density
            mesh_temp = mesh.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(mesh_temp)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.transform(mesh.matrix_world)
            vol = bm.calc_volume(signed=False)
            bm.free()

            inertial = create_element("inertial")
            mass = create_element("mass", value=str(mat_physics["density"] * vol))

            t = trimesh.Trimesh(
                vertices=[list(vertex.co) for vertex in mesh.data.vertices],
                faces=[
                    list(triangle.vertices) for triangle in mesh.data.loop_triangles
                ],
            )
            t.mass_properties["density"] = mass / t.volume
            I_tensor = t.moment_inertia
            inertial.append(mass)
            ixx, ixy, ixz = I_tensor[0]
            _,   iyy, iyz = I_tensor[1]
            _,   _,   izz = I_tensor[2]

            inertia = create_element(
                "inertia",
                ixx=str(ixx),
                ixy=str(ixy),
                ixz=str(ixz),
                iyy=str(iyy),
                iyz=str(iyz),
                izz=str(izz),
            )
            inertial.append(inertia)

            com = t.center_mass
            origin = create_element(
                "origin",
                xyz=exputils.array_to_string(com)
            )
            inertial.append(origin)

            link.append(inertial)

            collision_refs = []
            if not visual_only:
                for colasset_path in colasset_paths:
                    collision = create_element("collision")
                    collision_origin = create_element("origin", xyz="0.0 0.0 0.0")
                    geometry = create_element("geometry")
                    mesh_element = create_element("mesh")

                    mesh_element.set("filename", f"assets/{colasset_path.name}")
                    geometry.append(mesh_element)
                    collision.append(geometry)
                    collision.append(collision_origin)
                    link.append(collision)
                    collision_refs.append(collision_origin)

            vis_origin_refs.append(visual_origin)
            col_origin_refs.append(collision_refs)
            assets.append(mesh)

        aabb_center = exputils.get_aabb_center(assets)

        # calculate the absolute joint position
        if len(joint_nodes) > 1:
            raise NotImplementedError(
                "Multi jointed bodies not supported yet in URDF exporter."
            )

        if len(joint_nodes) > 0:
            # add any joint connecting the current link to its parent link
            joint_node = joint_nodes[0]

            joint_name = self.metadata[joint_node.idn]["joint label"]
            unique_joint_name = f"{joint_name}_{self.joint_freq[joint_name]}"
            self.joint_freq[joint_name] += 1

            rel_pos, axis, range_min, range_max = self.get_joint_information(
                joint_node.idn
            )
            abs_joint_pos = aabb_center + rel_pos

            joint_properties = exputils.get_joint_properties(joint_name, joint_params)

            self._create_joint(
                name=unique_joint_name,
                joint_type=joint_node.joint_type,
                origin=abs_joint_pos - pos_offset,
                parent_link=parent_link,
                child_link=link_name,
                min_range=range_min,
                max_range=range_max,
                axis=axis,
                damping=joint_properties["damping"],
                friction=joint_properties["friction"],
            )
            pos_offset = abs_joint_pos

        # set the position of the links geometries relative to the joint
        for vis_origin, col_origins, asset in zip(
            vis_origin_refs, col_origin_refs, assets
        ):
            geom_center = exputils.get_aabb_center(asset)
            offset = geom_center - pos_offset
            vis_origin.set("xyz", exputils.array_to_string(offset))
            for col_origin in col_origins:
                col_origin.set("xyz", exputils.array_to_string(offset))

        for child, joints in root.children.items():
            self._populate_links(
                child,
                joint_params,
                parent_link=link_name,
                joint_nodes=joints,
                pos_offset=pos_offset,
                visual_only=visual_only,
            )

        self.urdf.append(link)

    def _create_joint(
        self,
        name: str,
        joint_type: JointType,
        origin: np.ndarray,
        parent_link: str,
        child_link: str,
        damping: float = 0.0,
        friction: float = 0.0,
        min_range: Optional[float] = -np.pi,
        max_range: Optional[float] = np.pi,
        axis: Optional[np.ndarray] = None,
    ):
        if joint_type == JointType.HINGE:
            jt = "revolute"
        elif joint_type == JointType.SLIDING:
            jt = "prismatic"
        elif joint_type == JointType.WELD or joint_type == JointType.NONE:
            jt = "fixed"
        else:
            raise ValueError("Joint is not valid")

        joint = create_element("joint", name=name, type=jt)
        joint.append(create_element("origin", xyz=exputils.array_to_string(origin)))
        joint.append(create_element("parent", link=parent_link))
        joint.append(create_element("child", link=child_link))
        joint.append(
            create_element("dynamics", damping=str(damping), friction=str(friction))
        )

        if joint_type != JointType.WELD:
            joint.append(create_element("axis", xyz=exputils.array_to_string(axis)))

            if min_range == max_range == 0:
                # default ranges when range undefined
                if joint_type == JointType.HINGE:
                    min_range = -np.pi
                    max_range = np.pi
                elif joint_type == JointType.SLIDING:
                    min_range = -100
                    max_range = 100
            joint.append(
                ET.Element(
                    "limit", attrib={"lower": str(min_range), "upper": str(max_range)}
                )
            )

        self.urdf.append(joint)
        return joint

    def _get_mesh(self, attribs: List[PathItem], visual_only: bool, image_res: int):
        mesh = self._get_geometry(attribs)
        labels = self._get_labels(mesh)
        if len(labels) == 0:
            asset_name = "geom"
        else:
            asset_name = "_".join(list(labels))
        unique_name = f"{asset_name}_{self.asset_freq[asset_name]}"
        self.asset_freq[asset_name] += 1

        # export the asset
        geometry_center = exputils.get_aabb_center(mesh)
        export_paths = export_sim_ready(
            mesh,
            output_folder=self.assets_dir,
            image_res=image_res,
            translation=-geometry_center,
            separate_asset_dirs=False,
            name=unique_name,
            visual_only=visual_only,
        )

        visasset_path = export_paths["visual"][0]
        colasset_paths = export_paths["collision"]

        return visasset_path, colasset_paths, mesh

    def get_joint_information(self, joint_name: str):
        pos_vals = surface.read_attr_data(self.blend_obj, joint_name + "_pos")
        pos_mask = np.any(pos_vals != 0.0, axis=1)
        if all(~pos_mask):
            position = np.zeros(3)
        else:
            position = pos_vals[pos_mask].mean(axis=0)

        axis_vals = surface.read_attr_data(self.blend_obj, joint_name + "_axis")
        axis_mask = np.any(axis_vals != 0.0, axis=1)
        if all(~axis_mask):
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis_vals[axis_mask].mean(axis=0)

        min_vals = surface.read_attr_data(self.blend_obj, joint_name + "_min")
        min_mask = min_vals != 0.0
        if all(~min_mask):
            range_min = 0.0
        else:
            range_min = min_vals[min_mask].mean()

        max_vals = surface.read_attr_data(self.blend_obj, joint_name + "_max")
        max_mask = max_vals != 0.0
        if all(~max_mask):
            range_max = 0.0
        else:
            range_max = max_vals[max_mask].mean()

        return position, axis, range_min, range_max


def export(
    blend_obj: bpy.types.Object,
    sim_blueprint: Dict,
    seed: int,
    sample_joint_params_fn: Callable,
    export_dir: Path = Path("./sim_exports/urdf"),
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
    builder = URDFBuilder(obj_assets_dir)
    builder.build(
        blend_obj=blend_obj,
        kinematic_root=kinematic_root,
        sample_joint_params_fn=sample_joint_params_fn,
        metadata=metadata,
        visual_only=visual_only,
        image_res=image_res,
    )

    metadata.update(builder.get_bounding_box_info())

    # save the urdf
    urdf_path, metadata_path = save(
        fname=asset_name,
        export_dir=obj_export_dir,
        contents=builder.urdf,
        metadata=metadata,
    )

    return urdf_path, metadata_path


def save(fname: str, export_dir: Path, contents: ET.Element, metadata: Dict) -> None:
    """Save the URDF contents."""
    urdf_path = export_dir / f"{fname}.urdf"
    with open(urdf_path, "w") as f:
        raw_xml = ET.tostring(contents, encoding="unicode")
        formatted_xml = parseString(raw_xml).toprettyxml(indent="  ")
        lines = [line for line in formatted_xml.splitlines() if line.strip()]
        f.write("\n".join(lines))

    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return urdf_path, metadata_path
