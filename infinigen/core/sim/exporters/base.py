import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Union

import bpy
import mathutils
import numpy as np
import trimesh

import infinigen.core.sim.exporters.utils as exputils
from infinigen.core import surface, tagging
from infinigen.core.sim.kinematic_node import JointType, KinematicNode, KinematicType
from infinigen.core.util import blender as butil
from infinigen.tools.export import triangulate_mesh


@dataclass
class PathItem:
    """Class for keeping track of an item in inventory."""

    node: KinematicNode
    value: int

    def __repr__(self):
        if self.node.kinematic_type == KinematicType.JOINT:
            return f"{self.node.joint_type}-{self.node.idn}-{self.value}"
        else:
            return f"{self.node.kinematic_type}-{self.node.idn}-{self.value}"


class RigidAsset:
    """Represents an assets in the rigid body skeleton."""

    def __init__(self, attribs: List[PathItem] = []):
        self.attribs = attribs.copy()


class RigidBody:
    """Represents a bost in the rigid body skeleton."""

    def __init__(self):
        self.assets = []
        self.children = defaultdict(list)  # mapping from children rigid bodies to
        # joint kinematic nodes

    def __iter__(self):
        """
        Custom iteration function to go through all bodies under the current
        rigid body.
        """
        yield self
        for child in self.children.keys():
            yield from child

    def __str__(self, level=0):
        indent = "    " * level
        s = f"{indent}RigidBody(\n"
        s += f"{indent}  Assets: [\n"
        for asset in self.assets:
            asset_str = ", ".join(
                [f"{item.node.idn}={item.value}" for item in asset.attribs]
            )
            s += f"{indent}    RigidAsset({asset_str})\n"
        s += f"{indent}  ]\n"
        s += f"{indent}  Children:\n"
        for child, joints in self.children.items():
            joint_str = ", ".join([f"{j.idn}({j.joint_type})" for j in joints])
            s += f"{indent}    Joint(s): [{joint_str}]\n"
            s += child.__str__(level + 2)
        s += f"{indent})\n"
        return s


class SimBuilder:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir

        self.blend_obj = None  # set when building asset

    def build(self, blend_obj: bpy.types.Object, metadata: Dict):
        self.blend_obj = blend_obj
        self.metadata = metadata

        # in general, we want to only deal with faces with 3 vertices
        triangulate_mesh(self.blend_obj)

    def _construct_rigid_body_skeleton(
        self, node: KinematicNode, path: List[PathItem] = []
    ):
        """
        Given the blender object and the kinematic root, construct a rigid
        body skeleton that represents the articulated object.
        """
        dispatch_table = {
            (KinematicType.JOINT, JointType.WELD): self._handle_weld_node,
            (KinematicType.JOINT, JointType.HINGE): self._handle_joint_node,
            (KinematicType.JOINT, JointType.SLIDING): self._handle_joint_node,
            (KinematicType.SWITCH, JointType.NONE): self._handle_switch_node,
            (KinematicType.DUPLICATE, JointType.NONE): self._handle_duplicate_node,
            (KinematicType.ASSET, JointType.NONE): self._handle_asset_node,
        }
        key = (node.kinematic_type, node.joint_type)
        handler = dispatch_table.get(key)
        if handler is None:
            children = node.get_all_children()
            assert len(children) == 1
            return self._construct_rigid_body_skeleton(children[0])
        else:
            return handler(node, path)

    def _handle_weld_node(self, node, path):
        body = RigidBody()
        for i, c in node.children.items():
            child, _ = self._construct_rigid_body_skeleton(
                c, path + [PathItem(node=node, value=i)]
            )
            if isinstance(child, RigidBody):
                body.children[child].append(node)
            elif isinstance(child, RigidAsset):
                body.assets.append(child)

        return body, None

    def _handle_joint_node(self, node, path):
        new_path_parent, new_path_child = path, path
        if not (
            node.children[0] == node.children[1]
            and node.children[0].kinematic_type == KinematicType.JOINT
        ):
            new_path_parent = path + [PathItem(node=node, value=0)]
            new_path_child = path + [PathItem(node=node, value=1)]

        # get the parent rigid body
        parent, parents_child = self._construct_rigid_body_skeleton(
            node.children[0], new_path_parent
        )
        parent_body = self._wrap_in_body(parent)

        # handle case of multiple joints between two parts
        if (
            node.children[0] == node.children[1]
            and node.children[0].kinematic_type == KinematicType.JOINT
        ):
            parent.children[parents_child].append(node)
            return parent, parents_child

        # get the child rigid body
        child, _ = self._construct_rigid_body_skeleton(node.children[1], new_path_child)
        child_body = self._wrap_in_body(child)

        parent_body.children[child_body].append(node)
        return parent_body, child_body

    def _handle_switch_node(self, node, path):
        # incoming geometry is not provided
        if node.idn not in self.blend_obj.data.attributes:
            return RigidBody(), None

        # gets the actual value passed into the switch and returns the
        # coressponding geometry
        switch_value = self._get_switch_values(node.idn, path)
        return self._construct_rigid_body_skeleton(
            node.children[switch_value],
            path + [PathItem(node=node, value=switch_value)],
        )

    def _handle_duplicate_node(self, node, path):
        unique = self._get_unique_values(node.idn, path)
        parent, child = self._construct_rigid_body_skeleton(node.children[0], path)

        # remove the original instance of the child
        joints = parent.children[child]
        del parent.children[child]

        # for each unique instance of body, create a new child element
        for i in range(1, unique):
            unique_body = deepcopy(child)
            # add duplicate node to path for all assets in child
            for body in unique_body:
                for asset in body.assets:
                    asset.attribs.append(PathItem(node=node, value=i))

            parent.children[unique_body] = joints

        return parent, None

    def _handle_asset_node(self, node, path):
        return RigidAsset(attribs=path), None

    def _wrap_in_body(self, element: Union[RigidAsset, RigidBody]):
        """If elements is a singular assets, wrap it in its own rigid body."""
        body = element
        if isinstance(element, RigidAsset):
            body = RigidBody()
            body.assets.append(element)
        return body

    def _simplify_skeleton(self, root: RigidBody):
        for child_body, kinematic_node in root.children.items():
            self._simplify_skeleton(child_body)

        to_add = []
        to_delete = []

        for child_body, kinematic_node in root.children.items():
            if (
                len(kinematic_node) == 1
                and kinematic_node[0].joint_type == JointType.WELD
            ):
                # copy all assets from child body into current body
                root.assets.extend(child_body.assets)
                to_add.append(child_body.children)

                child_body.assets = []
                to_delete.append(child_body)

        for children in to_add:
            root.children.update(children)

        for child_body in to_delete:
            del root.children[child_body]

    def _get_geometry(
        self, attribs: List[PathItem], center_at_origin: bool = False
    ) -> bpy.types.Object:
        """
        Gets the geometry of a part of the whole asset based on the path.
        If center_at_origin is true, centers the objects such thats is axis
        aligned bounding box is at the origin.
        """
        vertex_mask = np.ones(len(self.blend_obj.data.vertices), dtype=bool)
        for item in attribs:
            attr = item.node.idn
            # handles switch cases where certain joints may not be a
            # part of the final geometry
            if attr not in self.blend_obj.data.attributes:
                continue
            data = surface.read_attr_data(self.blend_obj, attr)
            vertex_mask = vertex_mask & (data == item.value)

        # extract the mesh based on the vertex mask
        obj_clone = butil.deep_clone_obj(
            self.blend_obj, keep_modifiers=True, keep_materials=True
        )
        mesh_obj = tagging.extract_vertex_mask(obj_clone, vertex_mask)

        if center_at_origin:
            translation = mathutils.Vector(-exputils.get_aabb_center(mesh_obj))
            for v in mesh_obj.data.vertices:
                v.co += translation
        butil.delete(obj_clone)
        return mesh_obj

    def _get_labels(self, geometry: bpy.types.Object) -> Set[str]:
        """Gets the labels associated with a geometry."""
        geom_labels = set()
        for label in self.metadata["part_labels"]:
            if label not in set([n.name for n in geometry.data.attributes]):
                logging.warning(
                    f" Label {label} not found, skipping for now. It is likely this is an empty geometry."
                )
                continue
            data = surface.read_attr_data(geometry, label)
            label_instance = np.mean(data)
            if label_instance > 0:
                geom_labels.add(label)
        return geom_labels

    def _get_subsets(self, path: List[PathItem]) -> int:
        mask = None
        for item in path:
            path_attr_name, path_attr_value = item.node.idn, item.value
            path_attr_data = surface.read_attr_data(self.blend_obj, path_attr_name)
            path_attr_mask = path_attr_data == int(path_attr_value)

            if mask is None:
                mask = path_attr_mask
            else:
                mask = np.logical_and(mask, path_attr_mask)
        return mask

    def _get_unique_values(self, attr: str, path: List[PathItem]) -> int:
        """Gets unique values for a particular attribute."""
        if attr not in self.blend_obj.data.attributes:
            logging.warning(
                f" Attribute {attr} not found, skipping for now. It is likely this is an empty geometry."
            )
            return 0
        data = surface.read_attr_data(self.blend_obj, attr)
        mask = self._get_subsets(path)
        data = data[mask]
        return len(np.unique(data))

    def _get_switch_values(self, attr: str, path: List[PathItem]) -> int:
        """Gets the actual value of a switch attribute."""
        data = surface.read_attr_data(self.blend_obj, attr)
        mask = self._get_subsets(path)
        data = data[mask]
        indices = np.nonzero(data)[0]

        switch_value = 0
        # if there exists a non-zero value, switch is true
        if len(indices) > 0:
            switch_value = int(np.mean(data[indices]))
        return switch_value

    def get_bounding_box_info(self):
        # get bounding box information for the asset
        mesh = trimesh.Trimesh(
            vertices=[list(vertex.co) for vertex in self.blend_obj.data.vertices],
            faces=[
                list(triangle.vertices)
                for triangle in self.blend_obj.data.loop_triangles
            ],
        )
        vertices = mesh.vertices
        min_corner = np.min(vertices, axis=0)
        max_corner = np.max(vertices, axis=0)
        bounding_box_info = {
            "bounding_box": {"min": min_corner.tolist(), "max": max_corner.tolist()}
        }

        return bounding_box_info
